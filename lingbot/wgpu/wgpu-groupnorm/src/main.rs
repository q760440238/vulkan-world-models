use bytemuck::{Pod, Zeroable};
use std::borrow::Cow;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Params { n: u32, c: u32, d: u32, h: u32, w: u32, groups: u32, eps: f32, _p: u32 }

const WGSL: &str = r#"
struct Params { n: u32, c: u32, d: u32, h: u32, w: u32, groups: u32, eps: f32, p: u32 };
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> gamma: array<f32>;
@group(0) @binding(2) var<storage, read> beta: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@group(0) @binding(4) var<uniform> p: Params;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total_groups = p.n * p.groups;
    if (gid.x >= total_groups) { return; }
    let bn = gid.x / p.groups;
    let g = gid.x % p.groups;
    let cg = p.c / p.groups;
    let c0 = g * cg;
    let spatial = p.d * p.h * p.w;
    let count = cg * spatial;

    var sum = 0.0;
    var sumsq = 0.0;
    for (var c = c0; c < c0 + cg; c = c + 1u) {
        for (var s = 0u; s < spatial; s = s + 1u) {
            let v = x[(bn * p.c + c) * spatial + s];
            sum = sum + v;
            sumsq = sumsq + v * v;
        }
    }
    let mean = sum / f32(count);
    let variance = sumsq / f32(count) - mean * mean;
    let inv = inverseSqrt(variance + p.eps);

    for (var c = c0; c < c0 + cg; c = c + 1u) {
        for (var s = 0u; s < spatial; s = s + 1u) {
            let idx = (bn * p.c + c) * spatial + s;
            let norm = (x[idx] - mean) * inv;
            let v = norm * gamma[c] + beta[c];
            y[idx] = v / (1.0 + exp(-v));
        }
    }
}
"#;

fn cpu_gn(x: &[f32], gamma: &[f32], beta: &[f32], p: &Params) -> Vec<f32> {
    let spatial = (p.d * p.h * p.w) as usize;
    let cg = (p.c / p.groups) as usize;
    let mut y = vec![0.0f32; x.len()];
    for bn in 0..p.n as usize {
        for g in 0..p.groups as usize {
            let c0 = g * cg;
            let count = cg * spatial;
            let mut sum = 0.0f32;
            let mut sumsq = 0.0f32;
            for c in c0..c0 + cg {
                for s in 0..spatial {
                    let v = x[(bn * p.c as usize + c) * spatial + s];
                    sum += v;
                    sumsq += v * v;
                }
            }
            let mean = sum / count as f32;
            let var = sumsq / count as f32 - mean * mean;
            let inv = (var + p.eps).sqrt().recip();
            for c in c0..c0 + cg {
                for s in 0..spatial {
                    let idx = (bn * p.c as usize + c) * spatial + s;
                    let norm = (x[idx] - mean) * inv;
                    let v = norm * gamma[c] + beta[c];
                    y[idx] = v / (1.0 + (-v).exp());
                }
            }
        }
    }
    y
}

fn main() {
    let p = Params { n: 1, c: 4, d: 2, h: 2, w: 2, groups: 2, eps: 1e-5, _p: 0 };
    let n = (p.n * p.c * p.d * p.h * p.w) as usize;
    let mut x = vec![0.0f32; n];
    let mut gamma = vec![0.0f32; p.c as usize];
    let mut beta = vec![0.0f32; p.c as usize];
    for i in 0..n { x[i] = ((i * 13 % 71) as f32 - 35.0) / 19.0; }
    for i in 0..gamma.len() { gamma[i] = 1.0 + ((i * 7 % 11) as f32 - 5.0) * 0.02; }
    for i in 0..beta.len() { beta[i] = ((i * 5 % 9) as f32 - 4.0) * 0.01; }
    let expect = cpu_gn(&x, &gamma, &beta, &p);

    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor { backends: wgpu::Backends::VULKAN, ..Default::default() });
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance, compatible_surface: None, force_fallback_adapter: false,
    })).expect("no Vulkan adapter");
    let info = adapter.get_info();
    println!("adapter: {} | backend: {:?}", info.name, info.backend);
    assert_eq!(info.backend, wgpu::Backend::Vulkan);

    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("groupnorm"), required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::downlevel_defaults(), ..Default::default()
    })).expect("device");

    let mk = |data: &[f32]| device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None, contents: bytemuck::cast_slice(data), usage: wgpu::BufferUsages::STORAGE,
    });
    let x_buf = mk(&x); let gamma_buf = mk(&gamma); let beta_buf = mk(&beta);
    let y_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: None, size: (n * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false,
    });
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: None, size: (n * 4) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false,
    });
    let p_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None, contents: bytemuck::bytes_of(&p), usage: wgpu::BufferUsages::UNIFORM,
    });
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("groupnorm"), source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(WGSL)),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("groupnorm"), layout: None, module: &module, entry_point: Some("main"),
        compilation_options: Default::default(), cache: None,
    });
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None, layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: x_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: gamma_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: beta_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: y_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: p_buf.as_entire_binding() },
        ],
    });
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
        pass.set_pipeline(&pipeline); pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(p.n * p.groups, 1, 1);
    }
    enc.copy_buffer_to_buffer(&y_buf, 0, &readback, 0, (n * 4) as u64);
    queue.submit(Some(enc.finish()));
    let slice = readback.slice(..);
    slice.map_async(wgpu::MapMode::Read, |r| r.expect("map"));
    device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
    let data = slice.get_mapped_range();
    let got: &[f32] = bytemuck::cast_slice(&data);
    let mut max_err = 0.0f32;
    for (a, b) in got.iter().zip(expect.iter()) { max_err = max_err.max((a - b).abs()); }
    println!("groupnorm N={} C={} G={} D={} H={} W={} max_abs_err={max_err:.3e}",
             p.n, p.c, p.groups, p.d, p.h, p.w);
    assert!(max_err < 1e-5, "groupnorm mismatch");
    println!("VULKAN GROUPNORM+SILU OK");
}

