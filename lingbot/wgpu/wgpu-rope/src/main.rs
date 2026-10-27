use bytemuck::{Pod, Zeroable};
use std::borrow::Cow;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Params { tokens: u32, heads: u32, dim: u32, _pad: u32 }

const WGSL: &str = r#"
struct Params { tokens: u32, heads: u32, dim: u32, pad: u32 };
@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<storage, read> cos_tab: array<f32>;
@group(0) @binding(2) var<storage, read> sin_tab: array<f32>;
@group(0) @binding(3) var<uniform> p: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pair = gid.x;
    let pairs = p.dim / 2u;
    let total = p.tokens * p.heads * pairs;
    if (pair >= total) { return; }
    let pp = pair % pairs;
    let h = (pair / pairs) % p.heads;
    let t = pair / (pairs * p.heads);
    let base = (t * p.heads + h) * p.dim + pp * 2u;
    let a = x[base];
    let b = x[base + 1u];
    let c = cos_tab[t * pairs + pp];
    let s = sin_tab[t * pairs + pp];
    x[base] = a * c - b * s;
    x[base + 1u] = a * s + b * c;
}
"#;

fn cpu_rope(x: &mut [f32], cos_tab: &[f32], sin_tab: &[f32], tokens: usize, heads: usize, dim: usize) {
    let pairs = dim / 2;
    for t in 0..tokens {
        for h in 0..heads {
            for p in 0..pairs {
                let base = (t * heads + h) * dim + p * 2;
                let a = x[base];
                let b = x[base + 1];
                let c = cos_tab[t * pairs + p];
                let s = sin_tab[t * pairs + p];
                x[base] = a * c - b * s;
                x[base + 1] = a * s + b * c;
            }
        }
    }
}

fn main() {
    let (tokens, heads, dim) = (8usize, 4usize, 64usize);
    let n = tokens * heads * dim;
    let pairs = dim / 2;
    let mut x = vec![0.0f32; n];
    let mut cos_tab = vec![0.0f32; tokens * pairs];
    let mut sin_tab = vec![0.0f32; tokens * pairs];
    for i in 0..n { x[i] = ((i * 19 % 101) as f32 - 50.0) / 17.0; }
    for t in 0..tokens {
        for p in 0..pairs {
            let theta = (t as f32) * 0.37 + (p as f32) * 0.11;
            cos_tab[t * pairs + p] = theta.cos();
            sin_tab[t * pairs + p] = theta.sin();
        }
    }
    let mut expect = x.clone();
    cpu_rope(&mut expect, &cos_tab, &sin_tab, tokens, heads, dim);

    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN,
        ..Default::default()
    });
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    })).expect("no Vulkan adapter");
    let info = adapter.get_info();
    println!("adapter: {} | backend: {:?} | type: {:?}", info.name, info.backend, info.device_type);
    assert_eq!(info.backend, wgpu::Backend::Vulkan);

    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("rope"), required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::downlevel_defaults(), ..Default::default()
    })).expect("device");

    let x_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("x"), contents: bytemuck::cast_slice(&x),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    let c_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("cos"), contents: bytemuck::cast_slice(&cos_tab), usage: wgpu::BufferUsages::STORAGE,
    });
    let s_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("sin"), contents: bytemuck::cast_slice(&sin_tab), usage: wgpu::BufferUsages::STORAGE,
    });
    let params = Params { tokens: tokens as u32, heads: heads as u32, dim: dim as u32, _pad: 0 };
    let p_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("params"), contents: bytemuck::bytes_of(&params), usage: wgpu::BufferUsages::UNIFORM,
    });
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"), size: (n * 4) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false,
    });

    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("rope"), source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(WGSL)),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("rope"), layout: None, module: &module, entry_point: Some("main"),
        compilation_options: Default::default(), cache: None,
    });
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bg"), layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: x_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: c_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: s_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: p_buf.as_entire_binding() },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("enc") });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("rope"), timestamp_writes: None });
        pass.set_pipeline(&pipeline); pass.set_bind_group(0, &bg, &[]);
        let total = (tokens * heads * pairs) as u32;
        pass.dispatch_workgroups((total + 255) / 256, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&x_buf, 0, &readback, 0, (n * 4) as u64);
    queue.submit(Some(encoder.finish()));
    let slice = readback.slice(..);
    slice.map_async(wgpu::MapMode::Read, |r| r.expect("map"));
    device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
    let data = slice.get_mapped_range();
    let got: &[f32] = bytemuck::cast_slice(&data);
    let mut max_err = 0.0f32;
    for (a, b) in got.iter().zip(expect.iter()) { max_err = max_err.max((a - b).abs()); }
    println!("tokens={tokens} heads={heads} dim={dim} max_abs_err={max_err:.3e}");
    assert!(max_err < 1e-5, "RoPE mismatch");
    println!("VULKAN ROPE OK");
}
