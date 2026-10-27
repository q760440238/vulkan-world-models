use bytemuck::{Pod, Zeroable};
use std::borrow::Cow;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Params { heads: u32, l: u32, d: u32, scale: f32 }

const WGSL: &str = r#"
struct Params { heads: u32, l: u32, d: u32, scale: f32 };
@group(0) @binding(0) var<storage, read> q: array<f32>; // [L,D]
@group(0) @binding(1) var<storage, read> k: array<f32>;
@group(0) @binding(2) var<storage, read> v: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<uniform> p: Params;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let h = gid.x / p.l;
    let i = gid.x % p.l;
    if (h >= p.heads) { return; }
    if (i >= p.l) { return; }
    var scores: array<f32, 64>;
    var maxs = -1e30;
    for (var j = 0u; j < p.l; j = j + 1u) {
        if (j > i) { break; }
        var dot = 0.0;
        for (var d = 0u; d < p.d; d = d + 1u) {
            dot = dot + q[(h * p.l + i) * p.d + d] * k[(h * p.l + j) * p.d + d];
        }
        scores[j] = dot * p.scale;
        maxs = max(maxs, scores[j]);
    }
    var sum = 0.0;
    for (var j = 0u; j <= i; j = j + 1u) { sum = sum + exp(scores[j] - maxs); }
    for (var d = 0u; d < p.d; d = d + 1u) {
        var acc = 0.0;
        for (var j = 0u; j <= i; j = j + 1u) {
            acc = acc + (exp(scores[j] - maxs) / sum) * v[(h * p.l + j) * p.d + d];
        }
        out[(h * p.l + i) * p.d + d] = acc;
    }
}
"#;

fn cpu_attention(q: &[f32], k: &[f32], v: &[f32], heads: usize, l: usize, d: usize, scale: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; heads * l * d];
    for hh in 0..heads {
        for i in 0..l {
            let mut scores = vec![0.0f32; i + 1];
            let mut maxs = f32::NEG_INFINITY;
            for j in 0..=i {
                let mut dot = 0.0f32;
                for dd in 0..d { dot += q[(hh * l + i) * d + dd] * k[(hh * l + j) * d + dd]; }
                scores[j] = dot * scale;
                maxs = maxs.max(scores[j]);
            }
            let sum: f32 = scores.iter().map(|s| (s - maxs).exp()).sum();
            for dd in 0..d {
                let mut acc = 0.0f32;
                for j in 0..=i { acc += ((scores[j] - maxs).exp() / sum) * v[(hh * l + j) * d + dd]; }
                out[(hh * l + i) * d + dd] = acc;
            }
        }
    }
    out
}

fn main() {
    let (heads, l, d) = (4usize, 32usize, 128usize);
    let scale = 1.0 / (d as f32).sqrt();
    let mut q = vec![0.0f32; heads * l * d];
    let mut k = vec![0.0f32; heads * l * d];
    let mut v = vec![0.0f32; heads * l * d];
    for i in 0..heads * l * d {
        q[i] = ((i * 13 % 71) as f32 - 35.0) / 19.0;
        k[i] = ((i * 23 % 67) as f32 - 33.0) / 17.0;
        v[i] = ((i * 31 % 61) as f32 - 30.0) / 13.0;
    }
    let expect = cpu_attention(&q, &k, &v, heads, l, d, scale);

    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN, ..Default::default()
    });
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None, force_fallback_adapter: false,
    })).expect("no Vulkan adapter");
    let info = adapter.get_info();
    println!("adapter: {} | backend: {:?} | type: {:?}", info.name, info.backend, info.device_type);
    assert_eq!(info.backend, wgpu::Backend::Vulkan);

    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("attn"), required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::downlevel_defaults(), ..Default::default()
    })).expect("device");

    let q_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("q"), contents: bytemuck::cast_slice(&q), usage: wgpu::BufferUsages::STORAGE,
    });
    let k_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("k"), contents: bytemuck::cast_slice(&k), usage: wgpu::BufferUsages::STORAGE,
    });
    let v_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("v"), contents: bytemuck::cast_slice(&v), usage: wgpu::BufferUsages::STORAGE,
    });
    let o_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("o"), size: (heads * l * d * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false,
    });
    let params = Params { heads: heads as u32, l: l as u32, d: d as u32, scale };
    let p_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("params"), contents: bytemuck::bytes_of(&params), usage: wgpu::BufferUsages::UNIFORM,
    });
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"), size: (heads * l * d * 4) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false,
    });

    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("attn"), source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(WGSL)),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("attn"), layout: None, module: &module, entry_point: Some("main"),
        compilation_options: Default::default(), cache: None,
    });
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bg"), layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: q_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: k_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: v_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: o_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: p_buf.as_entire_binding() },
        ],
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("enc") });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("attn"), timestamp_writes: None });
        pass.set_pipeline(&pipeline); pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups((heads * l) as u32, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&o_buf, 0, &readback, 0, (heads * l * d * 4) as u64);
    queue.submit(Some(encoder.finish()));
    let slice = readback.slice(..);
    slice.map_async(wgpu::MapMode::Read, |r| r.expect("map"));
    device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
    let data = slice.get_mapped_range();
    let got: &[f32] = bytemuck::cast_slice(&data);
    let mut max_err = 0.0f32;
    for (a, b) in got.iter().zip(expect.iter()) { max_err = max_err.max((a - b).abs()); }
    println!("heads={heads} L={l} D={d} max_abs_err={max_err:.3e}");
    assert!(max_err < 1e-5, "attention mismatch");
    println!("VULKAN ATTENTION OK");
}


