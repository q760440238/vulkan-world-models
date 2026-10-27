use bytemuck::{Pod, Zeroable};
use std::borrow::Cow;
use wgpu::util::DeviceExt;

const C: usize = 96;
const D: usize = 2;
const H: usize = 2;
const W: usize = 2;
const S: usize = D * H * W;

const WGSL: &str = r#"
const C: u32 = 96u;
const S: u32 = 8u;
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> gamma: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let s = gid.x;
    if (s >= S) { return; }
    var ss = 0.0;
    for (var c = 0u; c < C; c = c + 1u) { let v = x[c * S + s]; ss = ss + v * v; }
    let inv = sqrt(f32(C)) / sqrt(ss + 1e-12);
    for (var c = 0u; c < C; c = c + 1u) { y[c * S + s] = x[c * S + s] * inv * gamma[c]; }
}
"#;

fn cpu(x: &[f32], g: &[f32]) -> Vec<f32> {
    let mut y = vec![0.0f32; C * S];
    for s in 0..S {
        let mut ss = 0.0f32;
        for c in 0..C { ss += x[c * S + s] * x[c * S + s]; }
        let inv = (C as f32).sqrt() / (ss + 1e-12).sqrt();
        for c in 0..C { y[c * S + s] = x[c * S + s] * inv * g[c]; }
    }
    y
}

fn main() {
    let mut x = vec![0.0f32; C * S];
    let mut g = vec![0.0f32; C];
    for i in 0..x.len() { x[i] = ((i * 13 % 71) as f32 - 35.0) / 19.0; }
    for i in 0..C { g[i] = 1.0 + ((i * 7 % 11) as f32 - 5.0) * 0.01; }
    let expect = cpu(&x, &g);
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor { backends: wgpu::Backends::VULKAN, ..Default::default() });
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance, compatible_surface: None, force_fallback_adapter: false,
    })).expect("no adapter");
    let info = adapter.get_info();
    println!("adapter: {} | backend: {:?}", info.name, info.backend);
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("vae-rms"), required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::downlevel_defaults(), ..Default::default()
    })).expect("device");
    let mk = |data: &[f32]| device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(data), usage: wgpu::BufferUsages::STORAGE });
    let xb = mk(&x); let gb = mk(&g);
    let yb = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (C * S * 4) as u64, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
    let rb = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (C * S * 4) as u64, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: None, source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(WGSL)) });
    let pipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: None, layout: None, module: &module, entry_point: Some("main"), compilation_options: Default::default(), cache: None });
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &pipe.get_bind_group_layout(0), entries: &[
        wgpu::BindGroupEntry { binding: 0, resource: xb.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 1, resource: gb.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 2, resource: yb.as_entire_binding() },
    ]});
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
        p.set_pipeline(&pipe); p.set_bind_group(0, &bg, &[]); p.dispatch_workgroups(S as u32, 1, 1);
    }
    enc.copy_buffer_to_buffer(&yb, 0, &rb, 0, (C * S * 4) as u64);
    queue.submit(Some(enc.finish()));
    let slice = rb.slice(..); slice.map_async(wgpu::MapMode::Read, |r| r.expect("map"));
    device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
    let data = slice.get_mapped_range();
    let got: &[f32] = bytemuck::cast_slice(&data);
    let mut max_err = 0.0f32;
    for (a, b) in got.iter().zip(expect.iter()) { max_err = max_err.max((a - b).abs()); }
    println!("vae-rms C={C} spatial={S} max_abs_err={max_err:.3e}");
    assert!(max_err < 1e-5, "rms mismatch");
    println!("VULKAN VAE RMS NORM OK");
}
