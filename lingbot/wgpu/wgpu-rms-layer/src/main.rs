use bytemuck::{Pod, Zeroable};
use half::f16;
use std::borrow::Cow;
use wgpu::util::DeviceExt;

const M: usize = 1560;
const C: usize = 1536;

const WGSL: &str = r#"
enable f16;
const C: u32 = 1536u;
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> g: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f16>;
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = gid.x;
    if (t >= 1560u) { return; }
    var ss = 0.0;
    for (var c = 0u; c < C; c = c + 1u) { let v = x[t * C + c]; ss = ss + v * v; }
    let inv = sqrt(f32(C)) / sqrt(ss + 1e-12);
    for (var c = 0u; c < C; c = c + 1u) { y[t * C + c] = f16(x[t * C + c] * inv * g[c] + b[c]); }
}
"#;

fn main() {
    let raw = std::fs::read("D:/3DWord/itseez3d/ai-lab/lingbot-world/port/wgpu-qkv-rope-out.bin").expect("out.bin");
    let x: Vec<f32> = bytemuck::cast_slice::<u8, f32>(&raw)[..M * C].to_vec();
    let eraw = std::fs::read("D:/3DWord/itseez3d/ai-lab/lingbot-world/port/wgpu-time-e.bin").expect("time e");
    let e: &[f32] = bytemuck::cast_slice(&eraw);
    let mut g = vec![0.0f32; C];
    let mut b = vec![0.0f32; C];
    for i in 0..C {
        let e3 = e[3 * C + i]; let e4 = e[4 * C + i];
        let gamma = 1.0 + ((i * 7 % 11) as f32 - 5.0) * 0.01;
        g[i] = gamma * (1.0 + e4);
        b[i] = e3;
    }
    let mut expect = vec![0.0f32; 2 * C];
    for t in 0..2 {
        let mut ss = 0.0f32;
        for c in 0..C { ss += x[t * C + c] * x[t * C + c]; }
        let inv = (C as f32).sqrt() / (ss + 1e-12).sqrt();
        for c in 0..C { expect[t * C + c] = x[t * C + c] * inv * g[c] + b[c]; }
    }
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor { backends: wgpu::Backends::VULKAN, ..Default::default() });
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions { power_preference: wgpu::PowerPreference::HighPerformance, compatible_surface: None, force_fallback_adapter: false })).expect("no adapter");
    let info = adapter.get_info();
    println!("adapter: {} | backend: {:?}", info.name, info.backend);
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("rms-layer"), required_features: wgpu::Features::SHADER_F16, required_limits: wgpu::Limits::downlevel_defaults(), ..Default::default()
    })).expect("device");
    let mk = |d: &[f32]| device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(d), usage: wgpu::BufferUsages::STORAGE });
    let xb=mk(&x); let gb=mk(&g); let bb=mk(&b);
    let yb = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (M*C*2) as u64, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
    let rb = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (M*C*2) as u64, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: None, source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(WGSL)) });
    let pipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: None, layout: None, module: &module, entry_point: Some("main"), compilation_options: Default::default(), cache: None });
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &pipe.get_bind_group_layout(0), entries: &[
        wgpu::BindGroupEntry { binding: 0, resource: xb.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 1, resource: gb.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 2, resource: bb.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 3, resource: yb.as_entire_binding() },
    ]});
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
        p.set_pipeline(&pipe); p.set_bind_group(0, &bg, &[]); p.dispatch_workgroups(M as u32, 1, 1);
    }
    enc.copy_buffer_to_buffer(&yb, 0, &rb, 0, (M*C*2) as u64);
    let t0 = std::time::Instant::now();
    queue.submit(Some(enc.finish()));
    let slice = rb.slice(..); slice.map_async(wgpu::MapMode::Read, |r| r.expect("map"));
    device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
    let data = slice.get_mapped_range();
    let got: &[f16] = bytemuck::cast_slice(&data);
    let mut max_err = 0.0f32;
    for t in 0..2 { for c in 0..C { max_err = max_err.max((got[t*C+c].to_f32() - expect[t*C+c]).abs()); } }
    std::fs::write("D:/3DWord/itseez3d/ai-lab/lingbot-world/port/wgpu-rms-layer-h2.bin", bytemuck::cast_slice(got)).expect("write h2");
    println!("rms-layer M={M} C={C} gpu_ms={:.3} first2_max_abs_err={max_err:.3e}", t0.elapsed().as_secs_f64()*1000.0);
    assert!(max_err < 1e-2, "rms layer mismatch");
    println!("VULKAN RMS LAYER OK");
}




