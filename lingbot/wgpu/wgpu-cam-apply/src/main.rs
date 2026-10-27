use bytemuck::{Pod, Zeroable};
use half::f16;
use std::borrow::Cow;
use wgpu::util::DeviceExt;

const M: usize = 1560;
const C: usize = 1536;

const WGSL: &str = r#"
enable f16;
const M: u32 = 1560u; const C: u32 = 1536u;
@group(0) @binding(0) var<storage, read> x: array<f16>;
@group(0) @binding(1) var<storage, read> scale: array<f32>;
@group(0) @binding(2) var<storage, read> shift: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f16>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= M * C) { return; }
    let t = i / C; let c = i % C;
    y[i] = f16(f32(x[i]) * (1.0 + scale[i]) + shift[i]);
}
"#;

fn main() {
    let raw = std::fs::read("D:/3DWord/itseez3d/ai-lab/lingbot-world/port/wgpu-qkv-rope-x.bin").expect("x.bin");
    let x16: Vec<f16> = bytemuck::cast_slice::<u8, f16>(&raw)[..M*C].to_vec();
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor { backends: wgpu::Backends::VULKAN, ..Default::default() });
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions { power_preference: wgpu::PowerPreference::HighPerformance, compatible_surface: None, force_fallback_adapter: false })).expect("no adapter");
    let info = adapter.get_info();
    println!("adapter: {} | backend: {:?}", info.name, info.backend);
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("cam-apply"), required_features: wgpu::Features::SHADER_F16, required_limits: wgpu::Limits::downlevel_defaults(), ..Default::default()
    })).expect("device");
    let sraw = std::fs::read("D:/3DWord/itseez3d/ai-lab/lingbot-world/port/wgpu-cam-scale.bin").expect("scale");
    let braw = std::fs::read("D:/3DWord/itseez3d/ai-lab/lingbot-world/port/wgpu-cam-shift.bin").expect("shift");
    let scale: Vec<f32> = bytemuck::cast_slice::<u8, f32>(&sraw)[..M*C].to_vec();
    let shift: Vec<f32> = bytemuck::cast_slice::<u8, f32>(&braw)[..M*C].to_vec();
    let xb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(&x16), usage: wgpu::BufferUsages::STORAGE });
    let sb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(&scale), usage: wgpu::BufferUsages::STORAGE });
    let bb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(&shift), usage: wgpu::BufferUsages::STORAGE });
    let yb = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (M*C*2) as u64, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
    let rb = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (M*C*2) as u64, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: None, source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(WGSL)) });
    let pipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: None, layout: None, module: &module, entry_point: Some("main"), compilation_options: Default::default(), cache: None });
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &pipe.get_bind_group_layout(0), entries: &[
        wgpu::BindGroupEntry { binding: 0, resource: xb.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 1, resource: sb.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 2, resource: bb.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 3, resource: yb.as_entire_binding() },
    ]});
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
        p.set_pipeline(&pipe); p.set_bind_group(0, &bg, &[]); p.dispatch_workgroups(((M*C + 63)/64) as u32, 1, 1);
    }
    enc.copy_buffer_to_buffer(&yb,0,&rb,0,(M*C*2) as u64);
    queue.submit(Some(enc.finish()));
    let slice = rb.slice(..); slice.map_async(wgpu::MapMode::Read, |r| r.expect("map"));
    device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
    let data = slice.get_mapped_range();
    let got: &[f16] = bytemuck::cast_slice(&data);
    let mut max_err = 0.0f32;
    for t in 0..2 { for c in 0..C {
        let exp = x16[t*C+c].to_f32() * (1.0 + scale[t*C+c]) + shift[t*C+c];
        max_err = max_err.max((got[t*C+c].to_f32() - exp).abs());
    }}
    std::fs::write("D:/3DWord/itseez3d/ai-lab/lingbot-world/port/wgpu-cam-x.bin", bytemuck::cast_slice(got)).expect("write");
    println!("cam-apply M={M} C={C} first2_max_abs_err={max_err:.3e}");
    assert!(max_err < 1e-2, "cam mismatch");
    println!("VULKAN CAM APPLY OK");
}

