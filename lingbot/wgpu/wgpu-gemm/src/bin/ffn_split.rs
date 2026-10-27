use bytemuck::{Pod, Zeroable};
use half::f16;
use std::borrow::Cow;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Params { m: u32, n: u32, k: u32, pad: u32 }

const WGSL: &str = r#"
enable f16;
struct Params { m: u32, n: u32, k: u32, pad: u32 };
@group(0) @binding(0) var<storage, read> a: array<f16>;
@group(0) @binding(1) var<storage, read> b: array<f16>;
@group(0) @binding(2) var<storage, read_write> c: array<f16>;
@group(0) @binding(3) var<uniform> p: Params;
var<workgroup> As: array<f16, 256>;
var<workgroup> Bs: array<f16, 256>;
@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let tx = lid.x; let ty = lid.y;
    let row0 = wid.x * 16u; let col0 = wid.y * 16u;
    let row = row0 + ty; let col = col0 + tx;
    var acc = 0.0;
    var k0 = 0u;
    loop {
        if (k0 >= p.k) { break; }
        if (row < p.m && (k0 + tx) < p.k) { As[ty * 16u + tx] = a[row * p.k + k0 + tx]; } else { As[ty * 16u + tx] = f16(0.0); }
        if ((col0 + ty) < p.n && (k0 + tx) < p.k) { Bs[ty * 16u + tx] = b[(col0 + ty) * p.k + k0 + tx]; } else { Bs[ty * 16u + tx] = f16(0.0); }
        workgroupBarrier();
        var kk = 0u;
        loop { if (kk >= 16u) { break; } acc = acc + f32(As[ty * 16u + kk]) * f32(Bs[tx * 16u + kk]); kk = kk + 1u; }
        workgroupBarrier();
        k0 = k0 + 16u;
    }
    if (row < p.m && col < p.n) { c[row * p.n + col] = f16(acc); }
}
"#;

fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + (0.79788456 * (x + 0.044715 * x * x * x)).tanh())
}
fn q(v: f32) -> f32 { f16::from_f32(v).to_f32() }

fn cpu_ffn(x: &[f32], w1: &[f32], w2: &[f32], m: usize, d: usize, f: usize) -> Vec<f32> {
    let mut up = vec![0.0f32; m * f];
    for i in 0..m { for j in 0..f {
        let mut acc = 0.0f32;
        for k in 0..d { acc += q(x[i * d + k]) * q(w1[j * d + k]); }
        up[i * f + j] = q(gelu(q(acc)));
    }}
    let mut y = vec![0.0f32; m * d];
    for i in 0..m { for j in 0..d {
        let mut acc = 0.0f32;
        for k in 0..f { acc += q(up[i * f + k]) * q(w2[j * f + k]); }
        y[i * d + j] = q(acc);
    }}
    y
}

fn main() {
    let (m, d, f) = (1560usize, 1536usize, 8960usize);
    let raw = std::fs::read("D:/3DWord/itseez3d/ai-lab/lingbot-world/port/wgpu-rms-layer-h2.bin").expect("h2.bin");
    let x16: Vec<f16> = bytemuck::cast_slice::<u8, f16>(&raw)[..m * d].to_vec();
    let x: Vec<f32> = x16.iter().map(|v| v.to_f32()).collect();
    let mut w1 = vec![0.0f32; f * d];
    let mut w2 = vec![0.0f32; d * f];

    for i in 0..w1.len() { w1[i] = ((i * 17 % 97) as f32 - 48.0) / (d as f32 * 8.0); }
    for i in 0..w2.len() { w2[i] = ((i * 23 % 89) as f32 - 44.0) / (f as f32 * 8.0); }
    println!("cpu reference...");
    let raw2 = std::fs::read("D:/3DWord/itseez3d/ai-lab/lingbot-world/port/wgpu-qkv-rope-out.bin").expect("out.bin");
    let x2: Vec<f32> = bytemuck::cast_slice::<u8, f32>(&raw2)[..m * d].to_vec();
    let eraw = std::fs::read("D:/3DWord/itseez3d/ai-lab/lingbot-world/port/wgpu-time-e.bin").expect("time e");
    let efull: &[f32] = bytemuck::cast_slice(&eraw);
    let e5: Vec<f32> = efull[5 * d..6 * d].to_vec();
    let mut expect = cpu_ffn(&x[..2 * d], &w1, &w2, 2, d, f);
    for i in 0..2 { for j in 0..d { expect[i * d + j] = expect[i * d + j] * e5[j] + x2[i * d + j]; } }

    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor { backends: wgpu::Backends::VULKAN, ..Default::default() });
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance, compatible_surface: None, force_fallback_adapter: false,
    })).expect("no Vulkan adapter");
    let info = adapter.get_info();
    println!("adapter: {} | backend: {:?}", info.name, info.backend);
    let mut limits = wgpu::Limits::downlevel_defaults();
    limits.max_storage_buffer_binding_size = 256 << 20;
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("ffn"), required_features: wgpu::Features::SHADER_F16,
        required_limits: limits, ..Default::default()
    })).expect("device");
    let mk = |data: &[f16]| device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None, contents: bytemuck::cast_slice(data), usage: wgpu::BufferUsages::STORAGE,
    });
    let x16: Vec<f16> = x.iter().map(|v| f16::from_f32(*v)).collect();
    let w1_16: Vec<f16> = w1.iter().map(|v| f16::from_f32(*v)).collect();
    let w2_16: Vec<f16> = w2.iter().map(|v| f16::from_f32(*v)).collect();
    let x_buf = mk(&x16); let w1_buf = mk(&w1_16); let w2_buf = mk(&w2_16);
    let ffn_buf = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (m * f * 2) as u64, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
    let act_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None, contents: bytemuck::cast_slice(&vec![f16::from_f32(0.0); m * f]),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });
    let y_buf = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (m * d * 2) as u64, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
    let up_p = Params { m: m as u32, n: f as u32, k: d as u32, pad: 0 };
    let dn_p = Params { m: m as u32, n: d as u32, k: f as u32, pad: 0 };
    let up_pb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::bytes_of(&up_p), usage: wgpu::BufferUsages::UNIFORM });
    let dn_pb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::bytes_of(&dn_p), usage: wgpu::BufferUsages::UNIFORM });
    let up_rb = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (m * f * 2) as u64, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
    let y_rb = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (m * d * 2) as u64, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: None, source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(WGSL)) });
    let pipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: None, layout: None, module: &module, entry_point: Some("main"), compilation_options: Default::default(), cache: None });
    let bind = |x: &wgpu::Buffer, w: &wgpu::Buffer, y: &wgpu::Buffer, pb: &wgpu::Buffer| device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &pipe.get_bind_group_layout(0), entries: &[
        wgpu::BindGroupEntry { binding: 0, resource: x.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 1, resource: w.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 2, resource: y.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 3, resource: pb.as_entire_binding() },
    ]});
    let up_bg = bind(&x_buf, &w1_buf, &ffn_buf, &up_pb);
    let dn_bg = bind(&act_buf, &w2_buf, &y_buf, &dn_pb);
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
        p.set_pipeline(&pipe); p.set_bind_group(0, &up_bg, &[]); p.dispatch_workgroups(((m + 15) / 16) as u32, ((f + 15) / 16) as u32, 1);
    }
    enc.copy_buffer_to_buffer(&ffn_buf, 0, &up_rb, 0, (m * f * 2) as u64);
    queue.submit(Some(enc.finish()));
    let su = up_rb.slice(..); su.map_async(wgpu::MapMode::Read, |r| r.expect("map"));
    device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
    let du = su.get_mapped_range();
    let up: &[f16] = bytemuck::cast_slice(&du);
    let mut act = vec![f16::from_f32(0.0); m * f];
    for i in 0..m * f { act[i] = f16::from_f32(gelu(up[i].to_f32())); }
    drop(du);
    queue.write_buffer(&act_buf, 0, bytemuck::cast_slice(&act));
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
        p.set_pipeline(&pipe); p.set_bind_group(0, &dn_bg, &[]); p.dispatch_workgroups(((m + 15) / 16) as u32, ((d + 15) / 16) as u32, 1);
    }
    enc.copy_buffer_to_buffer(&y_buf, 0, &y_rb, 0, (m * d * 2) as u64);
    let t0 = std::time::Instant::now();
    queue.submit(Some(enc.finish()));
    let sy = y_rb.slice(..); sy.map_async(wgpu::MapMode::Read, |r| r.expect("map"));
    device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
    let dy = sy.get_mapped_range();
    let y: &[f16] = bytemuck::cast_slice(&dy);
    let mut max_err = 0.0f32;
    for i in 0..2 * d {
        let got = y[i].to_f32() * e5[i % d] + x2[i];
        max_err = max_err.max((got - expect[i]).abs());
    }
    let mut y16 = vec![f16::from_f32(0.0); m * d];
    for i in 0..m * d { y16[i] = f16::from_f32(y[i].to_f32() * e5[i % d] + x2[i]); }
    std::fs::write("D:/3DWord/itseez3d/ai-lab/lingbot-world/port/wgpu-layer-y.bin", bytemuck::cast_slice(&y16)).expect("write y");
    println!("ffn split M={m} D={d} F={f} down_ms={:.3} max_abs_err={max_err:.3e}", t0.elapsed().as_secs_f64() * 1000.0);
    assert!(max_err < 5e-2, "ffn mismatch");
    println!("VULKAN SPLIT FFN OK");
}









