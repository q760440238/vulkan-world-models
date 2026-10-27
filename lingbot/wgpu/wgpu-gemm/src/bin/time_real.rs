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
@group(0) @binding(0) var<storage, read> a: array<f16>;   // [M, K]
@group(0) @binding(1) var<storage, read> b: array<f16>;   // [N, K]
@group(0) @binding(2) var<storage, read_write> c: array<f16>; // [M, N]
@group(0) @binding(3) var<uniform> p: Params;

var<workgroup> As: array<f16, 256>;
var<workgroup> Bs: array<f16, 256>;

@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let tx = lid.x;
    let ty = lid.y;
    let row0 = wid.x * 16u;
    let col0 = wid.y * 16u;
    let row = row0 + ty;
    let col = col0 + tx;
    var acc = 0.0;

    var k0 = 0u;
    loop {
        if (k0 >= p.k) { break; }
        if (row < p.m && (k0 + tx) < p.k) { As[ty * 16u + tx] = a[row * p.k + k0 + tx]; }
        else { As[ty * 16u + tx] = 0.0; }
        // B tile is indexed by the *local y* row, not by the output column.
        if ((col0 + ty) < p.n && (k0 + tx) < p.k) { Bs[ty * 16u + tx] = b[(col0 + ty) * p.k + k0 + tx]; }
        else { Bs[ty * 16u + tx] = 0.0; }
        workgroupBarrier();

        var kk = 0u;
        loop {
            if (kk >= 16u) { break; }
            acc = acc + f32(As[ty * 16u + kk]) * f32(Bs[tx * 16u + kk]);
            kk = kk + 1u;
        }
        workgroupBarrier();
        k0 = k0 + 16u;
    }

    if (row < p.m && col < p.n) { c[row * p.n + col] = f16(acc); }
}
"#;fn sig(v: f32) -> f32 { v / (1.0 + (-v).exp()) }
fn q(v: f32) -> f32 { f16::from_f32(v).to_f32() }

fn main() {
    let d = 1536usize;
    let h = 1536usize;
    let out = 9216usize;
    let mut x = vec![0.0f32; 256];
    for i in 0..256 { x[i] = (i as f32 * 0.1).sin(); }
    let mut w1 = vec![0.0f32; d * 256];
    let mut w2 = vec![0.0f32; d * d];
    let mut w3 = vec![0.0f32; out * d];
    for i in 0..w1.len() { w1[i] = ((i * 17 % 97) as f32 - 48.0) / (256.0 * 8.0); }
    for i in 0..w2.len() { w2[i] = ((i * 23 % 89) as f32 - 44.0) / (d as f32 * 8.0); }
    for i in 0..w3.len() { w3[i] = ((i * 29 % 83) as f32 - 41.0) / (d as f32 * 8.0); }
    let xq: Vec<f32> = x.iter().map(|v| q(*v)).collect();
    let w1q: Vec<f32> = w1.iter().map(|v| q(*v)).collect();
    let w2q: Vec<f32> = w2.iter().map(|v| q(*v)).collect();
    let w3q: Vec<f32> = w3.iter().map(|v| q(*v)).collect();
    let mut e1 = vec![0.0f32; d];
    for j in 0..d { let mut acc = 0.0f32; for kk in 0..256 { acc += xq[kk] * w1q[j * 256 + kk]; } e1[j] = sig(q(acc)); }
    let mut e2 = vec![0.0f32; d];
    for j in 0..d { let mut acc = 0.0f32; for kk in 0..d { acc += e1[kk] * w2q[j * d + kk]; } e2[j] = sig(q(acc)); }
    let mut e3 = vec![0.0f32; out];
    for j in 0..out { let mut acc = 0.0f32; for kk in 0..d { acc += e2[kk] * w3q[j * d + kk]; } e3[j] = q(acc); }

    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor { backends: wgpu::Backends::VULKAN, ..Default::default() });
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions { power_preference: wgpu::PowerPreference::HighPerformance, compatible_surface: None, force_fallback_adapter: false })).expect("no adapter");
    let info = adapter.get_info();
    println!("adapter: {} | backend: {:?}", info.name, info.backend);
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("time-real"), required_features: wgpu::Features::SHADER_F16, required_limits: wgpu::Limits::downlevel_defaults(), ..Default::default()
    })).expect("device");
    let mk = |data: &[f16]| device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(data), usage: wgpu::BufferUsages::STORAGE });
    let x16: Vec<f16> = x.iter().map(|v| f16::from_f32(*v)).collect();
    let w1_16: Vec<f16> = w1.iter().map(|v| f16::from_f32(*v)).collect();
    let w2_16: Vec<f16> = w2.iter().map(|v| f16::from_f32(*v)).collect();
    let w3_16: Vec<f16> = w3.iter().map(|v| f16::from_f32(*v)).collect();
    let xb=mk(&x16); let w1b=mk(&w1_16); let w2b=mk(&w2_16); let w3b=mk(&w3_16);
    let e1b = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (d*2) as u64, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
    let e2b = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (d*2) as u64, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
    let e3b = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (out*2) as u64, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
    let rb1 = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (d*2) as u64, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
    let rb2 = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (d*2) as u64, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
    let rb3 = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (out*2) as u64, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
    let p1 = Params { m:1,n:d as u32,k:256,pad:0 };
    let p2 = Params { m:1,n:d as u32,k:d as u32,pad:0 };
    let p3 = Params { m:1,n:out as u32,k:d as u32,pad:0 };
    let pb1 = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::bytes_of(&p1), usage: wgpu::BufferUsages::UNIFORM });
    let pb2 = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::bytes_of(&p2), usage: wgpu::BufferUsages::UNIFORM });
    let pb3 = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::bytes_of(&p3), usage: wgpu::BufferUsages::UNIFORM });
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: None, source: wgpu::ShaderSource::Wgsl(Cow::Owned(WGSL.to_string())) });
    let pipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: None, layout: None, module: &module, entry_point: Some("main"), compilation_options: Default::default(), cache: None });
    let bind = |a: &wgpu::Buffer, b: &wgpu::Buffer, c: &wgpu::Buffer, p: &wgpu::Buffer| device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &pipe.get_bind_group_layout(0), entries: &[
        wgpu::BindGroupEntry { binding: 0, resource: a.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 1, resource: b.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 2, resource: c.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 3, resource: p.as_entire_binding() },
    ]});
    let bg1 = bind(&xb,&w1b,&e1b,&pb1);
    let bg2 = bind(&e1b,&w2b,&e2b,&pb2);
    let bg3 = bind(&e2b,&w3b,&e3b,&pb3);
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
        p.set_pipeline(&pipe); p.set_bind_group(0,&bg1,&[]); p.dispatch_workgroups(1, (d as u32 + 15)/16, 1);
    }
    enc.copy_buffer_to_buffer(&e1b,0,&rb1,0,(d*2) as u64);
    queue.submit(Some(enc.finish()));
    let s1=rb1.slice(..); s1.map_async(wgpu::MapMode::Read, |r| r.expect("map")); device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
    let d1=s1.get_mapped_range(); let a1:&[f16]=bytemuck::cast_slice(&d1);
    let mut s1v=vec![f16::from_f32(0.0);d];
    for i in 0..d { s1v[i]=f16::from_f32(sig(a1[i].to_f32())); }
    drop(d1); queue.write_buffer(&e1b,0,bytemuck::cast_slice(&s1v));
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
        p.set_pipeline(&pipe); p.set_bind_group(0,&bg2,&[]); p.dispatch_workgroups(1, (d as u32 + 15)/16, 1);
    }
    enc.copy_buffer_to_buffer(&e2b,0,&rb2,0,(d*2) as u64);
    queue.submit(Some(enc.finish()));
    let s2=rb2.slice(..); s2.map_async(wgpu::MapMode::Read, |r| r.expect("map")); device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
    let d2=s2.get_mapped_range(); let a2:&[f16]=bytemuck::cast_slice(&d2);
    let mut s2v=vec![f16::from_f32(0.0);d];
    for i in 0..d { s2v[i]=f16::from_f32(sig(a2[i].to_f32())); }
    drop(d2); queue.write_buffer(&e2b,0,bytemuck::cast_slice(&s2v));
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
        p.set_pipeline(&pipe); p.set_bind_group(0,&bg3,&[]); p.dispatch_workgroups(1, (out as u32 + 15)/16, 1);
    }
    enc.copy_buffer_to_buffer(&e3b,0,&rb3,0,(out*2) as u64);
    let t0=std::time::Instant::now(); queue.submit(Some(enc.finish()));
    let s3=rb3.slice(..); s3.map_async(wgpu::MapMode::Read, |r| r.expect("map")); device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
    let d3=s3.get_mapped_range(); let a3:&[f16]=bytemuck::cast_slice(&d3);
    let mut max_err=0.0f32; for i in 0..out { max_err=max_err.max((a3[i].to_f32()-e3[i]).abs()); }
    let e32: Vec<f32> = a3.iter().map(|v| v.to_f32()).collect();
    std::fs::write("D:/3DWord/itseez3d/ai-lab/lingbot-world/port/wgpu-time-e.bin", bytemuck::cast_slice(&e32)).expect("write e");
    println!("time-real D={d} H={h} OUT={out} last_ms={:.3} max_abs_err={max_err:.3e}", t0.elapsed().as_secs_f64()*1000.0);
    assert!(max_err<5e-2,"time mismatch");
    println!("VULKAN TIME MLP OK");
}
