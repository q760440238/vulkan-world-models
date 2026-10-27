use bytemuck::{Pod, Zeroable};
use std::borrow::Cow;
use wgpu::util::DeviceExt;

const D: usize = 32;
const H: usize = 64;
const OUT: usize = 6 * D;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Params { t: f32, p0: u32, p1: u32, p2: u32 }

const WGSL: &str = r#"
const D: u32 = 32u;
const H: u32 = 64u;
const OUT: u32 = 192u;
struct Params { t: f32, p0: u32, p1: u32, p2: u32 };
@group(0) @binding(0) var<storage, read> w1: array<f32>; // [H,D]
@group(0) @binding(1) var<storage, read> b1: array<f32>; // [H]
@group(0) @binding(2) var<storage, read> w2: array<f32>; // [OUT,H]
@group(0) @binding(3) var<storage, read> b2: array<f32>; // [OUT]
@group(0) @binding(4) var<storage, read_write> e: array<f32>; // [OUT]
@group(0) @binding(5) var<uniform> p: Params;

@compute @workgroup_size(1)
fn main() {
    var emb: array<f32, 32>;
    for (var i = 0u; i < 16u; i = i + 1u) {
        let freq = exp(-log(10000.0) * f32(i) / 16.0);
        emb[i] = sin(p.t * freq);
        emb[16u + i] = cos(p.t * freq);
    }
    var h: array<f32, 64>;
    for (var j = 0u; j < H; j = j + 1u) {
        var acc = b1[j];
        for (var d = 0u; d < D; d = d + 1u) { acc = acc + w1[j * D + d] * emb[d]; }
        h[j] = acc / (1.0 + exp(-acc));
    }
    for (var o = 0u; o < OUT; o = o + 1u) {
        var acc = b2[o];
        for (var j = 0u; j < H; j = j + 1u) { acc = acc + w2[o * H + j] * h[j]; }
        e[o] = acc;
    }
}
"#;

fn cpu_time(t: f32, w1: &[f32], b1: &[f32], w2: &[f32], b2: &[f32]) -> Vec<f32> {
    let mut emb = vec![0.0f32; D];
    for i in 0..16 {
        let freq = (-(10000.0f32).ln() * i as f32 / 16.0).exp();
        emb[i] = (t * freq).sin();
        emb[16 + i] = (t * freq).cos();
    }
    let mut h = vec![0.0f32; H];
    for j in 0..H {
        let mut acc = b1[j];
        for d in 0..D { acc += w1[j * D + d] * emb[d]; }
        h[j] = acc / (1.0 + (-acc).exp());
    }
    let mut e = vec![0.0f32; OUT];
    for o in 0..OUT {
        let mut acc = b2[o];
        for j in 0..H { acc += w2[o * H + j] * h[j]; }
        e[o] = acc;
    }
    e
}

fn main() {
    let t = 0.7f32;
    let mut w1 = vec![0.0f32; H * D];
    let mut b1 = vec![0.0f32; H];
    let mut w2 = vec![0.0f32; OUT * H];
    let mut b2 = vec![0.0f32; OUT];
    for i in 0..w1.len() { w1[i] = ((i * 17 % 97) as f32 - 48.0) / (D as f32 * 8.0); }
    for i in 0..b1.len() { b1[i] = ((i * 7 % 13) as f32 - 6.0) * 0.01; }
    for i in 0..w2.len() { w2[i] = ((i * 23 % 89) as f32 - 44.0) / (H as f32 * 8.0); }
    for i in 0..b2.len() { b2[i] = ((i * 5 % 11) as f32 - 5.0) * 0.01; }
    let expect = cpu_time(t, &w1, &b1, &w2, &b2);

    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor { backends: wgpu::Backends::VULKAN, ..Default::default() });
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance, compatible_surface: None, force_fallback_adapter: false,
    })).expect("no Vulkan adapter");
    let info = adapter.get_info();
    println!("adapter: {} | backend: {:?}", info.name, info.backend);
    assert_eq!(info.backend, wgpu::Backend::Vulkan);

    let mut limits = wgpu::Limits::downlevel_defaults();
    limits.max_storage_buffers_per_shader_stage = 8;
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("time-mlp"), required_features: wgpu::Features::empty(),
        required_limits: limits, ..Default::default()
    })).expect("device");

    let mk = |data: &[f32]| device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None, contents: bytemuck::cast_slice(data), usage: wgpu::BufferUsages::STORAGE,
    });
    let w1_buf = mk(&w1); let b1_buf = mk(&b1); let w2_buf = mk(&w2); let b2_buf = mk(&b2);
    let e_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: None, size: (OUT * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false,
    });
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: None, size: (OUT * 4) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false,
    });
    let params = Params { t, p0: 0, p1: 0, p2: 0 };
    let p_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None, contents: bytemuck::bytes_of(&params), usage: wgpu::BufferUsages::UNIFORM,
    });
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("time-mlp"), source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(WGSL)),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("time-mlp"), layout: None, module: &module, entry_point: Some("main"),
        compilation_options: Default::default(), cache: None,
    });
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None, layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: w1_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: b1_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: w2_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: b2_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: e_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: p_buf.as_entire_binding() },
        ],
    });
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
        pass.set_pipeline(&pipeline); pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    enc.copy_buffer_to_buffer(&e_buf, 0, &readback, 0, (OUT * 4) as u64);
    queue.submit(Some(enc.finish()));
    let slice = readback.slice(..);
    slice.map_async(wgpu::MapMode::Read, |r| r.expect("map"));
    device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
    let data = slice.get_mapped_range();
    let got: &[f32] = bytemuck::cast_slice(&data);
    let mut max_err = 0.0f32;
    for (a, b) in got.iter().zip(expect.iter()) { max_err = max_err.max((a - b).abs()); }
    println!("t={t} D={D} H={H} OUT={OUT} max_abs_err={max_err:.3e}");
    assert!(max_err < 1e-5, "time MLP mismatch");
    println!("VULKAN TIME MLP OK");
}
