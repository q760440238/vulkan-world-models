use bytemuck::{Pod, Zeroable};
use std::borrow::Cow;
use wgpu::util::DeviceExt;

const T: usize = 8;
const DIM: usize = 32;
const HEADS: usize = 4;
const HD: usize = 8;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Params { scale: f32, _p0: u32, _p1: u32, _p2: u32 }

const WGSL: &str = r#"
const T: u32 = 8u;
const DIM: u32 = 32u;
const HEADS: u32 = 4u;
const HD: u32 = 8u;
struct Params { scale: f32, p0: u32, p1: u32, p2: u32 };
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> wq: array<f32>;
@group(0) @binding(2) var<storage, read> wk: array<f32>;
@group(0) @binding(3) var<storage, read> wv: array<f32>;
@group(0) @binding(4) var<storage, read> wo: array<f32>;
@group(0) @binding(5) var<storage, read_write> out: array<f32>;
@group(0) @binding(6) var<uniform> p: Params;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= T) { return; }
    var q: array<f32, 32>;
    var kk: array<f32, 256>;
    var vv: array<f32, 256>;
    var scores: array<f32, 32>;
    var attn: array<f32, 32>;

    for (var d = 0u; d < DIM; d = d + 1u) {
        var acc = 0.0;
        for (var e = 0u; e < DIM; e = e + 1u) { acc = acc + x[i * DIM + e] * wq[d * DIM + e]; }
        q[d] = acc;
    }
    for (var j = 0u; j < T; j = j + 1u) {
        for (var d = 0u; d < DIM; d = d + 1u) {
            var ks = 0.0; var vs = 0.0;
            for (var e = 0u; e < DIM; e = e + 1u) {
                ks = ks + x[j * DIM + e] * wk[d * DIM + e];
                vs = vs + x[j * DIM + e] * wv[d * DIM + e];
            }
            kk[j * DIM + d] = ks; vv[j * DIM + d] = vs;
        }
    }
    for (var hh = 0u; hh < HEADS; hh = hh + 1u) {
        var maxs = -1e30;
        for (var j = 0u; j < T; j = j + 1u) {
            var dot = 0.0;
            for (var d = 0u; d < HD; d = d + 1u) { dot = dot + q[hh * HD + d] * kk[j * DIM + hh * HD + d]; }
            scores[hh * T + j] = dot * p.scale;
            maxs = max(maxs, scores[hh * T + j]);
        }
        var sum = 0.0;
        for (var j = 0u; j < T; j = j + 1u) { sum = sum + exp(scores[hh * T + j] - maxs); }
        for (var d = 0u; d < HD; d = d + 1u) {
            var acc = 0.0;
            for (var j = 0u; j < T; j = j + 1u) {
                acc = acc + (exp(scores[hh * T + j] - maxs) / sum) * vv[j * DIM + hh * HD + d];
            }
            attn[hh * HD + d] = acc;
        }
    }
    for (var d = 0u; d < DIM; d = d + 1u) {
        var acc = 0.0;
        for (var e = 0u; e < DIM; e = e + 1u) { acc = acc + attn[e] * wo[d * DIM + e]; }
        out[i * DIM + d] = x[i * DIM + d] + acc;
    }
}
"#;

fn cpu_attn(x: &[f32], wq: &[f32], wk: &[f32], wv: &[f32], wo: &[f32], scale: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; T * DIM];
    for i in 0..T {
        let q: Vec<f32> = (0..DIM).map(|d| (0..DIM).map(|e| x[i * DIM + e] * wq[d * DIM + e]).sum()).collect();
        let mut kk = vec![0.0f32; T * DIM];
        let mut vv = vec![0.0f32; T * DIM];
        for j in 0..T {
            for d in 0..DIM {
                kk[j * DIM + d] = (0..DIM).map(|e| x[j * DIM + e] * wk[d * DIM + e]).sum();
                vv[j * DIM + d] = (0..DIM).map(|e| x[j * DIM + e] * wv[d * DIM + e]).sum();
            }
        }
        let mut attn = vec![0.0f32; DIM];
        for hh in 0..HEADS {
            let mut scores = vec![0.0f32; T];
            let mut maxs = f32::NEG_INFINITY;
            for j in 0..T {
                let dot: f32 = (0..HD).map(|d| q[hh * HD + d] * kk[j * DIM + hh * HD + d]).sum();
                scores[j] = dot * scale; maxs = maxs.max(scores[j]);
            }
            let sum: f32 = scores.iter().map(|s| (s - maxs).exp()).sum();
            for d in 0..HD {
                attn[hh * HD + d] = (0..T).map(|j| (scores[j] - maxs).exp() / sum * vv[j * DIM + hh * HD + d]).sum();
            }
        }
        for d in 0..DIM {
            let o: f32 = (0..DIM).map(|e| attn[e] * wo[d * DIM + e]).sum();
            out[i * DIM + d] = x[i * DIM + d] + o;
        }
    }
    out
}

fn main() {
    let scale = 1.0 / (HD as f32).sqrt();
    let mut x = vec![0.0f32; T * DIM];
    let mut wq = vec![0.0f32; DIM * DIM];
    let mut wk = vec![0.0f32; DIM * DIM];
    let mut wv = vec![0.0f32; DIM * DIM];
    let mut wo = vec![0.0f32; DIM * DIM];
    for i in 0..x.len() { x[i] = ((i * 13 % 71) as f32 - 35.0) / 19.0; }
    let fill = |dst: &mut [f32], k: usize| { for i in 0..dst.len() { dst[i] = ((i * 17 % 97) as f32 - 48.0) / (k as f32 * 8.0); } };
    fill(&mut wq, DIM); fill(&mut wk, DIM); fill(&mut wv, DIM); fill(&mut wo, DIM);
    let expect = cpu_attn(&x, &wq, &wk, &wv, &wo, scale);

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
        label: Some("vae-attn"), required_features: wgpu::Features::empty(),
        required_limits: limits, ..Default::default()
    })).expect("device");
    let mk = |data: &[f32]| device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None, contents: bytemuck::cast_slice(data), usage: wgpu::BufferUsages::STORAGE,
    });
    let x_buf = mk(&x); let wq_buf = mk(&wq); let wk_buf = mk(&wk); let wv_buf = mk(&wv); let wo_buf = mk(&wo);
    let out_buf = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (T * DIM * 4) as u64, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
    let readback = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (T * DIM * 4) as u64, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
    let params = Params { scale, _p0: 0, _p1: 0, _p2: 0 };
    let p_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::bytes_of(&params), usage: wgpu::BufferUsages::UNIFORM });
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: None, source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(WGSL)) });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: None, layout: None, module: &module, entry_point: Some("main"), compilation_options: Default::default(), cache: None });
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &pipeline.get_bind_group_layout(0), entries: &[
        wgpu::BindGroupEntry { binding: 0, resource: x_buf.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 1, resource: wq_buf.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 2, resource: wk_buf.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 3, resource: wv_buf.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 4, resource: wo_buf.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 5, resource: out_buf.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 6, resource: p_buf.as_entire_binding() },
    ]});
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
        p.set_pipeline(&pipeline); p.set_bind_group(0, &bg, &[]); p.dispatch_workgroups(T as u32, 1, 1);
    }
    enc.copy_buffer_to_buffer(&out_buf, 0, &readback, 0, (T * DIM * 4) as u64);
    queue.submit(Some(enc.finish()));
    let slice = readback.slice(..);
    slice.map_async(wgpu::MapMode::Read, |r| r.expect("map"));
    device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
    let data = slice.get_mapped_range();
    let got: &[f32] = bytemuck::cast_slice(&data);
    let mut max_err = 0.0f32;
    for (a, b) in got.iter().zip(expect.iter()) { max_err = max_err.max((a - b).abs()); }
    println!("vae-attn T={T} DIM={DIM} HEADS={HEADS} HD={HD} max_abs_err={max_err:.3e}");
    assert!(max_err < 1e-5, "vae attn mismatch");
    println!("VULKAN VAE ATTENTION OK");
}
