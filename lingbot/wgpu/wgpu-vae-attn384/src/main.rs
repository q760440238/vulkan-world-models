use bytemuck::{Pod, Zeroable};
use std::borrow::Cow;
use wgpu::util::DeviceExt;

const C: usize = 384;
const T: usize = 4;

const WGSL: &str = r#"
const C: u32 = 384u; const T: u32 = 4u;
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> gamma: array<f32>;
@group(0) @binding(2) var<storage, read> wqkv: array<f32>;
@group(0) @binding(3) var<storage, read> bqkv: array<f32>;
@group(0) @binding(4) var<storage, read> wp: array<f32>;
@group(0) @binding(5) var<storage, read> bp: array<f32>;
@group(0) @binding(6) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= T) { return; }
    var h: array<f32, 384>;
    var q: array<f32, 384>;
    var kk: array<f32, 1536>;
    var vv: array<f32, 1536>;
    var scores: array<f32, 4>;
    var attn: array<f32, 384>;
    for (var c = 0u; c < C; c = c + 1u) { h[c] = x[c * T + i]; }
    var ss = 0.0;
    for (var c = 0u; c < C; c = c + 1u) { ss = ss + h[c] * h[c]; }
    let inv = sqrt(f32(C)) / sqrt(ss + 1e-12);
    for (var c = 0u; c < C; c = c + 1u) { h[c] = h[c] * inv * gamma[c]; }
    for (var d = 0u; d < C; d = d + 1u) {
        var acc = bqkv[d];
        for (var c = 0u; c < C; c = c + 1u) { acc = acc + h[c] * wqkv[d * C + c]; }
        q[d] = acc;
    }
    for (var j = 0u; j < T; j = j + 1u) {
        var hj: array<f32, 384>;
        for (var c = 0u; c < C; c = c + 1u) { hj[c] = x[c * T + j]; }
        var ssj = 0.0;
        for (var c = 0u; c < C; c = c + 1u) { ssj = ssj + hj[c] * hj[c]; }
        let ij = sqrt(f32(C)) / sqrt(ssj + 1e-12);
        for (var c = 0u; c < C; c = c + 1u) { hj[c] = hj[c] * ij * gamma[c]; }
        for (var d = 0u; d < C; d = d + 1u) {
            var kacc = bqkv[C + d]; var vacc = bqkv[2u * C + d];
            for (var c = 0u; c < C; c = c + 1u) {
                kacc = kacc + hj[c] * wqkv[(C + d) * C + c];
                vacc = vacc + hj[c] * wqkv[(2u * C + d) * C + c];
            }
            kk[j * C + d] = kacc; vv[j * C + d] = vacc;
        }
    }
    var maxs = -1e30;
    for (var j = 0u; j < T; j = j + 1u) {
        var dot = 0.0;
        for (var d = 0u; d < C; d = d + 1u) { dot = dot + q[d] * kk[j * C + d]; }
        scores[j] = dot / sqrt(f32(C)); maxs = max(maxs, scores[j]);
    }
    var sum = 0.0;
    for (var j = 0u; j < T; j = j + 1u) { sum = sum + exp(scores[j] - maxs); }
    for (var d = 0u; d < C; d = d + 1u) {
        var acc = 0.0;
        for (var j = 0u; j < T; j = j + 1u) { acc = acc + (exp(scores[j] - maxs) / sum) * vv[j * C + d]; }
        attn[d] = acc;
    }
    for (var c = 0u; c < C; c = c + 1u) {
        var acc = bp[c];
        for (var d = 0u; d < C; d = d + 1u) { acc = acc + attn[d] * wp[c * C + d]; }
        out[c * T + i] = x[c * T + i] + acc;
    }
}
"#;

fn cpu(x: &[f32], g: &[f32], wqkv: &[f32], bqkv: &[f32], wp: &[f32], bp: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; C * T];
    for i in 0..T {
        let mut h = vec![0.0f32; C];
        for c in 0..C { h[c] = x[c * T + i]; }
        let ss: f32 = h.iter().map(|v| v * v).sum();
        let inv = (C as f32).sqrt() / (ss + 1e-12).sqrt();
        for c in 0..C { h[c] = h[c] * inv * g[c]; }
        let q: Vec<f32> = (0..C).map(|d| bqkv[d] + (0..C).map(|c| h[c] * wqkv[d * C + c]).sum::<f32>()).collect();
        let mut kk = vec![0.0f32; T * C];
        let mut vv = vec![0.0f32; T * C];
        for j in 0..T {
            let mut hj = vec![0.0f32; C];
            for c in 0..C { hj[c] = x[c * T + j]; }
            let ssj: f32 = hj.iter().map(|v| v * v).sum();
            let ij = (C as f32).sqrt() / (ssj + 1e-12).sqrt();
            for c in 0..C { hj[c] = hj[c] * ij * g[c]; }
            for d in 0..C {
                kk[j * C + d] = bqkv[C + d] + (0..C).map(|c| hj[c] * wqkv[(C + d) * C + c]).sum::<f32>();
                vv[j * C + d] = bqkv[2 * C + d] + (0..C).map(|c| hj[c] * wqkv[(2 * C + d) * C + c]).sum::<f32>();
            }
        }
        let mut scores = vec![0.0f32; T];
        let mut maxs = f32::NEG_INFINITY;
        for j in 0..T {
            let dot: f32 = (0..C).map(|d| q[d] * kk[j * C + d]).sum();
            scores[j] = dot / (C as f32).sqrt(); maxs = maxs.max(scores[j]);
        }
        let sum: f32 = scores.iter().map(|s| (s - maxs).exp()).sum();
        let attn: Vec<f32> = (0..C).map(|d| (0..T).map(|j| (scores[j] - maxs).exp() / sum * vv[j * C + d]).sum()).collect();
        for c in 0..C { out[c * T + i] = x[c * T + i] + bp[c] + (0..C).map(|d| attn[d] * wp[c * C + d]).sum::<f32>(); }
    }
    out
}

fn main() {
    let mut x = vec![0.0f32; C * T];
    let mut g = vec![0.0f32; C];
    let mut wqkv = vec![0.0f32; 3 * C * C];
    let mut bqkv = vec![0.0f32; 3 * C];
    let mut wp = vec![0.0f32; C * C];
    let mut bp = vec![0.0f32; C];
    for i in 0..x.len() { x[i] = ((i * 13 % 71) as f32 - 35.0) / 19.0; }
    for i in 0..C { g[i] = 1.0 + ((i * 7 % 11) as f32 - 5.0) * 0.01; bp[i] = ((i * 5 % 13) as f32 - 6.0) * 0.01; }
    for i in 0..wqkv.len() { wqkv[i] = ((i * 17 % 97) as f32 - 48.0) / (C as f32 * 8.0); }
    for i in 0..3 * C { bqkv[i] = ((i * 7 % 11) as f32 - 5.0) * 0.01; }
    for i in 0..wp.len() { wp[i] = ((i * 23 % 89) as f32 - 44.0) / (C as f32 * 8.0); }
    let expect = cpu(&x, &g, &wqkv, &bqkv, &wp, &bp);
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor { backends: wgpu::Backends::VULKAN, ..Default::default() });
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions { power_preference: wgpu::PowerPreference::HighPerformance, compatible_surface: None, force_fallback_adapter: false })).expect("no adapter");
    let info = adapter.get_info();
    println!("adapter: {} | backend: {:?}", info.name, info.backend);
    let mut limits = wgpu::Limits::downlevel_defaults();
    limits.max_storage_buffers_per_shader_stage = 8;
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("vae-attn384"), required_features: wgpu::Features::empty(), required_limits: limits, ..Default::default()
    })).expect("device");
    let mk = |d: &[f32]| device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(d), usage: wgpu::BufferUsages::STORAGE });
    let xb=mk(&x); let gb=mk(&g); let wqb=mk(&wqkv); let bqb=mk(&bqkv); let wpb=mk(&wp); let bpb=mk(&bp);
    let ob = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (C*T*4) as u64, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
    let rb = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (C*T*4) as u64, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: None, source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(WGSL)) });
    let pipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: None, layout: None, module: &module, entry_point: Some("main"), compilation_options: Default::default(), cache: None });
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &pipe.get_bind_group_layout(0), entries: &[
        wgpu::BindGroupEntry { binding: 0, resource: xb.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 1, resource: gb.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 2, resource: wqb.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 3, resource: bqb.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 4, resource: wpb.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 5, resource: bpb.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 6, resource: ob.as_entire_binding() },
    ]});
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
        p.set_pipeline(&pipe); p.set_bind_group(0, &bg, &[]); p.dispatch_workgroups(T as u32, 1, 1);
    }
    enc.copy_buffer_to_buffer(&ob, 0, &rb, 0, (C*T*4) as u64);
    queue.submit(Some(enc.finish()));
    let slice = rb.slice(..); slice.map_async(wgpu::MapMode::Read, |r| r.expect("map"));
    device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
    let data = slice.get_mapped_range();
    let got: &[f32] = bytemuck::cast_slice(&data);
    let mut max_err = 0.0f32;
    for (a, b) in got.iter().zip(expect.iter()) { max_err = max_err.max((a - b).abs()); }
    println!("vae-attn384 C={C} T={T} single-head max_abs_err={max_err:.3e}");
    assert!(max_err < 1e-4, "attn384 mismatch");
    println!("VULKAN VAE ATTN384 OK");
}
