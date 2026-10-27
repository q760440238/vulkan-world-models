use bytemuck::{Pod, Zeroable};
use half::f16;
use std::borrow::Cow;
use wgpu::util::DeviceExt;

const D: usize = 32;
const L: usize = 4;
const F: usize = 64;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Params { scale: f32, eps: f32, _p0: u32, _p1: u32 }

const WGSL: &str = r#"
enable f16;
const D: u32 = 32u;
const L: u32 = 4u;
const F: u32 = 64u;
struct Params { scale: f32, eps: f32, p0: u32, p1: u32 };
@group(0) @binding(0) var<storage, read> x: array<f16>;
@group(0) @binding(1) var<storage, read> g1: array<f16>;
@group(0) @binding(2) var<storage, read> g2: array<f16>;
@group(0) @binding(3) var<storage, read> wq: array<f16>;
@group(0) @binding(4) var<storage, read> wk: array<f16>;
@group(0) @binding(5) var<storage, read> wv: array<f16>;
@group(0) @binding(6) var<storage, read> wo: array<f16>;
@group(0) @binding(7) var<storage, read> w1: array<f16>;
@group(0) @binding(8) var<storage, read> w2: array<f16>;
@group(0) @binding(9) var<storage, read_write> out: array<f16>;
@group(0) @binding(10) var<uniform> p: Params;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= L) { return; }
    var h: array<f32, 32>;
    var q: array<f32, 32>;
    var kk: array<f32, 128>;
    var vv: array<f32, 128>;
    var scores: array<f32, 4>;
    var attn: array<f32, 32>;
    var x1: array<f32, 32>;
    var h2: array<f32, 32>;
    var ffn: array<f32, 64>;

    var ss = 0.0;
    for (var d = 0u; d < D; d = d + 1u) { let v = f32(x[i * D + d]); ss = ss + v * v; }
    let inv = inverseSqrt(ss / f32(D) + p.eps);
    for (var d = 0u; d < D; d = d + 1u) { h[d] = f32(x[i * D + d]) * inv * f32(g1[d]); }
    for (var d = 0u; d < D; d = d + 1u) {
        var acc = 0.0;
        for (var e = 0u; e < D; e = e + 1u) { acc = acc + h[e] * f32(wq[d * D + e]); }
        q[d] = acc;
    }
    var maxs = -1e30;
    for (var j = 0u; j < L; j = j + 1u) {
        if (j > i) { break; }
        var hj: array<f32, 32>;
        var sj = 0.0;
        for (var d = 0u; d < D; d = d + 1u) { let v = f32(x[j * D + d]); sj = sj + v * v; }
        let ij = inverseSqrt(sj / f32(D) + p.eps);
        for (var d = 0u; d < D; d = d + 1u) { hj[d] = f32(x[j * D + d]) * ij * f32(g1[d]); }
        for (var d = 0u; d < D; d = d + 1u) {
            var ks = 0.0; var vs = 0.0;
            for (var e = 0u; e < D; e = e + 1u) {
                ks = ks + hj[e] * f32(wk[d * D + e]);
                vs = vs + hj[e] * f32(wv[d * D + e]);
            }
            kk[j * D + d] = ks; vv[j * D + d] = vs;
        }
        var dot = 0.0;
        for (var d = 0u; d < D; d = d + 1u) { dot = dot + q[d] * kk[j * D + d]; }
        scores[j] = dot * p.scale; maxs = max(maxs, scores[j]);
    }
    var sum = 0.0;
    for (var j = 0u; j <= i; j = j + 1u) { sum = sum + exp(scores[j] - maxs); }
    for (var d = 0u; d < D; d = d + 1u) {
        var acc = 0.0;
        for (var j = 0u; j <= i; j = j + 1u) { acc = acc + (exp(scores[j] - maxs) / sum) * vv[j * D + d]; }
        attn[d] = acc;
    }
    for (var d = 0u; d < D; d = d + 1u) {
        var acc = 0.0;
        for (var e = 0u; e < D; e = e + 1u) { acc = acc + attn[e] * f32(wo[d * D + e]); }
        x1[d] = f32(x[i * D + d]) + acc;
    }
    var s2 = 0.0;
    for (var d = 0u; d < D; d = d + 1u) { s2 = s2 + x1[d] * x1[d]; }
    let i2 = inverseSqrt(s2 / f32(D) + p.eps);
    for (var d = 0u; d < D; d = d + 1u) { h2[d] = x1[d] * i2 * f32(g2[d]); }
    for (var f = 0u; f < F; f = f + 1u) {
        var acc = 0.0;
        for (var d = 0u; d < D; d = d + 1u) { acc = acc + h2[d] * f32(w1[f * D + d]); }
        ffn[f] = acc / (1.0 + exp(-acc));
    }
    for (var d = 0u; d < D; d = d + 1u) {
        var acc = 0.0;
        for (var f = 0u; f < F; f = f + 1u) { acc = acc + ffn[f] * f32(w2[d * F + f]); }
        out[i * D + d] = f16(x1[d] + acc);
    }
}
"#;

fn rms(v: &[f32], g: &[f32], eps: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; D];
    let ss: f32 = v.iter().map(|x| x * x).sum();
    let inv = (ss / D as f32 + eps).sqrt().recip();
    for d in 0..D { out[d] = v[d] * inv * g[d]; }
    out
}
fn matvec(w: &[f32], n: usize, k: usize, v: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; n];
    for i in 0..n { for j in 0..k { out[i] += w[i * k + j] * v[j]; } }
    out
}
fn cpu_block(x: &[f32], g1: &[f32], g2: &[f32], wq: &[f32], wk: &[f32], wv: &[f32], wo: &[f32],
             w1: &[f32], w2: &[f32], eps: f32, scale: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; L * D];
    for i in 0..L {
        let h = rms(&x[i * D..(i + 1) * D], g1, eps);
        let q = matvec(wq, D, D, &h);
        let mut kk = vec![0.0f32; L * D];
        let mut vv = vec![0.0f32; L * D];
        let mut scores = vec![0.0f32; L];
        let mut maxs = f32::NEG_INFINITY;
        for j in 0..=i {
            let hj = rms(&x[j * D..(j + 1) * D], g1, eps);
            let k = matvec(wk, D, D, &hj);
            let v = matvec(wv, D, D, &hj);
            kk[j * D..(j + 1) * D].copy_from_slice(&k);
            vv[j * D..(j + 1) * D].copy_from_slice(&v);
            scores[j] = (0..D).map(|d| q[d] * k[d]).sum::<f32>() * scale;
            maxs = maxs.max(scores[j]);
        }
        let sum: f32 = scores[..=i].iter().map(|s| (s - maxs).exp()).sum();
        let mut attn = vec![0.0f32; D];
        for d in 0..D { for j in 0..=i { attn[d] += ((scores[j] - maxs).exp() / sum) * vv[j * D + d]; } }
        let o = matvec(wo, D, D, &attn);
        let mut x1 = vec![0.0f32; D];
        for d in 0..D { x1[d] = x[i * D + d] + o[d]; }
        let h2 = rms(&x1, g2, eps);
        let mut ffn = matvec(w1, F, D, &h2);
        for f in 0..F { ffn[f] = ffn[f] / (1.0 + (-ffn[f]).exp()); }
        let o2 = matvec(w2, D, F, &ffn);
        for d in 0..D { out[i * D + d] = x1[d] + o2[d]; }
    }
    out
}

fn main() {
    let eps = 1e-6f32;
    let scale = 1.0 / (D as f32).sqrt();
    let mut xf = vec![0.0f32; L * D];
    let mut g1f = vec![0.0f32; D];
    let mut g2f = vec![0.0f32; D];
    let mut wqf = vec![0.0f32; D * D];
    let mut wkf = vec![0.0f32; D * D];
    let mut wvf = vec![0.0f32; D * D];
    let mut wof = vec![0.0f32; D * D];
    let mut w1f = vec![0.0f32; F * D];
    let mut w2f = vec![0.0f32; D * F];
    for i in 0..xf.len() { xf[i] = ((i * 13 % 71) as f32 - 35.0) / 19.0; }
    for i in 0..D {
        g1f[i] = 1.0 + ((i * 7 % 11) as f32 - 5.0) * 0.01;
        g2f[i] = 1.0 + ((i * 5 % 13) as f32 - 6.0) * 0.01;
    }
    let fill = |dst: &mut [f32], k: usize| { for i in 0..dst.len() { dst[i] = ((i * 17 % 97) as f32 - 48.0) / (k as f32 * 8.0); } };
    fill(&mut wqf, D); fill(&mut wkf, D); fill(&mut wvf, D); fill(&mut wof, D);
    fill(&mut w1f, D); fill(&mut w2f, F);
    let x16: Vec<f16> = xf.iter().map(|v| f16::from_f32(*v)).collect();
    let g1_16: Vec<f16> = g1f.iter().map(|v| f16::from_f32(*v)).collect();
    let g2_16: Vec<f16> = g2f.iter().map(|v| f16::from_f32(*v)).collect();
    let wq16: Vec<f16> = wqf.iter().map(|v| f16::from_f32(*v)).collect();
    let wk16: Vec<f16> = wkf.iter().map(|v| f16::from_f32(*v)).collect();
    let wv16: Vec<f16> = wvf.iter().map(|v| f16::from_f32(*v)).collect();
    let wo16: Vec<f16> = wof.iter().map(|v| f16::from_f32(*v)).collect();
    let w1_16: Vec<f16> = w1f.iter().map(|v| f16::from_f32(*v)).collect();
    let w2_16: Vec<f16> = w2f.iter().map(|v| f16::from_f32(*v)).collect();
    let xq: Vec<f32> = x16.iter().map(|v| v.to_f32()).collect();
    let g1q: Vec<f32> = g1_16.iter().map(|v| v.to_f32()).collect();
    let g2q: Vec<f32> = g2_16.iter().map(|v| v.to_f32()).collect();
    let wqq: Vec<f32> = wq16.iter().map(|v| v.to_f32()).collect();
    let wkq: Vec<f32> = wk16.iter().map(|v| v.to_f32()).collect();
    let wvq: Vec<f32> = wv16.iter().map(|v| v.to_f32()).collect();
    let woq: Vec<f32> = wo16.iter().map(|v| v.to_f32()).collect();
    let w1q: Vec<f32> = w1_16.iter().map(|v| v.to_f32()).collect();
    let w2q: Vec<f32> = w2_16.iter().map(|v| v.to_f32()).collect();
    let expect = cpu_block(&xq, &g1q, &g2q, &wqq, &wkq, &wvq, &woq, &w1q, &w2q, eps, scale);

    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor { backends: wgpu::Backends::VULKAN, ..Default::default() });
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance, compatible_surface: None, force_fallback_adapter: false,
    })).expect("no Vulkan adapter");
    let info = adapter.get_info();
    println!("adapter: {} | backend: {:?}", info.name, info.backend);
    assert_eq!(info.backend, wgpu::Backend::Vulkan);
    let mut limits = wgpu::Limits::downlevel_defaults();
    limits.max_storage_buffers_per_shader_stage = 16;
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("dit-fp16"), required_features: wgpu::Features::SHADER_F16,
        required_limits: limits, ..Default::default()
    })).expect("device");
    let mk = |data: &[f16]| device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None, contents: bytemuck::cast_slice(data), usage: wgpu::BufferUsages::STORAGE,
    });
    let x_buf = mk(&x16); let g1_buf = mk(&g1_16); let g2_buf = mk(&g2_16);
    let wq_buf = mk(&wq16); let wk_buf = mk(&wk16); let wv_buf = mk(&wv16); let wo_buf = mk(&wo16);
    let w1_buf = mk(&w1_16); let w2_buf = mk(&w2_16);
    let out_buf = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (L * D * 2) as u64, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
    let readback = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (L * D * 2) as u64, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
    let params = Params { scale, eps, _p0: 0, _p1: 0 };
    let p_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::bytes_of(&params), usage: wgpu::BufferUsages::UNIFORM });
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: None, source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(WGSL)) });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: None, layout: None, module: &module, entry_point: Some("main"), compilation_options: Default::default(), cache: None });
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &pipeline.get_bind_group_layout(0), entries: &[
        wgpu::BindGroupEntry { binding: 0, resource: x_buf.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 1, resource: g1_buf.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 2, resource: g2_buf.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 3, resource: wq_buf.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 4, resource: wk_buf.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 5, resource: wv_buf.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 6, resource: wo_buf.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 7, resource: w1_buf.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 8, resource: w2_buf.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 9, resource: out_buf.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 10, resource: p_buf.as_entire_binding() },
    ]});
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
        p.set_pipeline(&pipeline); p.set_bind_group(0, &bg, &[]); p.dispatch_workgroups(L as u32, 1, 1);
    }
    enc.copy_buffer_to_buffer(&out_buf, 0, &readback, 0, (L * D * 2) as u64);
    queue.submit(Some(enc.finish()));
    let slice = readback.slice(..);
    slice.map_async(wgpu::MapMode::Read, |r| r.expect("map"));
    device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
    let data = slice.get_mapped_range();
    let got: &[f16] = bytemuck::cast_slice(&data);
    let mut max_err = 0.0f32;
    for (a, b) in got.iter().zip(expect.iter()) { max_err = max_err.max((a.to_f32() - b).abs()); }
    println!("fp16 block L={L} D={D} F={F} max_abs_err={max_err:.3e}");
    assert!(max_err < 2e-2, "fp16 block mismatch");
    println!("VULKAN FP16 DIT BLOCK OK");
}
