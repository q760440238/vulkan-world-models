use bytemuck::{Pod, Zeroable};
use std::borrow::Cow;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Params { scale: f32, eps: f32, _p0: u32, _p1: u32 }

const D: usize = 32;
const L: usize = 4;
const F: usize = 64;

const WGSL: &str = r#"
const D: u32 = 32u;
const L: u32 = 4u;
const F: u32 = 64u;
struct Params { scale: f32, eps: f32, p0: u32, p1: u32 };
@group(0) @binding(0) var<storage, read> x: array<f32>;      // [L,D]
@group(0) @binding(1) var<storage, read> g1: array<f32>;     // [D]
@group(0) @binding(2) var<storage, read> g2: array<f32>;     // [D]
@group(0) @binding(3) var<storage, read> wq: array<f32>;     // [D,D]
@group(0) @binding(4) var<storage, read> wk: array<f32>;
@group(0) @binding(5) var<storage, read> wv: array<f32>;
@group(0) @binding(6) var<storage, read> wo: array<f32>;
@group(0) @binding(7) var<storage, read> w1: array<f32>;     // [F,D]
@group(0) @binding(8) var<storage, read> w2: array<f32>;     // [D,F]
@group(0) @binding(9) var<storage, read_write> out: array<f32>; // [L,D]
@group(0) @binding(10) var<storage, read> e: array<f32>; // [6,D]
@group(0) @binding(11) var<uniform> p: Params;

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

    // h = RMSNorm(x_i) * g1
    var ss = 0.0;
    for (var d = 0u; d < D; d = d + 1u) { ss = ss + x[i * D + d] * x[i * D + d]; }
    let inv = inverseSqrt(ss / f32(D) + p.eps);
    for (var d = 0u; d < D; d = d + 1u) { h[d] = (x[i * D + d] * inv * g1[d]) * (1.0 + e[D + d]) + e[d]; }

    // q
    for (var d = 0u; d < D; d = d + 1u) {
        var acc = 0.0;
        for (var e = 0u; e < D; e = e + 1u) { acc = acc + h[e] * wq[d * D + e]; }
        q[d] = acc;
    }

    // K/V for causal keys, and scores
    var maxs = -1e30;
    for (var j = 0u; j < L; j = j + 1u) {
        if (j > i) { break; }
        var hj: array<f32, 32>;
        var sj = 0.0;
        for (var d = 0u; d < D; d = d + 1u) { sj = sj + x[j * D + d] * x[j * D + d]; }
        let ij = inverseSqrt(sj / f32(D) + p.eps);
        for (var d = 0u; d < D; d = d + 1u) { hj[d] = (x[j * D + d] * ij * g1[d]) * (1.0 + e[D + d]) + e[d]; }
        for (var d = 0u; d < D; d = d + 1u) {
            var ks = 0.0;
            var vs = 0.0;
            for (var e = 0u; e < D; e = e + 1u) {
                ks = ks + hj[e] * wk[d * D + e];
                vs = vs + hj[e] * wv[d * D + e];
            }
            kk[j * D + d] = ks;
            vv[j * D + d] = vs;
        }
        var dot = 0.0;
        for (var d = 0u; d < D; d = d + 1u) { dot = dot + q[d] * kk[j * D + d]; }
        scores[j] = dot * p.scale;
        maxs = max(maxs, scores[j]);
    }

    var sum = 0.0;
    for (var j = 0u; j <= i; j = j + 1u) { sum = sum + exp(scores[j] - maxs); }
    for (var d = 0u; d < D; d = d + 1u) {
        var acc = 0.0;
        for (var j = 0u; j <= i; j = j + 1u) {
            acc = acc + (exp(scores[j] - maxs) / sum) * vv[j * D + d];
        }
        attn[d] = acc;
    }

    // out projection + residual
    for (var d = 0u; d < D; d = d + 1u) {
        var acc = 0.0;
        for (var e = 0u; e < D; e = e + 1u) { acc = acc + attn[e] * wo[d * D + e]; }
        x1[d] = x[i * D + d] + acc * e[2u * D + d];
    }

    // h2 = RMSNorm(x1) * g2
    var s2 = 0.0;
    for (var d = 0u; d < D; d = d + 1u) { s2 = s2 + x1[d] * x1[d]; }
    let i2 = inverseSqrt(s2 / f32(D) + p.eps);
    for (var d = 0u; d < D; d = d + 1u) { h2[d] = (x1[d] * i2 * g2[d]) * (1.0 + e[4u * D + d]) + e[3u * D + d]; }

    // FFN: W1 -> SiLU -> W2 + residual
    for (var f = 0u; f < F; f = f + 1u) {
        var acc = 0.0;
        for (var d = 0u; d < D; d = d + 1u) { acc = acc + h2[d] * w1[f * D + d]; }
        ffn[f] = acc / (1.0 + exp(-acc));
    }
    for (var d = 0u; d < D; d = d + 1u) {
        var acc = 0.0;
        for (var f = 0u; f < F; f = f + 1u) { acc = acc + ffn[f] * w2[d * F + f]; }
        out[i * D + d] = x1[d] + acc * e[5u * D + d];
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
    for i in 0..n {
        let mut acc = 0.0f32;
        for j in 0..k { acc += w[i * k + j] * v[j]; }
        out[i] = acc;
    }
    out
}
fn cpu_block(x: &[f32], g1: &[f32], g2: &[f32], wq: &[f32], wk: &[f32], wv: &[f32], wo: &[f32],
             w1: &[f32], w2: &[f32], e: &[f32], eps: f32, scale: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; L * D];
    for i in 0..L {
        let h0 = rms(&x[i * D..(i + 1) * D], g1, eps);
        let mut h = vec![0.0f32; D];
        for d in 0..D { h[d] = h0[d] * (1.0 + e[D + d]) + e[d]; }
        let q = matvec(wq, D, D, &h);
        let mut kk = vec![0.0f32; L * D];
        let mut vv = vec![0.0f32; L * D];
        let mut scores = vec![0.0f32; L];
        let mut maxs = f32::NEG_INFINITY;
        for j in 0..=i {
            let hj0 = rms(&x[j * D..(j + 1) * D], g1, eps);
            let mut hj = vec![0.0f32; D];
            for d in 0..D { hj[d] = hj0[d] * (1.0 + e[D + d]) + e[d]; }
            let k = matvec(wk, D, D, &hj);
            let v = matvec(wv, D, D, &hj);
            kk[j * D..(j + 1) * D].copy_from_slice(&k);
            vv[j * D..(j + 1) * D].copy_from_slice(&v);
            let dot: f32 = (0..D).map(|d| q[d] * k[d]).sum();
            scores[j] = dot * scale;
            maxs = maxs.max(scores[j]);
        }
        let sum: f32 = scores[..=i].iter().map(|s| (s - maxs).exp()).sum();
        let mut attn = vec![0.0f32; D];
        for d in 0..D {
            for j in 0..=i { attn[d] += ((scores[j] - maxs).exp() / sum) * vv[j * D + d]; }
        }
        let o = matvec(wo, D, D, &attn);
        let mut x1 = vec![0.0f32; D];
        for d in 0..D { x1[d] = x[i * D + d] + o[d] * e[2 * D + d]; }
        let h20 = rms(&x1, g2, eps);
        let mut h2 = vec![0.0f32; D];
        for d in 0..D { h2[d] = h20[d] * (1.0 + e[4 * D + d]) + e[3 * D + d]; }
        let mut ffn = matvec(w1, F, D, &h2);
        for f in 0..F { ffn[f] = ffn[f] / (1.0 + (-ffn[f]).exp()); }
        let o2 = matvec(w2, D, F, &ffn);
        for d in 0..D { out[i * D + d] = x1[d] + o2[d] * e[5 * D + d]; }
    }
    out
}

fn main() {
    let eps = 1e-6f32;
    let scale = 1.0 / (D as f32).sqrt();
    let mut x = vec![0.0f32; L * D];
    let mut g1 = vec![0.0f32; D];
    let mut g2 = vec![0.0f32; D];
    let mut wq = vec![0.0f32; D * D];
    let mut wk = vec![0.0f32; D * D];
    let mut wv = vec![0.0f32; D * D];
    let mut wo = vec![0.0f32; D * D];
    let mut w1 = vec![0.0f32; F * D];
    let mut w2 = vec![0.0f32; D * F];
    let mut e = vec![0.0f32; 6 * D];
    for i in 0..x.len() { x[i] = ((i * 13 % 71) as f32 - 35.0) / 19.0; }
    for i in 0..D {
        g1[i] = 1.0 + ((i * 7 % 11) as f32 - 5.0) * 0.01;
        g2[i] = 1.0 + ((i * 5 % 13) as f32 - 6.0) * 0.01;
    }
    let fill = |dst: &mut [f32], k: usize| {
        for i in 0..dst.len() { dst[i] = ((i * 17 % 97) as f32 - 48.0) / (k as f32 * 8.0); }
    };
    fill(&mut wq, D); fill(&mut wk, D); fill(&mut wv, D); fill(&mut wo, D);
    fill(&mut w1, D); fill(&mut w2, F);
    for cc in 0..6 {
        for d in 0..D {
            let gate = cc == 2 || cc == 5;
            e[cc * D + d] = if gate { 1.0 + ((d * 3 + cc) as f32 % 11.0 - 5.0) * 0.03 }
                            else { ((d * 7 + cc * 13) as f32 % 9.0 - 4.0) * 0.02 };
        }
    }
    let expect = cpu_block(&x, &g1, &g2, &wq, &wk, &wv, &wo, &w1, &w2, &e, eps, scale);

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
        label: Some("dit-block"), required_features: wgpu::Features::empty(),
        required_limits: limits, ..Default::default()
    })).expect("device");

    let mk = |data: &[f32]| device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None, contents: bytemuck::cast_slice(data), usage: wgpu::BufferUsages::STORAGE,
    });
    let x_buf = mk(&x); let g1_buf = mk(&g1); let g2_buf = mk(&g2);
    let wq_buf = mk(&wq); let wk_buf = mk(&wk); let wv_buf = mk(&wv); let wo_buf = mk(&wo);
    let w1_buf = mk(&w1); let w2_buf = mk(&w2); let e_buf = mk(&e);
    let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: None, size: (L * D * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false,
    });
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: None, size: (L * D * 4) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false,
    });
    let params = Params { scale, eps, _p0: 0, _p1: 0 };
    let p_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None, contents: bytemuck::bytes_of(&params), usage: wgpu::BufferUsages::UNIFORM,
    });
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("dit-block"), source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(WGSL)),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("dit-block"), layout: None, module: &module, entry_point: Some("main"),
        compilation_options: Default::default(), cache: None,
    });
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None, layout: &pipeline.get_bind_group_layout(0),
        entries: &[
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
            wgpu::BindGroupEntry { binding: 10, resource: e_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 11, resource: p_buf.as_entire_binding() },
        ],
    });
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
        pass.set_pipeline(&pipeline); pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(L as u32, 1, 1);
    }
    enc.copy_buffer_to_buffer(&out_buf, 0, &readback, 0, (L * D * 4) as u64);
    queue.submit(Some(enc.finish()));
    let slice = readback.slice(..);
    slice.map_async(wgpu::MapMode::Read, |r| r.expect("map"));
    device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
    let data = slice.get_mapped_range();
    let got: &[f32] = bytemuck::cast_slice(&data);
    let mut max_err = 0.0f32;
    for (a, b) in got.iter().zip(expect.iter()) { max_err = max_err.max((a - b).abs()); }
    println!("L={L} D={D} F={F} max_abs_err={max_err:.3e}");
    assert!(max_err < 1e-4, "DiT block mismatch");
    println!("VULKAN MICRO DIT BLOCK OK");
}








