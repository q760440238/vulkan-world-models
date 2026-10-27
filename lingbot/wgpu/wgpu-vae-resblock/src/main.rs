use bytemuck::{Pod, Zeroable};
use std::borrow::Cow;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct ConvParams {
    n: u32, cin: u32, cout: u32, din: u32, hin: u32, win: u32,
    kd: u32, kh: u32, kw: u32, pad: u32, stride: u32,
    dout: u32, hout: u32, wout: u32, _p0: u32, _p1: u32,
}
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct GNParams { n: u32, c: u32, d: u32, h: u32, w: u32, groups: u32, eps: f32, _p: u32 }
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct AddParams { n: u32, _p0: u32, _p1: u32, _p2: u32 }
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct UpP { n: u32, c: u32, din: u32, hin: u32, win: u32, scale: u32, dout: u32, hout: u32, wout: u32, _p0: u32, _p1: u32, _p2: u32 }
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct AttnP { scale: f32, _p0: u32, _p1: u32, _p2: u32 }

const CONV_WGSL: &str = r#"
struct P { n: u32, cin: u32, cout: u32, din: u32, hin: u32, win: u32, kd: u32, kh: u32, kw: u32,
           pad: u32, stride: u32, dout: u32, hout: u32, wout: u32, p0: u32, p1: u32 };
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@group(0) @binding(4) var<uniform> p: P;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = p.n * p.cout * p.dout * p.hout * p.wout;
    if (idx >= total) { return; }
    var t = idx;
    let ow = t % p.wout; t = t / p.wout;
    let oh = t % p.hout; t = t / p.hout;
    let od = t % p.dout; t = t / p.dout;
    let co = t % p.cout;
    let bn = t / p.cout;
    var acc = b[co];
    for (var ci = 0u; ci < p.cin; ci = ci + 1u) {
        for (var kd = 0u; kd < p.kd; kd = kd + 1u) {
            let id = i32(od * p.stride + kd) - i32(p.pad);
            if (id < 0 || id >= i32(p.din)) { continue; }
            for (var kh = 0u; kh < p.kh; kh = kh + 1u) {
                let ih = i32(oh * p.stride + kh) - i32(p.pad);
                if (ih < 0 || ih >= i32(p.hin)) { continue; }
                for (var kw = 0u; kw < p.kw; kw = kw + 1u) {
                    let iw = i32(ow * p.stride + kw) - i32(p.pad);
                    if (iw < 0 || iw >= i32(p.win)) { continue; }
                    let xi = (((bn * p.cin + ci) * p.din + u32(id)) * p.hin + u32(ih)) * p.win + u32(iw);
                    let wi = (((co * p.cin + ci) * p.kd + kd) * p.kh + kh) * p.kw + kw;
                    acc = acc + x[xi] * w[wi];
                }
            }
        }
    }
    y[idx] = acc;
}
"#;
const GN_WGSL: &str = r#"
struct P { n: u32, c: u32, d: u32, h: u32, w: u32, groups: u32, eps: f32, p: u32 };
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> gamma: array<f32>;
@group(0) @binding(2) var<storage, read> beta: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@group(0) @binding(4) var<uniform> p: P;
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tg = p.n * p.groups;
    if (gid.x >= tg) { return; }
    let bn = gid.x / p.groups;
    let g = gid.x % p.groups;
    let cg = p.c / p.groups;
    let c0 = g * cg;
    let spatial = p.d * p.h * p.w;
    let count = cg * spatial;
    var sum = 0.0; var sumsq = 0.0;
    for (var c = c0; c < c0 + cg; c = c + 1u) {
        for (var s = 0u; s < spatial; s = s + 1u) {
            let v = x[(bn * p.c + c) * spatial + s];
            sum = sum + v; sumsq = sumsq + v * v;
        }
    }
    let mean = sum / f32(count);
    let variance = sumsq / f32(count) - mean * mean;
    let inv = inverseSqrt(variance + p.eps);
    for (var c = c0; c < c0 + cg; c = c + 1u) {
        for (var s = 0u; s < spatial; s = s + 1u) {
            let idx = (bn * p.c + c) * spatial + s;
            let norm = (x[idx] - mean) * inv;
            let v = norm * gamma[c] + beta[c];
            y[idx] = v / (1.0 + exp(-v));
        }
    }
}
"#;
const UP_WGSL: &str = r#"
struct P { n: u32, c: u32, din: u32, hin: u32, win: u32, scale: u32, dout: u32, hout: u32, wout: u32, p0: u32, p1: u32, p2: u32 };
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;
@group(0) @binding(2) var<uniform> p: P;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = p.n * p.c * p.dout * p.hout * p.wout;
    if (idx >= total) { return; }
    var t = idx;
    let ow = t % p.wout; t = t / p.wout;
    let oh = t % p.hout; t = t / p.hout;
    let od = t % p.dout; t = t / p.dout;
    let co = t % p.c;
    let bn = t / p.c;
    let iw = ow / p.scale; let ih = oh / p.scale; let id = od / p.scale;
    y[idx] = x[(((bn * p.c + co) * p.din + id) * p.hin + ih) * p.win + iw];
}
"#;const LIN_IN_WGSL: &str = r#"
const T: u32 = 8u; const C: u32 = 2u; const DIM: u32 = 32u;
@group(0) @binding(0) var<storage, read> x: array<f32>;  // [C,T]
@group(0) @binding(1) var<storage, read> w: array<f32>;  // [DIM,C]
@group(0) @binding(2) var<storage, read_write> y: array<f32>; // [T,DIM]
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= T * DIM) { return; }
    let t = idx / DIM; let d = idx % DIM;
    var acc = 0.0;
    for (var c = 0u; c < C; c = c + 1u) { acc = acc + x[c * T + t] * w[d * C + c]; }
    y[t * DIM + d] = acc;
}
"#;
const LIN_OUT_WGSL: &str = r#"
const T: u32 = 8u; const C: u32 = 2u; const DIM: u32 = 32u;
@group(0) @binding(0) var<storage, read> x: array<f32>;  // [T,DIM]
@group(0) @binding(1) var<storage, read> w: array<f32>;  // [C,DIM]
@group(0) @binding(2) var<storage, read_write> y: array<f32>; // [C,T]
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= C * T) { return; }
    let c = idx / T; let t = idx % T;
    var acc = 0.0;
    for (var d = 0u; d < DIM; d = d + 1u) { acc = acc + x[t * DIM + d] * w[c * DIM + d]; }
    y[c * T + t] = acc;
}
"#;
const ATTN_WGSL: &str = r#"
const T: u32 = 8u; const DIM: u32 = 32u; const HEADS: u32 = 4u; const HD: u32 = 8u;
struct P { scale: f32, p0: u32, p1: u32, p2: u32 };
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> wq: array<f32>;
@group(0) @binding(2) var<storage, read> wk: array<f32>;
@group(0) @binding(3) var<storage, read> wv: array<f32>;
@group(0) @binding(4) var<storage, read> wo: array<f32>;
@group(0) @binding(5) var<storage, read_write> out: array<f32>;
@group(0) @binding(6) var<uniform> p: P;
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
            scores[hh * T + j] = dot * p.scale; maxs = max(maxs, scores[hh * T + j]);
        }
        var sum = 0.0;
        for (var j = 0u; j < T; j = j + 1u) { sum = sum + exp(scores[hh * T + j] - maxs); }
        for (var d = 0u; d < HD; d = d + 1u) {
            var acc = 0.0;
            for (var j = 0u; j < T; j = j + 1u) { acc = acc + (exp(scores[hh * T + j] - maxs) / sum) * vv[j * DIM + hh * HD + d]; }
            attn[hh * HD + d] = acc;
        }
    }
    for (var d = 0u; d < DIM; d = d + 1u) {
        var acc = 0.0;
        for (var e = 0u; e < DIM; e = e + 1u) { acc = acc + attn[e] * wo[d * DIM + e]; }
        out[i * DIM + d] = acc;
    }
}
"#;const ADD_WGSL: &str = r#"
struct P { n: u32, p0: u32, p1: u32, p2: u32 };
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> p: P;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.n) { return; }
    y[i] = a[i] + b[i];
}
"#;

fn cpu_conv(x: &[f32], w: &[f32], b: &[f32], p: &ConvParams) -> Vec<f32> {
    let total = (p.n * p.cout * p.dout * p.hout * p.wout) as usize;
    let mut y = vec![0.0f32; total];
    for idx in 0..total {
        let mut t = idx;
        let ow = t % p.wout as usize; t /= p.wout as usize;
        let oh = t % p.hout as usize; t /= p.hout as usize;
        let od = t % p.dout as usize; t /= p.dout as usize;
        let co = t % p.cout as usize;
        let bn = t / p.cout as usize;
        let mut acc = b[co];
        for ci in 0..p.cin as usize {
            for kd in 0..p.kd as usize {
                let id = od as i32 * p.stride as i32 + kd as i32 - p.pad as i32;
                if id < 0 || id >= p.din as i32 { continue; }
                for kh in 0..p.kh as usize {
                    let ih = oh as i32 * p.stride as i32 + kh as i32 - p.pad as i32;
                    if ih < 0 || ih >= p.hin as i32 { continue; }
                    for kw in 0..p.kw as usize {
                        let iw = ow as i32 * p.stride as i32 + kw as i32 - p.pad as i32;
                        if iw < 0 || iw >= p.win as i32 { continue; }
                        let xi = (((bn * p.cin as usize + ci) * p.din as usize + id as usize) * p.hin as usize + ih as usize) * p.win as usize + iw as usize;
                        let wi = (((co * p.cin as usize + ci) * p.kd as usize + kd) * p.kh as usize + kh) * p.kw as usize + kw;
                        acc += x[xi] * w[wi];
                    }
                }
            }
        }
        y[idx] = acc;
    }
    y
}
fn cpu_gn(x: &[f32], gamma: &[f32], beta: &[f32], p: &GNParams) -> Vec<f32> {
    let spatial = (p.d * p.h * p.w) as usize;
    let cg = (p.c / p.groups) as usize;
    let mut y = vec![0.0f32; x.len()];
    for bn in 0..p.n as usize {
        for g in 0..p.groups as usize {
            let c0 = g * cg;
            let count = cg * spatial;
            let mut sum = 0.0f32; let mut sumsq = 0.0f32;
            for c in c0..c0 + cg {
                for s in 0..spatial {
                    let v = x[(bn * p.c as usize + c) * spatial + s];
                    sum += v; sumsq += v * v;
                }
            }
            let mean = sum / count as f32;
            let variance = sumsq / count as f32 - mean * mean;
            let inv = (variance + p.eps).sqrt().recip();
            for c in c0..c0 + cg {
                for s in 0..spatial {
                    let idx = (bn * p.c as usize + c) * spatial + s;
                    let norm = (x[idx] - mean) * inv;
                    let v = norm * gamma[c] + beta[c];
                    y[idx] = v / (1.0 + (-v).exp());
                }
            }
        }
    }
    y
}

fn cpu_up(x: &[f32], p: &UpP) -> Vec<f32> {
    let mut y = vec![0.0f32; (p.n * p.c * p.dout * p.hout * p.wout) as usize];
    for idx in 0..y.len() {
        let mut t = idx;
        let ow = t % p.wout as usize; t /= p.wout as usize;
        let oh = t % p.hout as usize; t /= p.hout as usize;
        let od = t % p.dout as usize; t /= p.dout as usize;
        let co = t % p.c as usize;
        let bn = t / p.c as usize;
        y[idx] = x[(((bn * p.c as usize + co) * p.din as usize + od / p.scale as usize) * p.hin as usize + oh / p.scale as usize) * p.win as usize + ow / p.scale as usize];
    }
    y
}
fn cpu_lin_in(x: &[f32], w: &[f32]) -> Vec<f32> {
    let mut y = vec![0.0f32; 8 * 32];
    for t in 0..8 { for d in 0..32 {
        let mut acc = 0.0;
        for c in 0..2 { acc += x[c * 8 + t] * w[d * 2 + c]; }
        y[t * 32 + d] = acc;
    }}
    y
}
fn cpu_lin_out(x: &[f32], w: &[f32]) -> Vec<f32> {
    let mut y = vec![0.0f32; 16];
    for c in 0..2 { for t in 0..8 {
        let mut acc = 0.0;
        for d in 0..32 { acc += x[t * 32 + d] * w[c * 32 + d]; }
        y[c * 8 + t] = acc;
    }}
    y
}
fn cpu_attn(x: &[f32], wq: &[f32], wk: &[f32], wv: &[f32], wo: &[f32], scale: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; 8 * 32];
    for i in 0..8 {
        let q: Vec<f32> = (0..32).map(|d| (0..32).map(|e| x[i * 32 + e] * wq[d * 32 + e]).sum()).collect();
        let mut kk = vec![0.0f32; 8 * 32];
        let mut vv = vec![0.0f32; 8 * 32];
        for j in 0..8 { for d in 0..32 {
            kk[j * 32 + d] = (0..32).map(|e| x[j * 32 + e] * wk[d * 32 + e]).sum();
            vv[j * 32 + d] = (0..32).map(|e| x[j * 32 + e] * wv[d * 32 + e]).sum();
        }}
        let mut attn = vec![0.0f32; 32];
        for hh in 0..4 {
            let mut scores = vec![0.0f32; 8];
            let mut maxs = f32::NEG_INFINITY;
            for j in 0..8 {
                let dot: f32 = (0..8).map(|d| q[hh * 8 + d] * kk[j * 32 + hh * 8 + d]).sum();
                scores[j] = dot * scale; maxs = maxs.max(scores[j]);
            }
            let sum: f32 = scores.iter().map(|s| (s - maxs).exp()).sum();
            for d in 0..8 {
                attn[hh * 8 + d] = (0..8).map(|j| (scores[j] - maxs).exp() / sum * vv[j * 32 + hh * 8 + d]).sum();
            }
        }
        for d in 0..32 {
            out[i * 32 + d] = (0..32).map(|e| attn[e] * wo[d * 32 + e]).sum();
        }
    }
    out
}
fn main() {
    let cp = ConvParams { n: 1, cin: 2, cout: 2, din: 2, hin: 2, win: 2,
                         kd: 3, kh: 3, kw: 3, pad: 1, stride: 1,
                         dout: 2, hout: 2, wout: 2, _p0: 0, _p1: 0 };
    let gp = GNParams { n: 1, c: 2, d: 2, h: 2, w: 2, groups: 1, eps: 1e-5, _p: 0 };
    let ap = AddParams { n: 16, _p0: 0, _p1: 0, _p2: 0 };
    let cp4 = ConvParams { n: 1, cin: 2, cout: 3, din: 2, hin: 2, win: 2,
                          kd: 3, kh: 3, kw: 3, pad: 1, stride: 1,
                          dout: 4, hout: 4, wout: 4, _p0: 0, _p1: 0 };
    let upp = UpP { n: 1, c: 2, din: 2, hin: 2, win: 2, scale: 2, dout: 4, hout: 4, wout: 4, _p0: 0, _p1: 0, _p2: 0 };
    let upp2 = UpP { n: 1, c: 3, din: 4, hin: 4, win: 4, scale: 2, dout: 8, hout: 8, wout: 8, _p0: 0, _p1: 0, _p2: 0 };
    let upp3 = UpP { n: 1, c: 3, din: 8, hin: 8, win: 8, scale: 2, dout: 16, hout: 16, wout: 16, _p0: 0, _p1: 0, _p2: 0 };
    let upp4 = UpP { n: 1, c: 3, din: 16, hin: 16, win: 16, scale: 2, dout: 32, hout: 32, wout: 32, _p0: 0, _p1: 0, _p2: 0 };
    let xn = (cp.n * cp.cin * cp.din * cp.hin * cp.win) as usize;
    let wn = (cp.cout * cp.cin * cp.kd * cp.kh * cp.kw) as usize;
    let yn = (cp.n * cp.cout * cp.dout * cp.hout * cp.wout) as usize;
    let mut x = vec![0.0f32; xn];
    let mut w1 = vec![0.0f32; wn];
    let mut w2 = vec![0.0f32; wn];
    let mut b1 = vec![0.0f32; 2];
    let mut b2 = vec![0.0f32; 2];
    let w4n = (cp4.cout * cp4.cin * cp4.kd * cp4.kh * cp4.kw) as usize;
    let mut w4 = vec![0.0f32; w4n];
    let mut b4 = vec![0.0f32; 3];
    let mut w3 = vec![0.0f32; wn];
    let mut w5 = vec![0.0f32; wn];
    let mut b3 = vec![0.0f32; 2];
    let mut b5 = vec![0.0f32; 2];
    let mut gamma2 = vec![0.0f32; 2];
    let mut beta2 = vec![0.0f32; 2];
    let mut w_in = vec![0.0f32; 32 * 2];
    let mut w_out = vec![0.0f32; 2 * 32];
    let mut wq = vec![0.0f32; 32 * 32];
    let mut wk = vec![0.0f32; 32 * 32];
    let mut wv = vec![0.0f32; 32 * 32];
    let mut wo = vec![0.0f32; 32 * 32];
    let mut gamma = vec![0.0f32; 2];
    let mut beta = vec![0.0f32; 2];
    for i in 0..xn { x[i] = ((i * 13 % 71) as f32 - 35.0) / 19.0; }
    for i in 0..wn {
        w1[i] = ((i * 17 % 53) as f32 - 26.0) / 31.0;
        w2[i] = ((i * 19 % 47) as f32 - 23.0) / 29.0;
    }
    for i in 0..wn {
        w3[i] = ((i * 29 % 61) as f32 - 30.0) / 33.0;
        w5[i] = ((i * 31 % 67) as f32 - 33.0) / 35.0;
    }
    for i in 0..2 {
        b3[i] = ((i * 3 % 7) as f32 - 3.0) * 0.01;
        b5[i] = ((i * 7 % 11) as f32 - 5.0) * 0.01;
        gamma2[i] = 1.0 + ((i * 5 % 9) as f32 - 4.0) * 0.02;
        beta2[i] = ((i * 11 % 13) as f32 - 6.0) * 0.01;
    }
    let fill2 = |dst: &mut [f32], k: usize| { for i in 0..dst.len() { dst[i] = ((i * 37 % 101) as f32 - 50.0) / (k as f32 * 8.0); } };
    fill2(&mut w_in, 2); fill2(&mut w_out, 32); fill2(&mut wq, 32); fill2(&mut wk, 32); fill2(&mut wv, 32); fill2(&mut wo, 32);
    for i in 0..w4n { w4[i] = ((i * 23 % 59) as f32 - 29.0) / 37.0; }
    for i in 0..3 { b4[i] = ((i * 5 % 13) as f32 - 6.0) * 0.01; }
    for i in 0..2 {
        b1[i] = ((i * 7 % 11) as f32 - 5.0) * 0.01;
        b2[i] = ((i * 5 % 9) as f32 - 4.0) * 0.01;
        gamma[i] = 1.0 + ((i * 3 % 7) as f32 - 3.0) * 0.02;
        beta[i] = ((i * 11 % 13) as f32 - 6.0) * 0.01;
    }
    let c1 = cpu_conv(&x, &w1, &b1, &cp);
    let g1 = cpu_gn(&c1, &gamma, &beta, &gp);
    let c2 = cpu_conv(&g1, &w2, &b2, &cp);
    let res: Vec<f32> = x.iter().zip(c2.iter()).map(|(a, b)| a + b).collect();
    let c4 = cpu_conv(&res, &w3, &b3, &cp);
    let g4 = cpu_gn(&c4, &gamma2, &beta2, &gp);
    let c5 = cpu_conv(&g4, &w5, &b5, &cp);
    let res2: Vec<f32> = res.iter().zip(c5.iter()).map(|(a, b)| a + b).collect();
    let proj = cpu_lin_in(&res2, &w_in);
    let att = cpu_attn(&proj, &wq, &wk, &wv, &wo, 1.0 / (8.0f32).sqrt());
    let back = cpu_lin_out(&att, &w_out);
    let res3: Vec<f32> = res2.iter().zip(back.iter()).map(|(a, b)| a + b).collect();
    let up_cpu = cpu_up(&res3, &upp);
    let dec = cpu_conv(&up_cpu, &w4, &b4, &cp4);
    let dec2 = cpu_up(&dec, &upp2);
    let dec3 = cpu_up(&dec2, &upp3);
    let expect = cpu_up(&dec3, &upp4);

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
        label: Some("vae-resblock"), required_features: wgpu::Features::empty(),
        required_limits: limits, ..Default::default()
    })).expect("device");
    let mk = |data: &[f32]| device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None, contents: bytemuck::cast_slice(data), usage: wgpu::BufferUsages::STORAGE,
    });
    let x_buf = mk(&x); let w1_buf = mk(&w1); let b1_buf = mk(&b1); let w2_buf = mk(&w2); let b2_buf = mk(&b2);
    let gamma_buf = mk(&gamma); let beta_buf = mk(&beta); let w4_buf = mk(&w4); let b4_buf = mk(&b4);
    let w3_buf = mk(&w3); let b3_buf = mk(&b3); let w5_buf = mk(&w5); let b5_buf = mk(&b5);
    let w_in_buf = mk(&w_in); let w_out_buf = mk(&w_out);
    let wq_buf = mk(&wq); let wk_buf = mk(&wk); let wv_buf = mk(&wv); let wo_buf = mk(&wo);
    let gamma2_buf = mk(&gamma2); let beta2_buf = mk(&beta2);
    let y1 = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (yn * 4) as u64, usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false });
    let y2 = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (yn * 4) as u64, usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false });
    let y3 = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (yn * 4) as u64, usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false });
    let y4 = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (yn * 4) as u64, usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false });
    let y5 = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (yn * 4) as u64, usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false });
    let y6 = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (yn * 4) as u64, usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false });
    let res2 = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (yn * 4) as u64, usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false });
    let proj_buf = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (8 * 32 * 4) as u64, usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false });
    let att_buf = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (8 * 32 * 4) as u64, usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false });
    let back_buf = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (16 * 4) as u64, usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false });
    let res3 = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (yn * 4) as u64, usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false });
    let out_buf = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (yn * 4) as u64, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
    let un = (cp4.n * cp4.cout * cp4.dout * cp4.hout * cp4.wout) as usize;
    let upn = (cp.cout * 4 * cp.hout * cp.wout) as usize;
    let un2 = (upp2.n * upp2.c * upp2.dout * upp2.hout * upp2.wout) as usize;
    let un3 = (upp3.n * upp3.c * upp3.dout * upp3.hout * upp3.wout) as usize;
    let un4 = (upp4.n * upp4.c * upp4.dout * upp4.hout * upp4.wout) as usize;
    let up_buf = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (upn * 4) as u64, usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false });
    let final_buf = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (un * 4) as u64, usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false });
    let up2_buf = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (un2 * 4) as u64, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
    let readback = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (un4 * 4) as u64, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
    let cp_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::bytes_of(&cp), usage: wgpu::BufferUsages::UNIFORM });
    let gp_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::bytes_of(&gp), usage: wgpu::BufferUsages::UNIFORM });
    let ap_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::bytes_of(&ap), usage: wgpu::BufferUsages::UNIFORM });
    let cp4_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::bytes_of(&cp4), usage: wgpu::BufferUsages::UNIFORM });
    let up_p_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::bytes_of(&upp), usage: wgpu::BufferUsages::UNIFORM });
    let up2_p_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::bytes_of(&upp2), usage: wgpu::BufferUsages::UNIFORM });
    let up3_p_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::bytes_of(&upp3), usage: wgpu::BufferUsages::UNIFORM });
    let up4_p_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::bytes_of(&upp4), usage: wgpu::BufferUsages::UNIFORM });
    let up4_buf = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (un4 * 4) as u64, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
    let up3_buf = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (un3 * 4) as u64, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
    let ap2 = AttnP { scale: 1.0 / (8.0f32).sqrt(), _p0: 0, _p1: 0, _p2: 0 };
    let attn_p_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::bytes_of(&ap2), usage: wgpu::BufferUsages::UNIFORM });

    let conv_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: None, source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(CONV_WGSL)) });
    let gn_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: None, source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(GN_WGSL)) });
    let add_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: None, source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(ADD_WGSL)) });
    let conv_pipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: None, layout: None, module: &conv_mod, entry_point: Some("main"), compilation_options: Default::default(), cache: None });
    let gn_pipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: None, layout: None, module: &gn_mod, entry_point: Some("main"), compilation_options: Default::default(), cache: None });
    let add_pipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: None, layout: None, module: &add_mod, entry_point: Some("main"), compilation_options: Default::default(), cache: None });
    let up_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: None, source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(UP_WGSL)) });
    let up_pipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: None, layout: None, module: &up_mod, entry_point: Some("main"), compilation_options: Default::default(), cache: None });
    let li_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: None, source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(LIN_IN_WGSL)) });
    let lo_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: None, source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(LIN_OUT_WGSL)) });
    let at_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: None, source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(ATTN_WGSL)) });
    let li_pipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: None, layout: None, module: &li_mod, entry_point: Some("main"), compilation_options: Default::default(), cache: None });
    let lo_pipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: None, layout: None, module: &lo_mod, entry_point: Some("main"), compilation_options: Default::default(), cache: None });
    let at_pipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: None, layout: None, module: &at_mod, entry_point: Some("main"), compilation_options: Default::default(), cache: None });

    let bind = |pipe: &wgpu::ComputePipeline, entries: &[wgpu::BindGroupEntry]| device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &pipe.get_bind_group_layout(0), entries });
    let c1_bg = bind(&conv_pipe, &[
        wgpu::BindGroupEntry { binding: 0, resource: x_buf.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 1, resource: w1_buf.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 2, resource: b1_buf.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 3, resource: y1.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 4, resource: cp_buf.as_entire_binding() },
    ]);
    let g_bg = bind(&gn_pipe, &[
        wgpu::BindGroupEntry { binding: 0, resource: y1.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 1, resource: gamma_buf.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 2, resource: beta_buf.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 3, resource: y2.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 4, resource: gp_buf.as_entire_binding() },
    ]);
    let c2_bg = bind(&conv_pipe, &[
        wgpu::BindGroupEntry { binding: 0, resource: y2.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 1, resource: w2_buf.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 2, resource: b2_buf.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 3, resource: y3.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 4, resource: cp_buf.as_entire_binding() },
    ]);
    let a_bg = bind(&add_pipe, &[
        wgpu::BindGroupEntry { binding: 0, resource: x_buf.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 1, resource: y3.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 2, resource: out_buf.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 3, resource: ap_buf.as_entire_binding() },
    ]);
    let up_bg = bind(&up_pipe, &[
        wgpu::BindGroupEntry { binding: 0, resource: res3.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 1, resource: up_buf.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 2, resource: up_p_buf.as_entire_binding() },
    ]);
    let cf_bg = bind(&conv_pipe, &[
        wgpu::BindGroupEntry { binding: 0, resource: up_buf.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 1, resource: w4_buf.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 2, resource: b4_buf.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 3, resource: final_buf.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 4, resource: cp4_buf.as_entire_binding() },
    ]);
    let c3_bg = bind(&conv_pipe, &[
        wgpu::BindGroupEntry { binding: 0, resource: out_buf.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 1, resource: w3_buf.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 2, resource: b3_buf.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 3, resource: y4.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 4, resource: cp_buf.as_entire_binding() },
    ]);
    let g2_bg = bind(&gn_pipe, &[
        wgpu::BindGroupEntry { binding: 0, resource: y4.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 1, resource: gamma2_buf.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 2, resource: beta2_buf.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 3, resource: y5.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 4, resource: gp_buf.as_entire_binding() },
    ]);
    let c4_bg = bind(&conv_pipe, &[
        wgpu::BindGroupEntry { binding: 0, resource: y5.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 1, resource: w5_buf.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 2, resource: b5_buf.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 3, resource: y6.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 4, resource: cp_buf.as_entire_binding() },
    ]);
    let a2_bg = bind(&add_pipe, &[
        wgpu::BindGroupEntry { binding: 0, resource: out_buf.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 1, resource: y6.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 2, resource: res2.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 3, resource: ap_buf.as_entire_binding() },
    ]);
    let up2_bg = bind(&up_pipe, &[
        wgpu::BindGroupEntry { binding: 0, resource: final_buf.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 1, resource: up2_buf.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 2, resource: up2_p_buf.as_entire_binding() },
    ]);
    let li_bg = bind(&li_pipe, &[
        wgpu::BindGroupEntry { binding: 0, resource: res2.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 1, resource: w_in_buf.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 2, resource: proj_buf.as_entire_binding() },
    ]);
    let at_bg = bind(&at_pipe, &[
        wgpu::BindGroupEntry { binding: 0, resource: proj_buf.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 1, resource: wq_buf.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 2, resource: wk_buf.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 3, resource: wv_buf.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 4, resource: wo_buf.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 5, resource: att_buf.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 6, resource: attn_p_buf.as_entire_binding() },
    ]);
    let lo_bg = bind(&lo_pipe, &[
        wgpu::BindGroupEntry { binding: 0, resource: att_buf.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 1, resource: w_out_buf.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 2, resource: back_buf.as_entire_binding() },
    ]);
    let a3_bg = bind(&add_pipe, &[
        wgpu::BindGroupEntry { binding: 0, resource: res2.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 1, resource: back_buf.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 2, resource: res3.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 3, resource: ap_buf.as_entire_binding() },
    ]);
    let up3_bg = bind(&up_pipe, &[
        wgpu::BindGroupEntry { binding: 0, resource: up2_buf.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 1, resource: up3_buf.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 2, resource: up3_p_buf.as_entire_binding() },
    ]);
    let up4_bg = bind(&up_pipe, &[
        wgpu::BindGroupEntry { binding: 0, resource: up3_buf.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 1, resource: up4_buf.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 2, resource: up4_p_buf.as_entire_binding() },
    ]);
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    let mut pass = |pipe: &wgpu::ComputePipeline, bg: &wgpu::BindGroup, groups: u32| {
        let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
        p.set_pipeline(pipe); p.set_bind_group(0, bg, &[]); p.dispatch_workgroups(groups, 1, 1);
    };
    pass(&conv_pipe, &c1_bg, ((yn + 63) / 64) as u32);
    pass(&gn_pipe, &g_bg, 1);
    pass(&conv_pipe, &c2_bg, ((yn + 63) / 64) as u32);
    pass(&add_pipe, &a_bg, ((yn + 63) / 64) as u32);
    pass(&conv_pipe, &c3_bg, ((yn + 63) / 64) as u32);
    pass(&gn_pipe, &g2_bg, 1);
    pass(&conv_pipe, &c4_bg, ((yn + 63) / 64) as u32);
    pass(&add_pipe, &a2_bg, ((yn + 63) / 64) as u32);
    pass(&li_pipe, &li_bg, 256);
    pass(&at_pipe, &at_bg, 8);
    pass(&lo_pipe, &lo_bg, 16);
    pass(&add_pipe, &a3_bg, 16);
    pass(&up_pipe, &up_bg, ((upn + 63) / 64) as u32);
    pass(&conv_pipe, &cf_bg, ((un + 63) / 64) as u32);
    pass(&up_pipe, &up2_bg, ((un2 + 63) / 64) as u32);
    pass(&up_pipe, &up3_bg, ((un3 + 63) / 64) as u32);
    pass(&up_pipe, &up4_bg, ((un4 + 63) / 64) as u32);
    enc.copy_buffer_to_buffer(&up4_buf, 0, &readback, 0, (un4 * 4) as u64);
    queue.submit(Some(enc.finish()));
    let slice = readback.slice(..);
    slice.map_async(wgpu::MapMode::Read, |r| r.expect("map"));
    device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
    let data = slice.get_mapped_range();
    let got: &[f32] = bytemuck::cast_slice(&data);
    let mut max_err = 0.0f32;
    for (a, b) in got.iter().zip(expect.iter()) { max_err = max_err.max((a - b).abs()); }
    println!("vae-decoder 2xresblock 4xupsample out={} max_abs_err={max_err:.3e}", un4);
    assert!(max_err < 1e-5, "vae resblock mismatch");
    println!("VULKAN VAE DECODER OK");
}





















