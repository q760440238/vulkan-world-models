use bytemuck::{Pod, Zeroable};
use std::borrow::Cow;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Params { scale: f32, eps: f32, _p0: u32, _p1: u32 }
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct TimeParams { t: f32, p0: u32, p1: u32, p2: u32 }

const D: usize = 512;
const L: usize = 4;
const F: usize = 512;
const HEADS: usize = 4;
const HD: usize = 128;
const LK: usize = 8;

const WGSL: &str = r#"
const D: u32 = 512u;
const L: u32 = 4u;
const F: u32 = 512u;
const HEADS: u32 = 4u;
const HD: u32 = 128u;
const LK: u32 = 8u;
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
@group(0) @binding(11) var<storage, read> ctx_k: array<f32>; // [LK,D]
@group(0) @binding(12) var<storage, read> ctx_v: array<f32>; // [LK,D]
@group(0) @binding(13) var<storage, read> wqc: array<f32>;   // [D,D]
@group(0) @binding(14) var<storage, read> woc: array<f32>;   // [D,D]
@group(0) @binding(15) var<storage, read> cam: array<f32>; // [2,D]
@group(0) @binding(16) var<uniform> p: Params;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= L) { return; }

    var h: array<f32, 512>;
    var q: array<f32, 512>;
    var kk: array<f32, 2048>;
    var vv: array<f32, 2048>;
    var scores: array<f32, 16>;
    var attn: array<f32, 512>;
    var x1: array<f32, 512>;
    var h3: array<f32, 512>;
    var qc: array<f32, 512>;
    var cscores: array<f32, 512>;
    var attn_c: array<f32, 512>;
    var x2: array<f32, 512>;
    var h2: array<f32, 512>;
    var ffn: array<f32, 512>;

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

    // RoPE on q
    for (var hh = 0u; hh < HEADS; hh = hh + 1u) {
        for (var d = 0u; d < HD / 2u; d = d + 1u) {
            let freq = pow(10000.0, -2.0 * f32(d) / f32(HD));
            let ang = f32(i) * freq;
            let c = cos(ang); let s = sin(ang);
            let a = hh * HD + 2u * d;
            let q0 = q[a]; let q1 = q[a + 1u];
            q[a] = q0 * c - q1 * s;
            q[a + 1u] = q0 * s + q1 * c;
        }
    }

    // K/V for causal keys (all heads packed in D)
    for (var j = 0u; j < L; j = j + 1u) {
        if (j > i) { break; }
        var hj: array<f32, 512>;
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
        // RoPE on k at position j
        for (var hh = 0u; hh < HEADS; hh = hh + 1u) {
            for (var d = 0u; d < HD / 2u; d = d + 1u) {
                let freq = pow(10000.0, -2.0 * f32(d) / f32(HD));
                let ang = f32(j) * freq;
                let c = cos(ang); let s = sin(ang);
                let a = j * D + hh * HD + 2u * d;
                let k0 = kk[a]; let k1 = kk[a + 1u];
                kk[a] = k0 * c - k1 * s;
                kk[a + 1u] = k0 * s + k1 * c;
            }
        }
    }

    // multi-head causal attention: HEADS x HD = D
    for (var hh = 0u; hh < HEADS; hh = hh + 1u) {
        var maxs = -1e30;
        for (var j = 0u; j <= i; j = j + 1u) {
            var dot = 0.0;
            for (var d = 0u; d < HD; d = d + 1u) {
                dot = dot + q[hh * HD + d] * kk[j * D + hh * HD + d];
            }
            scores[hh * L + j] = dot * p.scale;
            maxs = max(maxs, scores[hh * L + j]);
        }
        var sum = 0.0;
        for (var j = 0u; j <= i; j = j + 1u) { sum = sum + exp(scores[hh * L + j] - maxs); }
        for (var d = 0u; d < HD; d = d + 1u) {
            var acc = 0.0;
            for (var j = 0u; j <= i; j = j + 1u) {
                acc = acc + (exp(scores[hh * L + j] - maxs) / sum) * vv[j * D + hh * HD + d];
            }
            attn[hh * HD + d] = acc;
        }
    }

    // out projection + residual
    for (var d = 0u; d < D; d = d + 1u) {
        var acc = 0.0;
        for (var e = 0u; e < D; e = e + 1u) { acc = acc + attn[e] * wo[d * D + e]; }
        x1[d] = x[i * D + d] + acc * e[2u * D + d];
    }

    for (var d = 0u; d < D; d = d + 1u) { x1[d] = x1[d] * (1.0 + cam[d]) + cam[D + d]; }

    // cross-attention: h3 = RMSNorm(x1) * g2
    var s3 = 0.0;
    for (var d = 0u; d < D; d = d + 1u) { s3 = s3 + x1[d] * x1[d]; }
    let i3 = inverseSqrt(s3 / f32(D) + p.eps);
    for (var d = 0u; d < D; d = d + 1u) { h3[d] = x1[d] * i3 * g2[d]; }
    for (var d = 0u; d < D; d = d + 1u) {
        var acc = 0.0;
        for (var e = 0u; e < D; e = e + 1u) { acc = acc + h3[e] * wqc[d * D + e]; }
        qc[d] = acc;
    }
    for (var hh = 0u; hh < HEADS; hh = hh + 1u) {
        var cmax = -1e30;
        for (var j = 0u; j < LK; j = j + 1u) {
            var dot = 0.0;
            for (var d = 0u; d < HD; d = d + 1u) {
                dot = dot + qc[hh * HD + d] * ctx_k[j * D + hh * HD + d];
            }
            cscores[hh * LK + j] = dot * p.scale;
            cmax = max(cmax, cscores[hh * LK + j]);
        }
        var csum = 0.0;
        for (var j = 0u; j < LK; j = j + 1u) { csum = csum + exp(cscores[hh * LK + j] - cmax); }
        for (var d = 0u; d < HD; d = d + 1u) {
            var acc = 0.0;
            for (var j = 0u; j < LK; j = j + 1u) {
                acc = acc + (exp(cscores[hh * LK + j] - cmax) / csum) * ctx_v[j * D + hh * HD + d];
            }
            attn_c[hh * HD + d] = acc;
        }
    }
    for (var d = 0u; d < D; d = d + 1u) {
        var acc = 0.0;
        for (var e = 0u; e < D; e = e + 1u) { acc = acc + attn_c[e] * woc[d * D + e]; }
        x2[d] = x1[d] + acc;
    }

    // h2 = RMSNorm(x2) * g2
    var s2 = 0.0;
    for (var d = 0u; d < D; d = d + 1u) { s2 = s2 + x2[d] * x2[d]; }
    let i2 = inverseSqrt(s2 / f32(D) + p.eps);
    for (var d = 0u; d < D; d = d + 1u) { h2[d] = (x2[d] * i2 * g2[d]) * (1.0 + e[4u * D + d]) + e[3u * D + d]; }

    // FFN: W1 -> SiLU -> W2 + residual
    for (var f = 0u; f < F; f = f + 1u) {
        var acc = 0.0;
        for (var d = 0u; d < D; d = d + 1u) { acc = acc + h2[d] * w1[f * D + d]; }
        ffn[f] = acc / (1.0 + exp(-acc));
    }
    for (var d = 0u; d < D; d = d + 1u) {
        var acc = 0.0;
        for (var f = 0u; f < F; f = f + 1u) { acc = acc + ffn[f] * w2[d * F + f]; }
        out[i * D + d] = x2[d] + acc * e[5u * D + d];
    }
}
"#;

const TIME_WGSL: &str = r#"
struct TimeParams { t: f32, p0: u32, p1: u32, p2: u32 };
@group(0) @binding(0) var<storage, read> w1t: array<f32>; // [64,32]
@group(0) @binding(1) var<storage, read> b1t: array<f32>; // [64]
@group(0) @binding(2) var<storage, read> w2t: array<f32>; // [192,64]
@group(0) @binding(3) var<storage, read> b2t: array<f32>; // [192]
@group(0) @binding(4) var<storage, read_write> eout: array<f32>; // [192]
@group(0) @binding(5) var<uniform> tp: TimeParams;

const D: u32 = 512u;
const H: u32 = 64u;
const OUT: u32 = 3072u;

@compute @workgroup_size(1)
fn main() {
    var emb: array<f32, 512>;
    for (var i = 0u; i < 256u; i = i + 1u) {
        let freq = exp(-log(10000.0) * f32(i) / 256.0);
        emb[i] = sin(tp.t * freq);
        emb[256u + i] = cos(tp.t * freq);
    }
    var h: array<f32, 256>;
    for (var j = 0u; j < H; j = j + 1u) {
        var acc = b1t[j];
        for (var d = 0u; d < D; d = d + 1u) { acc = acc + w1t[j * D + d] * emb[d]; }
        h[j] = acc / (1.0 + exp(-acc));
    }
    for (var o = 0u; o < OUT; o = o + 1u) {
        var acc = b2t[o];
        for (var j = 0u; j < H; j = j + 1u) { acc = acc + w2t[o * H + j] * h[j]; }
        eout[o] = acc;
    }
}
"#;

const CAM_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> pose: array<f32>;   // [16]
@group(0) @binding(1) var<storage, read> wc1: array<f32>;    // [32,16]
@group(0) @binding(2) var<storage, read> bc1: array<f32>;    // [32]
@group(0) @binding(3) var<storage, read> wc2: array<f32>;    // [1024,32]
@group(0) @binding(4) var<storage, read> bc2: array<f32>;    // [1024]
@group(0) @binding(5) var<storage, read_write> cam: array<f32>; // [1024]

const C: u32 = 16u;
const HC: u32 = 32u;
const COUT: u32 = 1024u;

@compute @workgroup_size(1)
fn main() {
    var h: array<f32, 32>;
    for (var j = 0u; j < HC; j = j + 1u) {
        var acc = bc1[j];
        for (var i = 0u; i < C; i = i + 1u) { acc = acc + wc1[j * C + i] * pose[i]; }
        h[j] = acc / (1.0 + exp(-acc));
    }
    for (var o = 0u; o < COUT; o = o + 1u) {
        var acc = bc2[o];
        for (var j = 0u; j < HC; j = j + 1u) { acc = acc + wc2[o * HC + j] * h[j]; }
        cam[o] = acc;
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
fn cpu_time(t: f32, w1t: &[f32], b1t: &[f32], w2t: &[f32], b2t: &[f32]) -> Vec<f32> {
    let mut emb = vec![0.0f32; D];
    for i in 0..D / 2 {
        let freq = (-(10000.0f32).ln() * i as f32 / (D as f32 / 2.0)).exp();
        emb[i] = (t * freq).sin();
        emb[D / 2 + i] = (t * freq).cos();
    }
    let mut h = vec![0.0f32; 64];
    for j in 0..64 {
        let mut acc = b1t[j];
        for d in 0..D { acc += w1t[j * D + d] * emb[d]; }
        h[j] = acc / (1.0 + (-acc).exp());
    }
    let mut e = vec![0.0f32; 6 * D];
    for o in 0..6 * D {
        let mut acc = b2t[o];
        for j in 0..64 { acc += w2t[o * 64 + j] * h[j]; }
        e[o] = acc;
    }
    e
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
fn rope(v: &mut [f32], pos: usize) {
    for hh in 0..HEADS {
        for d in 0..HD / 2 {
            let freq = 10000.0f32.powf(-2.0 * d as f32 / HD as f32);
            let ang = pos as f32 * freq;
            let c = ang.cos(); let s = ang.sin();
            let a = hh * HD + 2 * d;
            let a0 = v[a]; let a1 = v[a + 1];
            v[a] = a0 * c - a1 * s;
            v[a + 1] = a0 * s + a1 * c;
        }
    }
}
fn cpu_camera(pose: &[f32], wc1: &[f32], bc1: &[f32], wc2: &[f32], bc2: &[f32]) -> Vec<f32> {
    let mut h = vec![0.0f32; 32];
    for j in 0..32 {
        let mut acc = bc1[j];
        for i in 0..16 { acc += wc1[j * 16 + i] * pose[i]; }
        h[j] = acc / (1.0 + (-acc).exp());
    }
    let mut cam = vec![0.0f32; 2 * D];
    for o in 0..2 * D {
        let mut acc = bc2[o];
        for j in 0..32 { acc += wc2[o * 32 + j] * h[j]; }
        cam[o] = acc;
    }
    cam
}
fn cpu_block(x: &[f32], g1: &[f32], g2: &[f32], wq: &[f32], wk: &[f32], wv: &[f32], wo: &[f32],
             w1: &[f32], w2: &[f32], e: &[f32], ctx_k: &[f32], ctx_v: &[f32], wqc: &[f32], woc: &[f32], cam: &[f32], eps: f32, scale: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; L * D];
    for i in 0..L {
        let h0 = rms(&x[i * D..(i + 1) * D], g1, eps);
        let mut h = vec![0.0f32; D];
        for d in 0..D { h[d] = h0[d] * (1.0 + e[D + d]) + e[d]; }
        let mut q = matvec(wq, D, D, &h);
        rope(&mut q, i);
        let mut kk = vec![0.0f32; L * D];
        let mut vv = vec![0.0f32; L * D];
        let mut scores = vec![0.0f32; HEADS * L];
        for j in 0..=i {
            let hj0 = rms(&x[j * D..(j + 1) * D], g1, eps);
            let mut hj = vec![0.0f32; D];
            for d in 0..D { hj[d] = hj0[d] * (1.0 + e[D + d]) + e[d]; }
            let mut k = matvec(wk, D, D, &hj);
            rope(&mut k, j);
            let v = matvec(wv, D, D, &hj);
            kk[j * D..(j + 1) * D].copy_from_slice(&k);
            vv[j * D..(j + 1) * D].copy_from_slice(&v);
        }
        let mut attn = vec![0.0f32; D];
        for hh in 0..HEADS {
            let mut maxs = f32::NEG_INFINITY;
            for j in 0..=i {
                let dot: f32 = (0..HD).map(|d| q[hh * HD + d] * kk[j * D + hh * HD + d]).sum();
                scores[hh * L + j] = dot * scale;
                maxs = maxs.max(scores[hh * L + j]);
            }
            let sum: f32 = scores[hh * L..hh * L + i + 1].iter().map(|s| (s - maxs).exp()).sum();
            for d in 0..HD {
                for j in 0..=i {
                    attn[hh * HD + d] += ((scores[hh * L + j] - maxs).exp() / sum) * vv[j * D + hh * HD + d];
                }
            }
        }
        let o = matvec(wo, D, D, &attn);
        let mut x1 = vec![0.0f32; D];
        for d in 0..D { x1[d] = x[i * D + d] + o[d] * e[2 * D + d]; }
        for d in 0..D { x1[d] = x1[d] * (1.0 + cam[d]) + cam[D + d]; }
        let h3 = rms(&x1, g2, eps);
        let qc = matvec(wqc, D, D, &h3);
        let mut cscores = vec![0.0f32; HEADS * LK];
        let mut attn_c = vec![0.0f32; D];
        for hh in 0..HEADS {
            let mut cmax = f32::NEG_INFINITY;
            for j in 0..LK {
                let dot: f32 = (0..HD).map(|d| qc[hh * HD + d] * ctx_k[j * D + hh * HD + d]).sum();
                cscores[hh * LK + j] = dot * scale;
                cmax = cmax.max(cscores[hh * LK + j]);
            }
            let csum: f32 = cscores[hh * LK..hh * LK + LK].iter().map(|s| (s - cmax).exp()).sum();
            for d in 0..HD {
                for j in 0..LK {
                    attn_c[hh * HD + d] += ((cscores[hh * LK + j] - cmax).exp() / csum) * ctx_v[j * D + hh * HD + d];
                }
            }
        }
        let oc = matvec(woc, D, D, &attn_c);
        let mut x2 = vec![0.0f32; D];
        for d in 0..D { x2[d] = x1[d] + oc[d]; }
        let h20 = rms(&x2, g2, eps);
        let mut h2 = vec![0.0f32; D];
        for d in 0..D { h2[d] = h20[d] * (1.0 + e[4 * D + d]) + e[3 * D + d]; }
        let mut ffn = matvec(w1, F, D, &h2);
        for f in 0..F { ffn[f] = ffn[f] / (1.0 + (-ffn[f]).exp()); }
        let o2 = matvec(w2, D, F, &ffn);
        for d in 0..D { out[i * D + d] = x2[d] + o2[d] * e[5 * D + d]; }
    }
    out
}

fn main() {
    let eps = 1e-6f32;
    let scale = 1.0 / (HD as f32).sqrt();
    let mut x = vec![0.0f32; L * D];
    let mut g1 = vec![0.0f32; D];
    let mut g2 = vec![0.0f32; D];
    let mut wq = vec![0.0f32; D * D];
    let mut wk = vec![0.0f32; D * D];
    let mut wv = vec![0.0f32; D * D];
    let mut wo = vec![0.0f32; D * D];
    let mut w1 = vec![0.0f32; F * D];
    let mut w2 = vec![0.0f32; D * F];
    let mut w1t = vec![0.0f32; 64 * D];
    let mut b1t = vec![0.0f32; 64];
    let mut w2t = vec![0.0f32; 6 * D * 64];
    let mut b2t = vec![0.0f32; 6 * D];
    let t_time = 0.7f32;
    let mut pose = vec![0.0f32; 16];
    let mut wc1 = vec![0.0f32; 32 * 16];
    let mut bc1 = vec![0.0f32; 32];
    let mut wc2 = vec![0.0f32; 2 * D * 32];
    let mut bc2 = vec![0.0f32; 2 * D];
    let mut ctx_k = vec![0.0f32; LK * D];
    let mut ctx_v = vec![0.0f32; LK * D];
    let mut wqc = vec![0.0f32; D * D];
    let mut woc = vec![0.0f32; D * D];
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
    fill(&mut ctx_k, D); fill(&mut ctx_v, D); fill(&mut wqc, D); fill(&mut woc, D);
    fill(&mut w1t, D); fill(&mut w2t, 64);

    for i in 0..b1t.len() { b1t[i] = ((i * 7 % 13) as f32 - 6.0) * 0.01; }
    for i in 0..b2t.len() { b2t[i] = ((i * 5 % 11) as f32 - 5.0) * 0.01; }
    for i in 0..pose.len() { pose[i] = ((i * 11 % 17) as f32 - 8.0) * 0.05; }
    fill(&mut wc1, 4); fill(&mut wc2, 4);
    for i in 0..bc1.len() { bc1[i] = ((i * 3 % 7) as f32 - 3.0) * 0.01; }
    for i in 0..bc2.len() { bc2[i] = ((i * 5 % 9) as f32 - 4.0) * 0.01; }
    let e = cpu_time(t_time, &w1t, &b1t, &w2t, &b2t);
    let cam = cpu_camera(&pose, &wc1, &bc1, &wc2, &bc2);
    let expect = cpu_block(&x, &g1, &g2, &wq, &wk, &wv, &wo, &w1, &w2, &e, &ctx_k, &ctx_v, &wqc, &woc, &cam, eps, scale);

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
    let w1_buf = mk(&w1); let w2_buf = mk(&w2);
    let e_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("e"), size: (6 * D * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false,
    });
    let tw1_buf = mk(&w1t); let tb1_buf = mk(&b1t); let tw2_buf = mk(&w2t); let tb2_buf = mk(&b2t);
    let cam_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cam"), size: (2 * D * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false,
    });
    let pose_buf = mk(&pose); let wc1_buf = mk(&wc1); let bc1_buf = mk(&bc1);
    let wc2_buf = mk(&wc2); let bc2_buf = mk(&bc2);
    let ck_buf = mk(&ctx_k); let cv_buf = mk(&ctx_v); let wqc_buf = mk(&wqc); let woc_buf = mk(&woc);
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
    let time_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("time-mlp"), source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(TIME_WGSL)),
    });
    let time_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("time-mlp"), layout: None, module: &time_module, entry_point: Some("main"),
        compilation_options: Default::default(), cache: None,
    });
    let tparams = TimeParams { t: t_time, p0: 0, p1: 0, p2: 0 };
    let tp_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None, contents: bytemuck::bytes_of(&tparams), usage: wgpu::BufferUsages::UNIFORM,
    });
    let cam_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("camera-mlp"), source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(CAM_WGSL)),
    });
    let cam_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("camera-mlp"), layout: None, module: &cam_module, entry_point: Some("main"),
        compilation_options: Default::default(), cache: None,
    });
    let cam_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None, layout: &cam_pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: pose_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: wc1_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: bc1_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: wc2_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: bc2_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: cam_buf.as_entire_binding() },
        ],
    });
    let time_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None, layout: &time_pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: tw1_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: tb1_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: tw2_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: tb2_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: e_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: tp_buf.as_entire_binding() },
        ],
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
            wgpu::BindGroupEntry { binding: 11, resource: ck_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 12, resource: cv_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 13, resource: wqc_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 14, resource: woc_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 15, resource: cam_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 16, resource: p_buf.as_entire_binding() },
        ],
    });
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
        pass.set_pipeline(&time_pipeline); pass.set_bind_group(0, &time_bg, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
        pass.set_pipeline(&cam_pipeline); pass.set_bind_group(0, &cam_bg, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
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





























