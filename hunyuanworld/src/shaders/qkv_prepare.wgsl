struct Params {
    l: u32,
    c: u32,
    heads: u32,
    hd: u32,
    has_norm: u32,
    has_rope: u32,
    eps: f32,
    pad: u32,
};

@group(0) @binding(0) var<storage, read> qkv: array<f32>;
@group(0) @binding(1) var<storage, read> qw: array<f32>;
@group(0) @binding(2) var<storage, read> qb: array<f32>;
@group(0) @binding(3) var<storage, read> kw: array<f32>;
@group(0) @binding(4) var<storage, read> kb: array<f32>;
@group(0) @binding(5) var<storage, read> pos: array<u32>;
@group(0) @binding(6) var<storage, read> cos_tab: array<f32>;
@group(0) @binding(7) var<storage, read> sin_tab: array<f32>;
@group(0) @binding(8) var<storage, read_write> qout: array<f32>;
@group(0) @binding(9) var<storage, read_write> kout: array<f32>;
@group(0) @binding(10) var<storage, read_write> vout: array<f32>;
@group(0) @binding(11) var<uniform> p: Params;

var<workgroup> red: array<f32, 64>;
var<workgroup> qn: array<f32, 128>;
var<workgroup> kn: array<f32, 128>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_index) tid: u32) {
    let token = wid.x;
    let h = wid.y;
    if (token >= p.l || h >= p.heads) { return; }
    let base = token * 3u * p.c + h * p.hd;
    let qbase = base;
    let kbase = base + p.c;
    let vbase = base + 2u * p.c;

    if (p.has_norm != 0u) {
        var s = 0.0;
        var d = tid;
        loop {
            if (d >= p.hd) { break; }
            s = s + qkv[qbase + d];
            d = d + 64u;
        }
        red[tid] = s;
        workgroupBarrier();
        var stride = 32u;
        loop {
            if (stride == 0u) { break; }
            if (tid < stride) { red[tid] = red[tid] + red[tid + stride]; }
            workgroupBarrier();
            stride = stride / 2u;
        }
        let qmean = red[0] / f32(p.hd);
        workgroupBarrier();

        s = 0.0;
        d = tid;
        loop {
            if (d >= p.hd) { break; }
            let t = qkv[qbase + d] - qmean;
            s = s + t * t;
            d = d + 64u;
        }
        red[tid] = s;
        workgroupBarrier();
        stride = 32u;
        loop {
            if (stride == 0u) { break; }
            if (tid < stride) { red[tid] = red[tid] + red[tid + stride]; }
            workgroupBarrier();
            stride = stride / 2u;
        }
        let qinv = inverseSqrt(red[0] / f32(p.hd) + p.eps);
        workgroupBarrier();

        var s2 = 0.0;
        d = tid;
        loop {
            if (d >= p.hd) { break; }
            s2 = s2 + qkv[kbase + d];
            d = d + 64u;
        }
        red[tid] = s2;
        workgroupBarrier();
        stride = 32u;
        loop {
            if (stride == 0u) { break; }
            if (tid < stride) { red[tid] = red[tid] + red[tid + stride]; }
            workgroupBarrier();
            stride = stride / 2u;
        }
        let kmean = red[0] / f32(p.hd);
        workgroupBarrier();

        s2 = 0.0;
        d = tid;
        loop {
            if (d >= p.hd) { break; }
            let t = qkv[kbase + d] - kmean;
            s2 = s2 + t * t;
            d = d + 64u;
        }
        red[tid] = s2;
        workgroupBarrier();
        stride = 32u;
        loop {
            if (stride == 0u) { break; }
            if (tid < stride) { red[tid] = red[tid] + red[tid + stride]; }
            workgroupBarrier();
            stride = stride / 2u;
        }
        let kinv = inverseSqrt(red[0] / f32(p.hd) + p.eps);
        workgroupBarrier();

        d = tid;
        loop {
            if (d >= p.hd) { break; }
            qn[d] = (qkv[qbase + d] - qmean) * qinv * qw[d] + qb[d];
            kn[d] = (qkv[kbase + d] - kmean) * kinv * kw[d] + kb[d];
            d = d + 64u;
        }
    } else {
        var d = tid;
        loop {
            if (d >= p.hd) { break; }
            qn[d] = qkv[qbase + d];
            kn[d] = qkv[kbase + d];
            d = d + 64u;
        }
    }

    var d = tid;
    loop {
        if (d >= p.hd) { break; }
        vout[(h * p.hd + d) * p.l + token] = qkv[vbase + d];
        d = d + 64u;
    }
    workgroupBarrier();

    if (p.has_rope != 0u) {
        if (tid < 16u) {
            let py = pos[token * 2u];
            let px = pos[token * 2u + 1u];
            let cv = cos_tab[py * 32u + tid];
            let sv = sin_tab[py * 32u + tid];
            let ch = cos_tab[px * 32u + tid];
            let sh = sin_tab[px * 32u + tid];
            let d0 = tid;
            let d1 = tid + 16u;
            let d2 = tid + 32u;
            let d3 = tid + 48u;
            let q0 = qn[d0];
            let q1 = qn[d1];
            let q2 = qn[d2];
            let q3 = qn[d3];
            qout[(h * p.l + token) * p.hd + d0] = q0 * cv - q1 * sv;
            qout[(h * p.l + token) * p.hd + d1] = q1 * cv + q0 * sv;
            qout[(h * p.l + token) * p.hd + d2] = q2 * ch - q3 * sh;
            qout[(h * p.l + token) * p.hd + d3] = q3 * ch + q2 * sh;
            let k0 = kn[d0];
            let k1 = kn[d1];
            let k2 = kn[d2];
            let k3 = kn[d3];
            kout[(h * p.l + token) * p.hd + d0] = k0 * cv - k1 * sv;
            kout[(h * p.l + token) * p.hd + d1] = k1 * cv + k0 * sv;
            kout[(h * p.l + token) * p.hd + d2] = k2 * ch - k3 * sh;
            kout[(h * p.l + token) * p.hd + d3] = k3 * ch + k2 * sh;
        }
    } else {
        d = tid;
        loop {
            if (d >= p.hd) { break; }
            qout[(h * p.l + token) * p.hd + d] = qn[d];
            kout[(h * p.l + token) * p.hd + d] = kn[d];
            d = d + 64u;
        }
    }
}
