struct Params {
    m: u32,
    n: u32,
    k: u32,
    lda: u32,
    ldb: u32,
    ldc: u32,
    c_col_offset: u32,
    flags: u32,
    act: u32,
    alpha: f32,
    spatial: u32,
    pad1: u32,
};

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read> scale: array<f32>;
@group(0) @binding(4) var<storage, read> residual: array<f32>;
@group(0) @binding(5) var<storage, read_write> c: array<f32>;
@group(0) @binding(6) var<uniform> p: Params;

var<workgroup> atile: array<f32, 1024>;
var<workgroup> btile: array<f32, 1024>;

fn erf(x: f32) -> f32 {
    let s = sign(x);
    let ax = abs(x);
    let t = 1.0 / (1.0 + 0.3275911 * ax);
    let poly = ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t;
    return s * (1.0 - poly * exp(-ax * ax));
}

fn gelu(x: f32) -> f32 {
    return 0.5 * x * (1.0 + erf(x * 0.7071067811865476));
}

fn act(v: f32, kind: u32) -> f32 {
    if (kind == 0u) { return v; }
    if (kind == 1u) { return gelu(v); }
    if (kind == 2u) { return max(v, 0.0); }
    if (kind == 3u) { return v / (1.0 + exp(-v)); }
    return v;
}

@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.y * 16u + lid.x;
    let row0 = wid.x * 64u;
    let col0 = wid.y * 64u;
    var acc: array<f32, 16>;
    for (var i = 0u; i < 16u; i = i + 1u) { acc[i] = 0.0; }

    var k0 = 0u;
    loop {
        if (k0 >= p.k) { break; }
        for (var i = 0u; i < 4u; i = i + 1u) {
            let idx = tid + i * 256u;
            let r = idx / 16u;
            let kk = idx % 16u;
            let gr = row0 + r;
            let gk = k0 + kk;
            var va = 0.0;
            if (gr < p.m && gk < p.k) { va = a[gr * p.lda + gk]; }
            atile[idx] = va;
            let gn = col0 + r;
            var vb = 0.0;
            if (gn < p.n && gk < p.k) { vb = b[gn * p.ldb + gk]; }
            btile[idx] = vb;
        }
        workgroupBarrier();
        for (var kk = 0u; kk < 16u; kk = kk + 1u) {
            let a0 = atile[(lid.y * 4u + 0u) * 16u + kk];
            let a1 = atile[(lid.y * 4u + 1u) * 16u + kk];
            let a2 = atile[(lid.y * 4u + 2u) * 16u + kk];
            let a3 = atile[(lid.y * 4u + 3u) * 16u + kk];
            let b0 = btile[(lid.x * 4u + 0u) * 16u + kk];
            let b1 = btile[(lid.x * 4u + 1u) * 16u + kk];
            let b2 = btile[(lid.x * 4u + 2u) * 16u + kk];
            let b3 = btile[(lid.x * 4u + 3u) * 16u + kk];
            acc[0] = acc[0] + a0 * b0;
            acc[1] = acc[1] + a0 * b1;
            acc[2] = acc[2] + a0 * b2;
            acc[3] = acc[3] + a0 * b3;
            acc[4] = acc[4] + a1 * b0;
            acc[5] = acc[5] + a1 * b1;
            acc[6] = acc[6] + a1 * b2;
            acc[7] = acc[7] + a1 * b3;
            acc[8] = acc[8] + a2 * b0;
            acc[9] = acc[9] + a2 * b1;
            acc[10] = acc[10] + a2 * b2;
            acc[11] = acc[11] + a2 * b3;
            acc[12] = acc[12] + a3 * b0;
            acc[13] = acc[13] + a3 * b1;
            acc[14] = acc[14] + a3 * b2;
            acc[15] = acc[15] + a3 * b3;
        }
        workgroupBarrier();
        k0 = k0 + 16u;
    }

    let has_bias = (p.flags & 1u) != 0u;
    let has_scale = (p.flags & 2u) != 0u;
    let has_residual = (p.flags & 4u) != 0u;
    for (var i = 0u; i < 4u; i = i + 1u) {
        let gr = row0 + lid.y * 4u + i;
        if (gr >= p.m) { continue; }
        for (var j = 0u; j < 4u; j = j + 1u) {
            let gc = col0 + lid.x * 4u + j;
            if (gc >= p.n) { continue; }
            var v = acc[i * 4u + j];
            if (has_bias) { v = v + bias[gc]; }
            v = act(v, p.act);
            v = v * p.alpha;
            if (has_scale) { v = v * scale[gc]; }
            var oi = gr * p.ldc + gc + p.c_col_offset;
            if ((p.flags & 8u) != 0u) {
                let s = gr / p.spatial;
                let pix = gr % p.spatial;
                oi = (s * p.n + gc) * p.spatial + pix;
            }
            if (has_residual) { v = v + residual[oi]; }
            c[oi] = v;
        }
    }
}
