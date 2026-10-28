struct Params {
    n: u32,
    cin: u32,
    h: u32,
    w: u32,
    cout: u32,
    kh: u32,
    kw: u32,
    stride: u32,
    pad: u32,
    has_bias: u32,
    act: u32,
    oh: u32,
    ow: u32,
    co_start: u32,
    cout_total: u32,
    pad1: u32,
};

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> wt: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@group(0) @binding(4) var<uniform> p: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total = p.n * p.cout * p.oh * p.ow;
    let idx = gid.x;
    if (idx >= total) { return; }
    let ox = idx % p.ow;
    let t1 = idx / p.ow;
    let oy = t1 % p.oh;
    let t2 = t1 / p.oh;
    let co = t2 % p.cout;
    let b = t2 / p.cout;
    let co_g = co + p.co_start;
    var acc = 0.0;
    for (var ci = 0u; ci < p.cin; ci = ci + 1u) {
        for (var ky = 0u; ky < p.kh; ky = ky + 1u) {
            let iy = oy * p.stride + ky;
            if (iy < p.pad) { continue; }
            let yy = iy - p.pad;
            if (yy >= p.h) { continue; }
            for (var kx = 0u; kx < p.kw; kx = kx + 1u) {
                let ix = ox * p.stride + kx;
                if (ix < p.pad) { continue; }
                let xx = ix - p.pad;
                if (xx >= p.w) { continue; }
                let xv = x[(b * p.cin + ci) * p.h * p.w + yy * p.w + xx];
                let wv = wt[((co_g * p.cin + ci) * p.kh + ky) * p.kw + kx];
                acc = acc + xv * wv;
            }
        }
    }
    if (p.has_bias != 0u) { acc = acc + bias[co_g]; }
    if (p.act == 2u) { acc = max(acc, 0.0); }
    if (p.act == 3u) { acc = acc / (1.0 + exp(-acc)); }
    y[(b * p.cout_total + co_g) * p.oh * p.ow + oy * p.ow + ox] = acc;
}
