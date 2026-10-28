struct Params {
    n: u32,
    cin: u32,
    cout: u32,
    h: u32,
    w: u32,
    kh: u32,
    kw: u32,
    oh: u32,
    ow: u32,
    has_bias: u32,
    pad0: u32,
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
    let iy = oy / p.kh;
    let ky = oy % p.kh;
    let ix = ox / p.kw;
    let kx = ox % p.kw;
    var acc = 0.0;
    for (var ci = 0u; ci < p.cin; ci = ci + 1u) {
        let xv = x[(b * p.cin + ci) * p.h * p.w + iy * p.w + ix];
        let wv = wt[((ci * p.cout + co) * p.kh + ky) * p.kw + kx];
        acc = acc + xv * wv;
    }
    if (p.has_bias != 0u) { acc = acc + bias[co]; }
    y[(b * p.cout + co) * p.oh * p.ow + oy * p.ow + ox] = acc;
}
