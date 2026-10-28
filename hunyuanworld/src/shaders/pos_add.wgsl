struct Params {
    n: u32,
    c: u32,
    h: u32,
    w: u32,
    c_start: u32,
    c_total: u32,
    pad2: u32,
    pad3: u32,
};

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> table: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> p: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total = p.n * p.c * p.h * p.w;
    let idx = gid.x;
    if (idx >= total) { return; }
    let pix = idx % (p.h * p.w);
    let t = idx / (p.h * p.w);
    let c = t % p.c;
    let b = t / p.c;
    let cg = c + p.c_start;
    y[(b * p.c_total + cg) * p.h * p.w + pix] = x[(b * p.c_total + cg) * p.h * p.w + pix] + table[pix * p.c_total + cg];
}
