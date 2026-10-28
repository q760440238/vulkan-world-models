struct Params {
    n: u32,
    c: u32,
    h: u32,
    w: u32,
    oh: u32,
    ow: u32,
    c_start: u32,
    c_total: u32,
};

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;
@group(0) @binding(2) var<uniform> p: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total = p.n * p.c * p.oh * p.ow;
    let idx = gid.x;
    if (idx >= total) { return; }
    let ox = idx % p.ow;
    let t1 = idx / p.ow;
    let oy = t1 % p.oh;
    let t2 = t1 / p.oh;
    let c = t2 % p.c;
    let b = t2 / p.c;
    let cg = c + p.c_start;
    let sy = f32(oy) * f32(p.h - 1u) / f32(p.oh - 1u);
    let sx = f32(ox) * f32(p.w - 1u) / f32(p.ow - 1u);
    let y0 = u32(floor(sy));
    let x0 = u32(floor(sx));
    let y1 = min(y0 + 1u, p.h - 1u);
    let x1 = min(x0 + 1u, p.w - 1u);
    let wy = sy - f32(y0);
    let wx = sx - f32(x0);
    let base = (b * p.c_total + cg) * p.h * p.w;
    let v00 = x[base + y0 * p.w + x0];
    let v01 = x[base + y0 * p.w + x1];
    let v10 = x[base + y1 * p.w + x0];
    let v11 = x[base + y1 * p.w + x1];
    let top = v00 * (1.0 - wx) + v01 * wx;
    let bot = v10 * (1.0 - wx) + v11 * wx;
    y[(b * p.c_total + cg) * p.oh * p.ow + oy * p.ow + ox] = top * (1.0 - wy) + bot * wy;
}
