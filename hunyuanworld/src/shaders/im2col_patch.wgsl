struct Params {
    c: u32,
    h: u32,
    w: u32,
    gw: u32,
    n: u32,
};

@group(0) @binding(0) var<storage, read> img: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> p: Params;

const MEAN: array<f32, 3> = array<f32, 3>(0.485, 0.456, 0.406);
const STD: array<f32, 3> = array<f32, 3>(0.229, 0.224, 0.225);
const PATCH: u32 = 14u;
const KSIZE: u32 = 588u;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let id = gid.x;
    if (id >= p.n) { return; }
    let pi = id / KSIZE;
    let k = id % KSIZE;
    let py = pi / p.gw;
    let px = pi % p.gw;
    let c = k / (PATCH * PATCH);
    let rem = k % (PATCH * PATCH);
    let ky = rem / PATCH;
    let kx = rem % PATCH;
    let iy = py * PATCH + ky;
    let ix = px * PATCH + kx;
    let v = (img[c * p.h * p.w + iy * p.w + ix] - MEAN[c]) / STD[c];
    out[id] = v;
}
