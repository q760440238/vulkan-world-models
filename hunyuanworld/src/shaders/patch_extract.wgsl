struct Params {
    s: u32,
    l: u32,
    npatch: u32,
    c: u32,
    start: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<uniform> p: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total = p.s * p.npatch * p.c;
    let i = gid.x;
    if (i >= total) { return; }
    let d = i % p.c;
    let t = i / p.c;
    let pix = t % p.npatch;
    let s = t / p.npatch;
    dst[i] = src[(s * p.l + p.start + pix) * p.c + d];
}
