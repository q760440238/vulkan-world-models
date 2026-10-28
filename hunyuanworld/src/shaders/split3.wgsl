struct Params {
    s: u32,
    c: u32,
    pad0: u32,
    pad1: u32,
};

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> a: array<f32>;
@group(0) @binding(2) var<storage, read_write> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> c: array<f32>;
@group(0) @binding(4) var<uniform> p: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.s * p.c) { return; }
    let row = i / p.c;
    let col = i % p.c;
    let base = row * 3u * p.c + col;
    a[i] = x[base];
    b[i] = x[base + p.c];
    c[i] = x[base + 2u * p.c];
}
