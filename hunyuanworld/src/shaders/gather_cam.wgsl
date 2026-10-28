struct Params {
    s: u32,
    l: u32,
    c: u32,
    pad: u32,
};

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> p: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.s * p.c) { return; }
    let s = i / p.c;
    let c = i % p.c;
    out[i] = x[s * p.l * p.c + c];
}
