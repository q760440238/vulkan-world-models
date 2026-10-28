struct Params {
    n: u32,
    c: u32,
    pad0: u32,
    pad1: u32,
};

@group(0) @binding(0) var<storage, read> normed: array<f32>;
@group(0) @binding(1) var<storage, read> shift: array<f32>;
@group(0) @binding(2) var<storage, read> scale: array<f32>;
@group(0) @binding(3) var<storage, read> gate: array<f32>;
@group(0) @binding(4) var<storage, read> cam: array<f32>;
@group(0) @binding(5) var<storage, read_write> out: array<f32>;
@group(0) @binding(6) var<uniform> p: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.n) { return; }
    let m = normed[i] * (1.0 + scale[i]) + shift[i];
    out[i] = gate[i] * m + cam[i];
}
