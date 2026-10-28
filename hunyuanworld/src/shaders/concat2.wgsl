struct Params {
    rows: u32,
    c: u32,
    pad0: u32,
    pad1: u32,
};

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> p: Params;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_index) tid: u32) {
    let row = wid.x;
    if (row >= p.rows) { return; }
    var d = tid;
    loop {
        if (d >= 2u * p.c) { break; }
        var v = 0.0;
        if (d < p.c) { v = a[row * p.c + d]; } else { v = b[row * p.c + d - p.c]; }
        out[row * 2u * p.c + d] = v;
        d = d + 256u;
    }
}
