struct Params {
    npatch: u32,
    c: u32,
    nreg: u32,
    pad: u32,
};

@group(0) @binding(0) var<storage, read> pat: array<f32>;
@group(0) @binding(1) var<storage, read> cls: array<f32>;
@group(0) @binding(2) var<storage, read> pos: array<f32>;
@group(0) @binding(3) var<storage, read> reg: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;
@group(0) @binding(5) var<uniform> p: Params;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_index) tid: u32) {
    let row = wid.x;
    let rows = 1u + p.nreg + p.npatch;
    if (row >= rows) { return; }
    var d = tid;
    loop {
        if (d >= p.c) { break; }
        var v = 0.0;
        if (row == 0u) {
            v = cls[d] + pos[d];
        } else if (row <= p.nreg) {
            v = reg[(row - 1u) * p.c + d];
        } else {
            let pi = row - 1u - p.nreg;
            v = pat[pi * p.c + d] + pos[(pi + 1u) * p.c + d];
        }
        out[row * p.c + d] = v;
        d = d + 256u;
    }
}
