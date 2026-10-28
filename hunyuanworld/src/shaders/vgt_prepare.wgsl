struct Params {
    s: u32,
    l: u32,
    c: u32,
    npatch: u32,
    nreg: u32,
    start: u32,
    pad0: u32,
    pad1: u32,
};

@group(0) @binding(0) var<storage, read> pat: array<f32>;
@group(0) @binding(1) var<storage, read> cam: array<f32>;
@group(0) @binding(2) var<storage, read> reg: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<uniform> p: Params;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_index) tid: u32) {
    let flat = wid.x;
    let s = flat / p.l;
    let row = flat % p.l;
    if (s >= p.s) { return; }
    let src = select(1u, 0u, s == 0u);
    var d = tid;
    loop {
        if (d >= p.c) { break; }
        var v = 0.0;
        if (row == 0u) {
            v = cam[src * p.c + d];
        } else if (row <= p.nreg) {
            v = reg[(src * p.nreg + row - 1u) * p.c + d];
        } else if (row >= p.start) {
            v = pat[(s * p.npatch + row - p.start) * p.c + d];
        }
        out[(s * p.l + row) * p.c + d] = v;
        d = d + 256u;
    }
}
