struct Params {
    rows: u32,
    cols: u32,
    start: u32,
    pad: u32,
};

@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<uniform> p: Params;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_index) tid: u32) {
    let row = wid.x;
    if (row >= p.rows) { return; }
    var d = tid;
    loop {
        if (d >= p.cols) { break; }
        dst[row * p.cols + d] = src[(row + p.start) * p.cols + d];
        d = d + 256u;
    }
}
