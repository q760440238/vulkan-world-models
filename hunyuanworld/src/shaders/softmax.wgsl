struct Params {
    rows: u32,
    cols: u32,
};

@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<uniform> p: Params;

var<workgroup> red: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_index) tid: u32) {
    let row = wid.x;
    if (row >= p.rows) { return; }
    let base = row * p.cols;

    var m = -3.0e38;
    var i = tid;
    loop {
        if (i >= p.cols) { break; }
        m = max(m, x[base + i]);
        i = i + 256u;
    }
    red[tid] = m;
    workgroupBarrier();
    var stride = 128u;
    loop {
        if (stride == 0u) { break; }
        if (tid < stride) { red[tid] = max(red[tid], red[tid + stride]); }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let mx = red[0];
    workgroupBarrier();

    var s = 0.0;
    i = tid;
    loop {
        if (i >= p.cols) { break; }
        s = s + exp(x[base + i] - mx);
        i = i + 256u;
    }
    red[tid] = s;
    workgroupBarrier();
    stride = 128u;
    loop {
        if (stride == 0u) { break; }
        if (tid < stride) { red[tid] = red[tid] + red[tid + stride]; }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let inv = 1.0 / red[0];

    i = tid;
    loop {
        if (i >= p.cols) { break; }
        x[base + i] = exp(x[base + i] - mx) * inv;
        i = i + 256u;
    }
}
