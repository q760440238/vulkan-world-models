struct Params {
    rows: u32,
    cols: u32,
    eps: f32,
    has_affine: u32,
};

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> gamma: array<f32>;
@group(0) @binding(2) var<storage, read> beta: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@group(0) @binding(4) var<uniform> p: Params;

var<workgroup> red: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_index) tid: u32) {
    let row = wid.x;
    if (row >= p.rows) { return; }
    let base = row * p.cols;
    var s = 0.0;
    var i = tid;
    loop {
        if (i >= p.cols) { break; }
        s = s + x[base + i];
        i = i + 256u;
    }
    red[tid] = s;
    workgroupBarrier();
    var stride = 128u;
    loop {
        if (stride == 0u) { break; }
        if (tid < stride) { red[tid] = red[tid] + red[tid + stride]; }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let mean = red[0] / f32(p.cols);
    workgroupBarrier();
    var v = 0.0;
    i = tid;
    loop {
        if (i >= p.cols) { break; }
        let d = x[base + i] - mean;
        v = v + d * d;
        i = i + 256u;
    }
    red[tid] = v;
    workgroupBarrier();
    stride = 128u;
    loop {
        if (stride == 0u) { break; }
        if (tid < stride) { red[tid] = red[tid] + red[tid + stride]; }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let inv = inverseSqrt(red[0] / f32(p.cols) + p.eps);
    i = tid;
    loop {
        if (i >= p.cols) { break; }
        var out = (x[base + i] - mean) * inv;
        if (p.has_affine != 0u) {
            out = out * gamma[i] + beta[i];
        }
        y[base + i] = out;
        i = i + 256u;
    }
}
