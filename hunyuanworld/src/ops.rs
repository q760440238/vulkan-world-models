use crate::gpu::{Buf, Gpu};
use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct GemmParams {
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub lda: u32,
    pub ldb: u32,
    pub ldc: u32,
    pub c_col_offset: u32,
    pub flags: u32,
    pub act: u32,
    pub alpha: f32,
    pub spatial: u32,
    pub pad1: u32,
}

pub const HAS_BIAS: u32 = 1;
pub const HAS_SCALE: u32 = 2;
pub const HAS_RESIDUAL: u32 = 4;
pub const TRANSPOSE_OUT: u32 = 8;
pub const ACT_NONE: u32 = 0;
pub const ACT_GELU: u32 = 1;
pub const ACT_RELU: u32 = 2;
pub const ACT_SILU: u32 = 3;

#[allow(clippy::too_many_arguments)]
pub fn gemm(
    gpu: &Gpu,
    enc: &mut wgpu::CommandEncoder,
    a: &Buf,
    b: &Buf,
    bias: &Buf,
    scale: &Buf,
    residual: &Buf,
    c: &Buf,
    m: u32,
    n: u32,
    k: u32,
    lda: u32,
    ldb: u32,
    ldc: u32,
    c_col_offset: u32,
    flags: u32,
    act: u32,
    alpha: f32,
) {
    let p = GemmParams {
        m, n, k, lda, ldb, ldc, c_col_offset, flags, act, alpha, spatial: 0, pad1: 0,
    };
    gpu.dispatch(
        enc,
        "gemm",
        &[a, b, bias, scale, residual, c],
        bytemuck::bytes_of(&p),
        ((m + 63) / 64, (n + 63) / 64, 1),
    );
}

#[allow(clippy::too_many_arguments)]
pub fn gemm_transpose(
    gpu: &Gpu,
    enc: &mut wgpu::CommandEncoder,
    a: &Buf,
    b: &Buf,
    bias: &Buf,
    c: &Buf,
    m: u32,
    n: u32,
    k: u32,
    lda: u32,
    ldb: u32,
    spatial: u32,
    act: u32,
) {
    let p = GemmParams {
        m,
        n,
        k,
        lda,
        ldb,
        ldc: spatial,
        c_col_offset: 0,
        flags: HAS_BIAS | TRANSPOSE_OUT,
        act,
        alpha: 1.0,
        spatial,
        pad1: 0,
    };
    gpu.dispatch(
        enc,
        "gemm",
        &[a, b, bias, &gpu.dummy, &gpu.dummy, c],
        bytemuck::bytes_of(&p),
        ((m + 63) / 64, (n + 63) / 64, 1),
    );
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct LnParams {    pub rows: u32,
    pub cols: u32,
    pub eps: f32,
    pub has_affine: u32,
}

pub fn layernorm(
    gpu: &Gpu,
    enc: &mut wgpu::CommandEncoder,
    x: &Buf,
    gamma: &Buf,
    beta: &Buf,
    y: &Buf,
    rows: u32,
    cols: u32,
    eps: f32,
    has_affine: bool,
) {
    let p = LnParams { rows, cols, eps, has_affine: has_affine as u32 };
    gpu.dispatch(enc, "layernorm", &[x, gamma, beta, y], bytemuck::bytes_of(&p), (rows, 1, 1));
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct QkvParams {
    pub l: u32,
    pub c: u32,
    pub heads: u32,
    pub hd: u32,
    pub has_norm: u32,
    pub has_rope: u32,
    pub eps: f32,
    pub pad: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct SoftmaxParams {
    pub rows: u32,
    pub cols: u32,
}

pub fn softmax(gpu: &Gpu, enc: &mut wgpu::CommandEncoder, x: &Buf, rows: u32, cols: u32) {
    let p = SoftmaxParams { rows, cols };
    gpu.dispatch(enc, "softmax", &[x], bytemuck::bytes_of(&p), (rows, 1, 1));
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct ReluParams {
    pub n: u32,
    pub pad0: u32,
    pub pad1: u32,
    pub pad2: u32,
}

pub fn relu(gpu: &Gpu, enc: &mut wgpu::CommandEncoder, x: &Buf, y: &Buf, n: usize) {
    let max = 65535usize * 256;
    let mut start = 0usize;
    while start < n {
        let chunk = (n - start).min(max);
        let xs = x.slice((start * 4) as u64, (chunk * 4) as u64);
        let ys = y.slice((start * 4) as u64, (chunk * 4) as u64);
        let p = ReluParams { n: chunk as u32, pad0: 0, pad1: 0, pad2: 0 };
        gpu.dispatch(enc, "relu", &[&xs, &ys], bytemuck::bytes_of(&p), (((chunk as u32) + 255) / 256, 1, 1));
        start += chunk;
    }
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct PatchParams {
    pub s: u32,
    pub l: u32,
    pub npatch: u32,
    pub c: u32,
    pub start: u32,
    pub pad0: u32,
    pub pad1: u32,
    pub pad2: u32,
}

pub fn patch_extract(
    gpu: &Gpu,
    enc: &mut wgpu::CommandEncoder,
    src: &Buf,
    dst: &Buf,
    s: u32,
    l: u32,
    npatch: u32,
    c: u32,
    start: u32,
) {
    let p = PatchParams { s, l, npatch, c, start, pad0: 0, pad1: 0, pad2: 0 };
    let total = s * npatch * c;
    gpu.dispatch(enc, "patch_extract", &[src, dst], bytemuck::bytes_of(&p), ((total + 255) / 256, 1, 1));
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct AddParams {
    pub n: u32,
}

pub fn add(gpu: &Gpu, enc: &mut wgpu::CommandEncoder, a: &Buf, b: &Buf, y: &Buf, n: usize) {
    let max = 65535usize * 256;
    let mut start = 0usize;
    while start < n {
        let chunk = (n - start).min(max);
        let as_ = a.slice((start * 4) as u64, (chunk * 4) as u64);
        let bs = b.slice((start * 4) as u64, (chunk * 4) as u64);
        let ys = y.slice((start * 4) as u64, (chunk * 4) as u64);
        let p = AddParams { n: chunk as u32 };
        gpu.dispatch(enc, "add", &[&as_, &bs, &ys], bytemuck::bytes_of(&p), (((chunk as u32) + 255) / 256, 1, 1));
        start += chunk;
    }
}
