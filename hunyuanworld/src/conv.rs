use crate::gpu::{Buf, Gpu};
use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct ConvParams {
    pub n: u32,
    pub cin: u32,
    pub h: u32,
    pub w: u32,
    pub cout: u32,
    pub kh: u32,
    pub kw: u32,
    pub stride: u32,
    pub pad: u32,
    pub has_bias: u32,
    pub act: u32,
    pub oh: u32,
    pub ow: u32,
    pub co_start: u32,
    pub cout_total: u32,
    pub pad1: u32,
}

#[allow(clippy::too_many_arguments)]
pub fn conv2d(
    gpu: &Gpu,
    enc: &mut wgpu::CommandEncoder,
    x: &Buf,
    wt: &Buf,
    bias: &Buf,
    y: &Buf,
    n: u32,
    cin: u32,
    h: u32,
    w: u32,
    cout: u32,
    kh: u32,
    kw: u32,
    stride: u32,
    pad: u32,
    has_bias: bool,
    act: u32,
    oh: u32,
    ow: u32,
) {
    let per = n * oh * ow;
    let max_chunk = (65535u32 * 256 / per.max(1)).max(1).min(cout);
    let mut start = 0u32;
    while start < cout {
        let chunk = max_chunk.min(cout - start);
        let p = ConvParams {
            n, cin, h, w, cout: chunk, kh, kw, stride, pad,
            has_bias: has_bias as u32, act, oh, ow, co_start: start, cout_total: cout, pad1: 0,
        };
        let total = n * chunk * oh * ow;
        gpu.dispatch(enc, "conv2d", &[x, wt, bias, y], bytemuck::bytes_of(&p), ((total + 255) / 256, 1, 1));
        start += chunk;
    }
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct DeconvParams {
    pub n: u32,
    pub cin: u32,
    pub cout: u32,
    pub h: u32,
    pub w: u32,
    pub kh: u32,
    pub kw: u32,
    pub oh: u32,
    pub ow: u32,
    pub has_bias: u32,
    pub pad0: u32,
    pub pad1: u32,
}

#[allow(clippy::too_many_arguments)]
pub fn deconv(
    gpu: &Gpu,
    enc: &mut wgpu::CommandEncoder,
    x: &Buf,
    wt: &Buf,
    bias: &Buf,
    y: &Buf,
    n: u32,
    cin: u32,
    cout: u32,
    h: u32,
    w: u32,
    kh: u32,
    kw: u32,
    has_bias: bool,
) {
    let oh = h * kh;
    let ow = w * kw;
    let p = DeconvParams {
        n, cin, cout, h, w, kh, kw, oh, ow, has_bias: has_bias as u32, pad0: 0, pad1: 0,
    };
    let total = n * cout * oh * ow;
    gpu.dispatch(enc, "deconv", &[x, wt, bias, y], bytemuck::bytes_of(&p), ((total + 255) / 256, 1, 1));
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct InterpParams {
    pub n: u32,
    pub c: u32,
    pub h: u32,
    pub w: u32,
    pub oh: u32,
    pub ow: u32,
    pub c_start: u32,
    pub c_total: u32,
}

#[allow(clippy::too_many_arguments)]
pub fn interpolate(
    gpu: &Gpu,
    enc: &mut wgpu::CommandEncoder,
    x: &Buf,
    y: &Buf,
    n: u32,
    c: u32,
    h: u32,
    w: u32,
    oh: u32,
    ow: u32,
) {
    let per = n * oh * ow;
    let max_chunk = (65535u32 * 256 / per.max(1)).max(1).min(c);
    let mut start = 0u32;
    while start < c {
        let chunk = max_chunk.min(c - start);
        let p = InterpParams { n, c: chunk, h, w, oh, ow, c_start: start, c_total: c };
        let total = n * chunk * oh * ow;
        gpu.dispatch(enc, "interpolate", &[x, y], bytemuck::bytes_of(&p), ((total + 255) / 256, 1, 1));
        start += chunk;
    }
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct PosParams {
    pub n: u32,
    pub c: u32,
    pub h: u32,
    pub w: u32,
    pub c_start: u32,
    pub c_total: u32,
    pub pad2: u32,
    pub pad3: u32,
}

pub fn pos_add(
    gpu: &Gpu,
    enc: &mut wgpu::CommandEncoder,
    x: &Buf,
    table: &Buf,
    y: &Buf,
    n: u32,
    c: u32,
    h: u32,
    w: u32,
) {
    let per = n * h * w;
    let max_chunk = (65535u32 * 256 / per.max(1)).max(1).min(c);
    let mut start = 0u32;
    while start < c {
        let chunk = max_chunk.min(c - start);
        let p = PosParams { n, c: chunk, h, w, c_start: start, c_total: c, pad2: 0, pad3: 0 };
        let total = n * chunk * h * w;
        gpu.dispatch(enc, "pos_add", &[x, table, y], bytemuck::bytes_of(&p), ((total + 255) / 256, 1, 1));
        start += chunk;
    }
}
