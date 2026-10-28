use crate::gpu::{Buf, Gpu};
use crate::ops::{gemm, softmax, QkvParams, ACT_NONE};

#[derive(Clone, Copy)]
pub struct AttnCfg {
    pub l: u32,
    pub c: u32,
    pub heads: u32,
    pub hd: u32,
    pub eps: f32,
    pub qk_norm: bool,
    pub rope: bool,
}

#[allow(clippy::too_many_arguments)]
pub fn prepare_qkv(
    gpu: &Gpu,
    enc: &mut wgpu::CommandEncoder,
    qkv: &Buf,
    cfg: AttnCfg,
    qw: &Buf,
    qb: &Buf,
    kw: &Buf,
    kb: &Buf,
    pos: &Buf,
    cos_tab: &Buf,
    sin_tab: &Buf,
) -> (Buf, Buf, Buf) {
    let l = cfg.l;
    let c = cfg.c;
    let h = cfg.heads;
    let hd = cfg.hd;
    let q = gpu.buffer((h * l * hd) as u64 * 4, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC);
    let k = gpu.buffer((h * l * hd) as u64 * 4, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC);
    let v = gpu.buffer((h * hd * l) as u64 * 4, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC);
    let p = QkvParams {
        l,
        c,
        heads: h,
        hd,
        has_norm: cfg.qk_norm as u32,
        has_rope: cfg.rope as u32,
        eps: cfg.eps,
        pad: 0,
    };
    gpu.dispatch(
        enc,
        "qkv_prepare",
        &[qkv, qw, qb, kw, kb, pos, cos_tab, sin_tab, &q, &k, &v],
        bytemuck::bytes_of(&p),
        (l, h, 1),
    );
    (q, k, v)
}

#[allow(clippy::too_many_arguments)]
pub fn attention(
    gpu: &Gpu,
    enc: &mut wgpu::CommandEncoder,
    qkv: &Buf,
    cfg: AttnCfg,
    qw: &Buf,
    qb: &Buf,
    kw: &Buf,
    kb: &Buf,
    pos: &Buf,
    cos_tab: &Buf,
    sin_tab: &Buf,
) -> Buf {
    let l = cfg.l;
    let c = cfg.c;
    let h = cfg.heads;
    let hd = cfg.hd;
    let (q, k, v) = prepare_qkv(gpu, enc, qkv, cfg, qw, qb, kw, kb, pos, cos_tab, sin_tab);

    let block = 256u32;
    let scores = gpu.buffer((block * l) as u64 * 4, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC);
    let out = gpu.buffer((l * c) as u64 * 4, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC);
    let scale = 1.0f32 / (hd as f32).sqrt();
    let mut qb0 = 0u32;
    while qb0 < l {
        let m = block.min(l - qb0);
        for head in 0..h {
            let qslice = q.slice(((head * l + qb0) * hd) as u64 * 4, (m * hd) as u64 * 4);
            let kslice = k.slice((head * l * hd) as u64 * 4, (l * hd) as u64 * 4);
            let sslice = scores.slice(0, (m * l) as u64 * 4);
            gemm(
                gpu,
                enc,
                &qslice,
                &kslice,
                &gpu.dummy,
                &gpu.dummy,
                &gpu.dummy,
                &sslice,
                m,
                l,
                hd,
                hd,
                hd,
                l,
                0,
                0,
                ACT_NONE,
                scale,
            );
            softmax(gpu, enc, &sslice, m, l);
            let vslice = v.slice((head * hd * l) as u64 * 4, (hd * l) as u64 * 4);
            let oslice = out.slice((qb0 * c) as u64 * 4, (m * c) as u64 * 4);
            gemm(
                gpu,
                enc,
                &sslice,
                &vslice,
                &gpu.dummy,
                &gpu.dummy,
                &gpu.dummy,
                &oslice,
                m,
                hd,
                l,
                l,
                l,
                c,
                head * hd,
                0,
                ACT_NONE,
                1.0,
            );
        }
        qb0 += block;
    }
    out
}
