use crate::attn::{attention, AttnCfg};
use crate::gpu::{Buf, Gpu, WeightStore};
use crate::ops::{add, gemm, layernorm, ACT_GELU, ACT_NONE, HAS_BIAS, HAS_RESIDUAL, HAS_SCALE};

pub struct BlockWeights {
    pub norm1_w: Buf,
    pub norm1_b: Buf,
    pub qkv_w: Buf,
    pub qkv_b: Buf,
    pub q_norm_w: Option<Buf>,
    pub q_norm_b: Option<Buf>,
    pub k_norm_w: Option<Buf>,
    pub k_norm_b: Option<Buf>,
    pub proj_w: Buf,
    pub proj_b: Buf,
    pub ls1: Buf,
    pub norm2_w: Buf,
    pub norm2_b: Buf,
    pub fc1_w: Buf,
    pub fc1_b: Buf,
    pub fc2_w: Buf,
    pub fc2_b: Buf,
    pub ls2: Buf,
}

impl BlockWeights {
    pub fn load(ws: &WeightStore, prefix: &str) -> Self {
        let g = |s: &str| ws.get(&format!("{prefix}.{s}"));
        Self {
            norm1_w: g("norm1.weight"),
            norm1_b: g("norm1.bias"),
            qkv_w: g("attn.qkv.weight"),
            qkv_b: g("attn.qkv.bias"),
            q_norm_w: ws.try_get(&format!("{prefix}.attn.q_norm.weight")),
            q_norm_b: ws.try_get(&format!("{prefix}.attn.q_norm.bias")),
            k_norm_w: ws.try_get(&format!("{prefix}.attn.k_norm.weight")),
            k_norm_b: ws.try_get(&format!("{prefix}.attn.k_norm.bias")),
            proj_w: g("attn.proj.weight"),
            proj_b: g("attn.proj.bias"),
            ls1: g("ls1.gamma"),
            norm2_w: g("norm2.weight"),
            norm2_b: g("norm2.bias"),
            fc1_w: g("mlp.fc1.weight"),
            fc1_b: g("mlp.fc1.bias"),
            fc2_w: g("mlp.fc2.weight"),
            fc2_b: g("mlp.fc2.bias"),
            ls2: g("ls2.gamma"),
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn block_forward(
    gpu: &Gpu,
    enc: &mut wgpu::CommandEncoder,
    x: &Buf,
    w: &BlockWeights,
    cfg: AttnCfg,
    pos: &Buf,
    cos: &Buf,
    sin: &Buf,
    ln_eps: f32,
) -> Buf {
    let m = cfg.l;
    let c = cfg.c;
    let n3 = 3 * c;
    let hidden = 4 * c;

    let n1 = gpu.buffer((m * c) as u64 * 4, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC);
    layernorm(gpu, enc, x, &w.norm1_w, &w.norm1_b, &n1, m, c, ln_eps, true);

    let qkv = gpu.buffer((m * n3) as u64 * 4, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC);
    gemm(gpu, enc, &n1, &w.qkv_w, &w.qkv_b, &gpu.dummy, &gpu.dummy, &qkv,
         m, n3, c, c, c, n3, 0, HAS_BIAS, ACT_NONE, 1.0);

    let qw = w.q_norm_w.as_ref().unwrap_or(&gpu.dummy);
    let qb = w.q_norm_b.as_ref().unwrap_or(&gpu.dummy);
    let kw = w.k_norm_w.as_ref().unwrap_or(&gpu.dummy);
    let kb = w.k_norm_b.as_ref().unwrap_or(&gpu.dummy);
    let attn = attention(gpu, enc, &qkv, cfg, qw, qb, kw, kb, pos, cos, sin);

    let x1 = gpu.buffer((m * c) as u64 * 4, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC);
    gemm(gpu, enc, &attn, &w.proj_w, &w.proj_b, &w.ls1, x, &x1,
         m, c, c, c, c, c, 0, HAS_BIAS | HAS_SCALE | HAS_RESIDUAL, ACT_NONE, 1.0);

    let n2 = gpu.buffer((m * c) as u64 * 4, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC);
    layernorm(gpu, enc, &x1, &w.norm2_w, &w.norm2_b, &n2, m, c, ln_eps, true);

    let h = gpu.buffer((m * hidden) as u64 * 4, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC);
    gemm(gpu, enc, &n2, &w.fc1_w, &w.fc1_b, &gpu.dummy, &gpu.dummy, &h,
         m, hidden, c, c, c, hidden, 0, HAS_BIAS, ACT_GELU, 1.0);

    let x2 = gpu.buffer((m * c) as u64 * 4, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC);
    gemm(gpu, enc, &h, &w.fc2_w, &w.fc2_b, &w.ls2, &x1, &x2,
         m, c, hidden, hidden, hidden, c, 0, HAS_BIAS | HAS_SCALE | HAS_RESIDUAL, ACT_NONE, 1.0);
    x2
}

pub fn add_buf(gpu: &Gpu, enc: &mut wgpu::CommandEncoder, a: &Buf, b: &Buf, y: &Buf, n: usize) {
    add(gpu, enc, a, b, y, n);
}
