use crate::attn::AttnCfg;
use crate::blocks::{block_forward, BlockWeights};
use crate::gpu::{Buf, Gpu, WeightStore};
use crate::ops::{add, gemm, layernorm, ACT_GELU, ACT_NONE, HAS_BIAS};
use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct SiluParams { n: u32, c: u32, pad0: u32, pad1: u32 }
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct SplitParams { s: u32, c: u32, pad0: u32, pad1: u32 }
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct ModParams { n: u32, c: u32, pad0: u32, pad1: u32 }
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct CamParams { s: u32, l: u32, c: u32, pad: u32 }

pub fn camera_head(
    gpu: &Gpu,
    enc: &mut wgpu::CommandEncoder,
    tokens_last: &Buf,
    s: u32,
    l: u32,
    ws: &WeightStore,
) -> Buf {
    let mut trace = Vec::new();
    camera_head_trace(gpu, enc, tokens_last, s, l, ws, &mut trace)
}

pub fn camera_head_trace(
    gpu: &Gpu,
    enc: &mut wgpu::CommandEncoder,
    tokens_last: &Buf,
    s: u32,
    l: u32,
    ws: &WeightStore,
    trace: &mut Vec<(String, Buf)>,
) -> Buf {
    let c = 2048u32;
    let init = ws.get("cam_head.init_token");
    let init_host = gpu.read_f32(&init);
    let mut repeated = Vec::with_capacity((s * 9) as usize);
    for _ in 0..s { repeated.extend_from_slice(&init_host); }
    let init_buf = gpu.upload(bytemuck::cast_slice(&repeated));

    let cam = gpu.buffer((s * c) as u64 * 4, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC);
    let cp = CamParams { s, l, c, pad: 0 };
    gpu.dispatch(enc, "gather_cam", &[tokens_last, &cam], bytemuck::bytes_of(&cp), ((s * c + 255) / 256, 1, 1));

    let token_norm_w = ws.get("cam_head.token_norm.weight");
    let token_norm_b = ws.get("cam_head.token_norm.bias");
    let cam_norm = gpu.buffer((s * c) as u64 * 4, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC);
    layernorm(gpu, enc, &cam, &token_norm_w, &token_norm_b, &cam_norm, s, c, 1e-5, true);
    trace.push(("cam_norm".to_string(), cam_norm.clone()));

    let param_embed_w = ws.get("cam_head.param_embed.weight");
    let param_embed_b = ws.get("cam_head.param_embed.bias");
    let adapt_w = ws.get("cam_head.adapt_norm_gen.1.weight");
    let adapt_b = ws.get("cam_head.adapt_norm_gen.1.bias");
    let out_norm_w = ws.get("cam_head.out_norm.weight");
    let out_norm_b = ws.get("cam_head.out_norm.bias");
    let fc1_w = ws.get("cam_head.param_predictor.fc1.weight");
    let fc1_b = ws.get("cam_head.param_predictor.fc1.bias");
    let fc2_w = ws.get("cam_head.param_predictor.fc2.weight");
    let fc2_b = ws.get("cam_head.param_predictor.fc2.bias");
    let blocks: Vec<BlockWeights> = (0..4)
        .map(|i| BlockWeights::load(ws, &format!("cam_head.refine_net.{i}")))
        .collect();

    let mut curr: Option<Buf> = None;
    for step in 0..4 {
        let pred_in = curr.as_ref().unwrap_or(&init_buf);
        let net = gpu.buffer((s * c) as u64 * 4, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC);
        gemm(gpu, enc, pred_in, &param_embed_w, &param_embed_b, &gpu.dummy, &gpu.dummy, &net,
             s, c, 9, 9, 9, c, 0, HAS_BIAS, ACT_NONE, 1.0);
        trace.push((format!("net{step}"), net.clone()));

        let act = gpu.buffer((s * c) as u64 * 4, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC);
        let sp = SiluParams { n: s * c, c, pad0: 0, pad1: 0 };
        gpu.dispatch(enc, "silu", &[&net, &act], bytemuck::bytes_of(&sp), ((s * c + 255) / 256, 1, 1));

        let adapt = gpu.buffer((s * 3 * c) as u64 * 4, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC);
        gemm(gpu, enc, &act, &adapt_w, &adapt_b, &gpu.dummy, &gpu.dummy, &adapt,
             s, 3 * c, c, c, c, 3 * c, 0, HAS_BIAS, ACT_NONE, 1.0);
        trace.push((format!("adapt{step}"), adapt.clone()));

        let shift = gpu.buffer((s * c) as u64 * 4, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC);
        let scale = gpu.buffer((s * c) as u64 * 4, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC);
        let gate = gpu.buffer((s * c) as u64 * 4, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC);
        let sp3 = SplitParams { s, c, pad0: 0, pad1: 0 };
        gpu.dispatch(enc, "split3", &[&adapt, &shift, &scale, &gate], bytemuck::bytes_of(&sp3),
                     ((s * c + 255) / 256, 1, 1));

        let adapt_norm = gpu.buffer((s * c) as u64 * 4, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC);
        layernorm(gpu, enc, &cam_norm, &gpu.dummy, &gpu.dummy, &adapt_norm, s, c, 1e-6, false);
        trace.push((format!("adapt_norm{step}"), adapt_norm.clone()));

        let modf = gpu.buffer((s * c) as u64 * 4, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC);
        let mp = ModParams { n: s * c, c, pad0: 0, pad1: 0 };
        gpu.dispatch(enc, "modulate", &[&adapt_norm, &shift, &scale, &gate, &cam_norm, &modf],
                     bytemuck::bytes_of(&mp), ((s * c + 255) / 256, 1, 1));
        trace.push((format!("mod{step}"), modf.clone()));

        let cfg = AttnCfg { l: s, c, heads: 16, hd: 128, eps: 1e-5, qk_norm: false, rope: false };
        let mut proc = modf;
        for b in &blocks {
            proc = block_forward(gpu, enc, &proc, b, cfg, &gpu.dummy, &gpu.dummy, &gpu.dummy, 1e-5);
        }
        trace.push((format!("proc{step}"), proc.clone()));

        let on = gpu.buffer((s * c) as u64 * 4, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC);
        layernorm(gpu, enc, &proc, &out_norm_w, &out_norm_b, &on, s, c, 1e-5, true);
        trace.push((format!("on{step}"), on.clone()));

        let h = gpu.buffer((s * 1024) as u64 * 4, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC);
        gemm(gpu, enc, &on, &fc1_w, &fc1_b, &gpu.dummy, &gpu.dummy, &h,
             s, 1024, c, c, c, 1024, 0, HAS_BIAS, ACT_GELU, 1.0);
        let delta = gpu.buffer((s * 9) as u64 * 4, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC);
        gemm(gpu, enc, &h, &fc2_w, &fc2_b, &gpu.dummy, &gpu.dummy, &delta,
             s, 9, 1024, 1024, 1024, 9, 0, HAS_BIAS, ACT_NONE, 1.0);
        trace.push((format!("delta{step}"), delta.clone()));

        curr = Some(match curr {
            None => delta,
            Some(prev) => {
                let next = gpu.buffer((s * 9) as u64 * 4, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC);
                add(gpu, enc, &prev, &delta, &next, (s * 9) as usize);
                next
            }
        });
        let _ = step;
    }
    curr.unwrap()
}
