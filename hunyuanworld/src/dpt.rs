use crate::conv::{conv2d, deconv, interpolate, pos_add};
use crate::gpu::{Buf, Gpu, WeightStore};
use crate::ops::{
    add, gemm_transpose, layernorm, patch_extract, relu, ACT_NONE, ACT_RELU,
};

pub struct PosTables {
    pub p256: Buf,
    pub p512: Buf,
    pub p1024: Buf,
    pub p_final: Buf,
}

pub struct ResUnit {
    pub c1w: Buf,
    pub c1b: Buf,
    pub c2w: Buf,
    pub c2b: Buf,
}

pub struct Fusion {
    pub res1: Option<ResUnit>,
    pub res2: ResUnit,
    pub out_w: Buf,
    pub out_b: Buf,
}

pub struct DptWeights {
    pub norm_w: Buf,
    pub norm_b: Buf,
    pub proj_w: [Buf; 4],
    pub proj_b: [Buf; 4],
    pub proj_c: [u32; 4],
    pub resize_w: [Buf; 4],
    pub resize_b: [Buf; 4],
    pub layer_rn: [Buf; 4],
    pub fusions: [Fusion; 4],
    pub out1_w: Buf,
    pub out1_b: Buf,
    pub out2_0_w: Buf,
    pub out2_0_b: Buf,
    pub out2_2_w: Buf,
    pub out2_2_b: Buf,
    pub input_merger: Option<(Buf, Buf)>,
    pub gs0_w: Option<Buf>,
    pub gs2_w: Option<Buf>,
    pub gs2_b: Option<Buf>,
}

fn g(ws: &WeightStore, prefix: &str, name: &str) -> Buf {
    ws.get(&format!("{prefix}.{name}"))
}

fn load_res_unit(ws: &WeightStore, prefix: &str) -> ResUnit {
    ResUnit {
        c1w: g(ws, prefix, "conv1.weight"),
        c1b: g(ws, prefix, "conv1.bias"),
        c2w: g(ws, prefix, "conv2.weight"),
        c2b: g(ws, prefix, "conv2.bias"),
    }
}

impl DptWeights {
    pub fn load(ws: &WeightStore, prefix: &str, is_gs: bool) -> Self {
        let proj_c = [256u32, 512, 1024, 1024];
        let proj_w = std::array::from_fn(|i| g(ws, prefix, &format!("projects.{i}.weight")));
        let proj_b = std::array::from_fn(|i| g(ws, prefix, &format!("projects.{i}.bias")));
        let mut resize_w = std::array::from_fn(|i| {
            let n = match i { 0 => "resize_layers.0.weight", 1 => "resize_layers.1.weight", _ => "resize_layers.3.weight" };
            g(ws, prefix, n)
        });
        let mut resize_b = std::array::from_fn(|i| {
            let n = match i { 0 => "resize_layers.0.bias", 1 => "resize_layers.1.bias", _ => "resize_layers.3.bias" };
            g(ws, prefix, n)
        });
        // resize_layers.2 is Identity: give it harmless weight/bias slices.
        resize_w[2] = resize_w[3].clone();
        resize_b[2] = resize_b[3].clone();
        let layer_rn = std::array::from_fn(|i| g(ws, prefix, &format!("scratch.layer{}_rn.weight", i + 1)));
        let fusions = std::array::from_fn(|i| {
            let p = format!("{prefix}.scratch.refinenet{}", i + 1);
            let res1 = if i < 3 { Some(load_res_unit(ws, &format!("{p}.resConfUnit1"))) } else { None };
            Fusion {
                res1,
                res2: load_res_unit(ws, &format!("{p}.resConfUnit2")),
                out_w: ws.get(&format!("{p}.out_conv.weight")),
                out_b: ws.get(&format!("{p}.out_conv.bias")),
            }
        });
        let input_merger = if is_gs {
            Some((g(ws, prefix, "input_merger.0.weight"), g(ws, prefix, "input_merger.0.bias")))
        } else { None };
        let gs0_w = if is_gs { Some(ws.get("gs_renderer.gs_head.0.weight")) } else { None };
        let gs2_w = if is_gs { Some(ws.get("gs_renderer.gs_head.2.weight")) } else { None };
        let gs2_b = if is_gs { Some(ws.get("gs_renderer.gs_head.2.bias")) } else { None };
        Self {
            norm_w: g(ws, prefix, "norm.weight"),
            norm_b: g(ws, prefix, "norm.bias"),
            proj_w, proj_b, proj_c, resize_w, resize_b, layer_rn, fusions,
            out1_w: g(ws, prefix, "scratch.output_conv1.weight"),
            out1_b: g(ws, prefix, "scratch.output_conv1.bias"),
            out2_0_w: g(ws, prefix, "scratch.output_conv2.0.weight"),
            out2_0_b: g(ws, prefix, "scratch.output_conv2.0.bias"),
            out2_2_w: g(ws, prefix, "scratch.output_conv2.2.weight"),
            out2_2_b: g(ws, prefix, "scratch.output_conv2.2.bias"),
            input_merger, gs0_w, gs2_w, gs2_b,
        }
    }
}

pub fn make_pos_table(h: u32, w: u32, c: u32, img_w: u32, img_h: u32) -> Vec<f32> {
    assert_eq!(h, w, "DPT positional embedding assumes square maps");
    let width = w as f64;
    let height = h as f64;
    let aspect = img_w as f64 / img_h as f64;
    let diag = (aspect * aspect + 1.0).sqrt();
    let span_x = aspect / diag;
    let span_y = 1.0 / diag;
    let left = -span_x * (width - 1.0) / width;
    let right = span_x * (width - 1.0) / width;
    let top = -span_y * (height - 1.0) / height;
    let bottom = span_y * (height - 1.0) / height;
    let mut xs = vec![0.0f64; w as usize];
    let mut ys = vec![0.0f64; h as usize];
    for i in 0..w as usize {
        xs[i] = left + (right - left) * (i as f64) / ((w - 1) as f64);
    }
    for i in 0..h as usize {
        ys[i] = top + (bottom - top) * (i as f64) / ((h - 1) as f64);
    }
    let q = (c / 4) as usize;
    let mut omega = vec![0.0f64; q];
    for d in 0..q {
        omega[d] = 1.0 / 100.0f64.powf((d as f64) / (c as f64 / 4.0));
    }
    let mut table = vec![0.0f32; (h * w * c) as usize];
    for yi in 0..h as usize {
        for xi in 0..w as usize {
            let u = xs[xi];
            let v = ys[yi];
            let base = (yi * w as usize + xi) * c as usize;
            for d in 0..q {
                table[base + d] = (0.1 * (u * omega[d]).sin()) as f32;
                table[base + q + d] = (0.1 * (u * omega[d]).cos()) as f32;
                table[base + 2 * q + d] = (0.1 * (v * omega[d]).sin()) as f32;
                table[base + 3 * q + d] = (0.1 * (v * omega[d]).cos()) as f32;
            }
        }
    }
    table
}

pub fn pos_tables(gpu: &Gpu) -> PosTables {
    let mk = |h: u32, c: u32| {
        let t = make_pos_table(h, h, c, 518, 518);
        gpu.upload(bytemuck::cast_slice(&t))
    };
    PosTables { p256: mk(37, 256), p512: mk(37, 512), p1024: mk(37, 1024), p_final: mk(518, 128) }
}

#[allow(clippy::too_many_arguments)]
fn conv3(
    gpu: &Gpu,
    enc: &mut wgpu::CommandEncoder,
    x: &Buf,
    w: &Buf,
    bias: &Buf,
    n: u32,
    cin: u32,
    h: u32,
    wd: u32,
    cout: u32,
    has_bias: bool,
    act: u32,
    stride: u32,
    pad: u32,
) -> Buf {
    let oh = (h + 2 * pad - 3) / stride + 1;
    let ow = (wd + 2 * pad - 3) / stride + 1;
    let y = gpu.buffer((n * cout * oh * ow) as u64 * 4, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC);
    conv2d(gpu, enc, x, w, bias, &y, n, cin, h, wd, cout, 3, 3, stride, pad, has_bias, act, oh, ow);
    y
}

#[allow(clippy::too_many_arguments)]
fn conv1(
    gpu: &Gpu,
    enc: &mut wgpu::CommandEncoder,
    x: &Buf,
    w: &Buf,
    bias: &Buf,
    n: u32,
    cin: u32,
    h: u32,
    wd: u32,
    cout: u32,
) -> Buf {
    let y = gpu.buffer((n * cout * h * wd) as u64 * 4, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC);
    conv2d(gpu, enc, x, w, bias, &y, n, cin, h, wd, cout, 1, 1, 1, 0, true, ACT_NONE, h, wd);
    y
}

#[allow(clippy::too_many_arguments)]
fn res_unit(
    gpu: &Gpu,
    enc: &mut wgpu::CommandEncoder,
    x: &Buf,
    w: &ResUnit,
    n: u32,
    c: u32,
    h: u32,
    wd: u32,
) -> Buf {
    let total = (n * c * h * wd) as usize;
    let r1 = gpu.buffer((total * 4) as u64, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC);
    relu(gpu, enc, x, &r1, total);
    let c1 = conv3(gpu, enc, &r1, &w.c1w, &w.c1b, n, c, h, wd, c, true, ACT_NONE, 1, 1);
    let r2 = gpu.buffer((total * 4) as u64, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC);
    relu(gpu, enc, &c1, &r2, total);
    let c2 = conv3(gpu, enc, &r2, &w.c2w, &w.c2b, n, c, h, wd, c, true, ACT_NONE, 1, 1);
    let out = gpu.buffer((total * 4) as u64, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC);
    // The official ResidualConvUnit uses nn.ReLU(inplace=True), so its residual
    // is the *rectified* input, not the original input.
    add(gpu, enc, &c2, &r1, &out, total);
    out
}

#[allow(clippy::too_many_arguments)]
fn fusion(
    gpu: &Gpu,
    enc: &mut wgpu::CommandEncoder,
    main: &Buf,
    skip: Option<&Buf>,
    w: &Fusion,
    n: u32,
    c: u32,
    h: u32,
    wd: u32,
    oh: u32,
    ow: u32,
) -> Buf {
    let mut out = main.clone();
    if let Some(s) = skip {
        let r = res_unit(gpu, enc, s, w.res1.as_ref().unwrap(), n, c, h, wd);
        let total = (n * c * h * wd) as usize;
        let tmp = gpu.buffer((total * 4) as u64, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC);
        add(gpu, enc, &out, &r, &tmp, total);
        out = tmp;
    }
    out = res_unit(gpu, enc, &out, &w.res2, n, c, h, wd);
    let total_out = (n * c * oh * ow) as usize;
    let up = gpu.buffer((total_out * 4) as u64, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC);
    interpolate(gpu, enc, &out, &up, n, c, h, wd, oh, ow);
    conv1(gpu, enc, &up, &w.out_w, &w.out_b, n, c, oh, ow, c)
}

#[allow(clippy::too_many_arguments)]
pub fn dpt_forward(
    gpu: &Gpu,
    enc: &mut wgpu::CommandEncoder,
    tokens: &[Buf],
    images: &Buf,
    w: &DptWeights,
    pos: &PosTables,
    s: u32,
    out_dim: u32,
    is_gs: bool,
    trace: &mut Vec<(String, Buf)>,
) -> (Buf, Option<(Buf, Buf)>) {
    let l = 1376u32;
    let c = 2048u32;
    let npatch = 1369u32;
    let patch_start = 7u32;
    let mut feats: Vec<Buf> = Vec::with_capacity(4);
    for i in 0..4 {
        let pat = gpu.buffer((s * npatch * c) as u64 * 4, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC);
        patch_extract(gpu, enc, &tokens[i], &pat, s, l, npatch, c, patch_start);
        let normed = gpu.buffer((s * npatch * c) as u64 * 4, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC);
        layernorm(gpu, enc, &pat, &w.norm_w, &w.norm_b, &normed, s * npatch, c, 1e-5, true);
        trace.push((format!("norm{i}"), normed.clone()));

        let cout = w.proj_c[i];
        let proj = gpu.buffer((s * cout * npatch) as u64 * 4, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC);
        gemm_transpose(gpu, enc, &normed, &w.proj_w[i], &w.proj_b[i], &proj,
                       s * npatch, cout, c, c, c, npatch, ACT_NONE);
        trace.push((format!("proj{i}"), proj.clone()));

        let pos_table = match i { 0 => &pos.p256, 1 => &pos.p512, _ => &pos.p1024 };
        let with_pos = gpu.buffer((s * cout * npatch) as u64 * 4, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC);
        pos_add(gpu, enc, &proj, pos_table, &with_pos, s, cout, 37, 37);
        trace.push((format!("pos{i}"), with_pos.clone()));

        let resized = match i {
            0 => {
                let y = gpu.buffer((s * 256 * 148 * 148) as u64 * 4, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC);
                deconv(gpu, enc, &with_pos, &w.resize_w[0], &w.resize_b[0], &y, s, 256, 256, 37, 37, 4, 4, true);
                y
            }
            1 => {
                let y = gpu.buffer((s * 512 * 74 * 74) as u64 * 4, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC);
                deconv(gpu, enc, &with_pos, &w.resize_w[1], &w.resize_b[1], &y, s, 512, 512, 37, 37, 2, 2, true);
                y
            }
            2 => with_pos,
            _ => {
                let y = gpu.buffer((s * 1024 * 19 * 19) as u64 * 4, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC);
                conv2d(gpu, enc, &with_pos, &w.resize_w[3], &w.resize_b[3], &y, s, 1024, 37, 37, 1024, 3, 3, 2, 1, true, ACT_NONE, 19, 19);
                y
            }
        };
        let (cin, hh, ww) = match i { 0 => (256, 148, 148), 1 => (512, 74, 74), 2 => (1024, 37, 37), _ => (1024, 19, 19) };
        let rn = conv3(gpu, enc, &resized, &w.layer_rn[i], &gpu.dummy, s, cin, hh, ww, 256, false, ACT_NONE, 1, 1);
        trace.push((format!("resize{i}"), resized.clone()));
        trace.push((format!("rn{i}"), rn.clone()));
        feats.push(rn);
    }

    let out4 = fusion(gpu, enc, &feats[3], None, &w.fusions[3], s, 256, 19, 19, 37, 37);
    let out3 = fusion(gpu, enc, &out4, Some(&feats[2]), &w.fusions[2], s, 256, 37, 37, 74, 74);
    let out2 = fusion(gpu, enc, &out3, Some(&feats[1]), &w.fusions[1], s, 256, 74, 74, 148, 148);
    let out1 = fusion(gpu, enc, &out2, Some(&feats[0]), &w.fusions[0], s, 256, 148, 148, 296, 296);
    trace.push(("refine4".to_string(), out4.clone()));
    trace.push(("refine3".to_string(), out3.clone()));
    trace.push(("refine2".to_string(), out2.clone()));
    trace.push(("refine1".to_string(), out1.clone()));

    let c1 = conv3(gpu, enc, &out1, &w.out1_w, &w.out1_b, s, 256, 296, 296, 128, true, ACT_NONE, 1, 1);
    trace.push(("out1".to_string(), c1.clone()));
    let c1u = gpu.buffer((s * 128 * 518 * 518) as u64 * 4, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC);
    interpolate(gpu, enc, &c1, &c1u, s, 128, 296, 296, 518, 518);
    let fused = gpu.buffer((s * 128 * 518 * 518) as u64 * 4, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC);
    pos_add(gpu, enc, &c1u, &pos.p_final, &fused, s, 128, 518, 518);
    trace.push(("fused".to_string(), fused.clone()));

    let mut final_feat = fused.clone();
    if is_gs {
        let (mw, mb) = w.input_merger.as_ref().unwrap();
        let img_feat = gpu.buffer((s * 128 * 518 * 518) as u64 * 4, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC);
        conv2d(gpu, enc, images, mw, mb, &img_feat, s, 3, 518, 518, 128, 7, 7, 1, 3, true, ACT_RELU, 518, 518);
        let added = gpu.buffer((s * 128 * 518 * 518) as u64 * 4, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC);
        add(gpu, enc, &fused, &img_feat, &added, (s * 128 * 518 * 518) as usize);
        final_feat = added;
    }

    let o0 = conv3(gpu, enc, &fused, &w.out2_0_w, &w.out2_0_b, s, 128, 518, 518, 32, true, ACT_RELU, 1, 1);
    let raw = gpu.buffer((s * out_dim * 518 * 518) as u64 * 4, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC);
    conv2d(gpu, enc, &o0, &w.out2_2_w, &w.out2_2_b, &raw, s, 32, 518, 518, out_dim, 1, 1, 1, 0, true, ACT_NONE, 518, 518);
    trace.push(("raw".to_string(), raw.clone()));

    if is_gs {
        let g0 = conv3(gpu, enc, &final_feat, w.gs0_w.as_ref().unwrap(), &gpu.dummy, s, 128, 518, 518, 256, false, ACT_RELU, 1, 1);
        let g2_w = w.gs2_w.as_ref().unwrap();
        let g2_b = w.gs2_b.as_ref().unwrap();
        let y = gpu.buffer((s * 12 * 518 * 518) as u64 * 4, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC);
        conv2d(gpu, enc, &g0, g2_w, g2_b, &y, s, 256, 518, 518, 12, 1, 1, 1, 0, true, ACT_NONE, 518, 518);
        trace.push(("gs_raw".to_string(), y.clone()));
        (raw, Some((final_feat, y)))
    } else {
        (raw, None)
    }
}
