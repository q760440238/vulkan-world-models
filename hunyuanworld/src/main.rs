use bytemuck::{Pod, Zeroable};
use hwm_vk::attn::AttnCfg;
use hwm_vk::blocks::{block_forward, BlockWeights};
use hwm_vk::camera::camera_head;
use hwm_vk::dpt::{dpt_forward, pos_tables, DptWeights};
use hwm_vk::gs::{flatten, Splats};
use hwm_vk::gpu::{Buf, Gpu, WeightStore};
use hwm_vk::ops::{
    gemm, layernorm, ACT_NONE, HAS_BIAS,
};
use std::io::Write;
use std::path::{Path, PathBuf};

const HW: usize = 518 * 518;
const W: usize = 518;
const H: usize = 518;
const NPATCH: usize = 1369;
const L: usize = 1376;
const PATCH_START: usize = 7;
const C: usize = 1024;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct PatchParams { c: u32, h: u32, w: u32, gw: u32, n: u32 }
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct DinoParams { npatch: u32, c: u32, nreg: u32, pad: u32 }
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct SliceParams { rows: u32, cols: u32, start: u32, pad: u32 }
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct VgtParams { s: u32, l: u32, c: u32, npatch: u32, nreg: u32, start: u32, pad0: u32, pad1: u32 }
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct ConcatParams { rows: u32, c: u32, pad0: u32, pad1: u32 }

fn rf(p: &Path) -> Vec<f32> {
    let b = std::fs::read(p).unwrap_or_else(|e| panic!("read {p:?}: {e}"));
    bytemuck::cast_slice(&b).to_vec()
}

fn add_pipelines(gpu: &mut Gpu) {
    for (n, s) in [
        ("gemm", include_str!("shaders/gemm.wgsl")),
        ("layernorm", include_str!("shaders/layernorm.wgsl")),
        ("qkv_prepare", include_str!("shaders/qkv_prepare.wgsl")),
        ("softmax", include_str!("shaders/softmax.wgsl")),
        ("add", include_str!("shaders/add.wgsl")),
        ("relu", include_str!("shaders/relu.wgsl")),
        ("silu", include_str!("shaders/silu.wgsl")),
        ("split3", include_str!("shaders/split3.wgsl")),
        ("modulate", include_str!("shaders/modulate.wgsl")),
        ("gather_cam", include_str!("shaders/gather_cam.wgsl")),
        ("conv2d", include_str!("shaders/conv2d.wgsl")),
        ("deconv", include_str!("shaders/deconv.wgsl")),
        ("interpolate", include_str!("shaders/interpolate.wgsl")),
        ("pos_add", include_str!("shaders/pos_add.wgsl")),
        ("patch_extract", include_str!("shaders/patch_extract.wgsl")),
        ("im2col_patch", include_str!("shaders/im2col_patch.wgsl")),
        ("dino_prepare", include_str!("shaders/dino_prepare.wgsl")),
        ("slice_rows", include_str!("shaders/slice_rows.wgsl")),
        ("vgt_prepare", include_str!("shaders/vgt_prepare.wgsl")),
        ("concat2", include_str!("shaders/concat2.wgsl")),
    ] { gpu.add_pipeline(n, s); }
}

fn patch_embed(gpu: &Gpu, ws: &WeightStore, input: &Buf) -> Buf {
    let patches = gpu.buffer((NPATCH * 588 * 4) as u64, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC);
    let mut enc = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("patch") });
    let pp = PatchParams { c: 3, h: 518, w: 518, gw: 37, n: (NPATCH * 588) as u32 };
    gpu.dispatch(&mut enc, "im2col_patch", &[input, &patches], bytemuck::bytes_of(&pp),
                 (((NPATCH * 588) as u32 + 255) / 256, 1, 1));
    let w = ws.get("visual_geometry_transformer.patch_embed.patch_embed.proj.weight");
    let b = ws.get("visual_geometry_transformer.patch_embed.patch_embed.proj.bias");
    let conv = gpu.buffer((NPATCH * C * 4) as u64, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC);
    gemm(gpu, &mut enc, &patches, &w, &b, &gpu.dummy, &gpu.dummy, &conv,
         NPATCH as u32, C as u32, 588, 588, 588, C as u32, 0, HAS_BIAS, ACT_NONE, 1.0);
    gpu.finish(enc);

    let cls = ws.get("visual_geometry_transformer.patch_embed.cls_token");
    let pos = ws.get("visual_geometry_transformer.patch_embed.pos_embed");
    let reg = ws.get("visual_geometry_transformer.patch_embed.register_tokens");
    let tokens = gpu.buffer((1374 * C * 4) as u64, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC);
    let dp = DinoParams { npatch: NPATCH as u32, c: C as u32, nreg: 4, pad: 0 };
    let mut enc = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("dino_prep") });
    gpu.dispatch(&mut enc, "dino_prepare", &[&conv, &cls, &pos, &reg, &tokens],
                 bytemuck::bytes_of(&dp), (1374, 1, 1));
    gpu.finish(enc);
    tokens
}

fn dino_backbone(gpu: &Gpu, ws: &WeightStore, input: &Buf) -> Buf {
    let mut x = patch_embed(gpu, ws, input);
    let cfg = AttnCfg { l: 1374, c: C as u32, heads: 16, hd: 64, eps: 1e-5, qk_norm: false, rope: false };
    for i in 0..24 {
        let w = BlockWeights::load(ws, &format!("visual_geometry_transformer.patch_embed.blocks.{i}"));
        let mut enc = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("dino") });
        let y = block_forward(gpu, &mut enc, &x, &w, cfg, &gpu.dummy, &gpu.dummy, &gpu.dummy, 1e-6);
        gpu.finish(enc);
        x = y;
        if i % 6 == 5 { gpu.wait(); }
    }
    let nw = ws.get("visual_geometry_transformer.patch_embed.norm.weight");
    let nb = ws.get("visual_geometry_transformer.patch_embed.norm.bias");
    let xn = gpu.buffer((1374 * C * 4) as u64, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC);
    let out = gpu.buffer((NPATCH * C * 4) as u64, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC);
    let mut enc = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("norm") });
    layernorm(gpu, &mut enc, &x, &nw, &nb, &xn, 1374, C as u32, 1e-6, true);
    let sp = SliceParams { rows: NPATCH as u32, cols: C as u32, start: 5, pad: 0 };
    gpu.dispatch(&mut enc, "slice_rows", &[&xn, &out], bytemuck::bytes_of(&sp), (NPATCH as u32, 1, 1));
    gpu.finish(enc);
    out
}

fn dino_multi(gpu: &Gpu, ws: &WeightStore, inputs: &[Buf], s: u32) -> Buf {
    let all = gpu.buffer((s as usize * NPATCH * C * 4) as u64, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST);
    for v in 0..s {
        let patch = dino_backbone(gpu, ws, &inputs[v as usize]);
        let mut enc = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("dino_copy") });
        enc.copy_buffer_to_buffer(
            &patch.buffer, patch.offset,
            &all.buffer, all.offset + (v as usize * NPATCH * C * 4) as u64,
            (NPATCH * C * 4) as u64,
        );
        gpu.finish(enc);
    }
    all
}

fn rope_table(maxpos: usize) -> (Vec<f32>, Vec<f32>) {
    let mut cos = vec![0.0f32; maxpos * 32];
    let mut sin = vec![0.0f32; maxpos * 32];
    for p in 0..maxpos {
        for j in 0..16 {
            let inv = 1.0f64 / 100.0f64.powf((2 * j) as f64 / 32.0);
            let a = (p as f64) * inv;
            cos[p * 32 + j] = a.cos() as f32;
            cos[p * 32 + 16 + j] = a.cos() as f32;
            sin[p * 32 + j] = a.sin() as f32;
            sin[p * 32 + 16 + j] = a.sin() as f32;
        }
    }
    (cos, sin)
}

fn vgt_positions() -> Vec<u32> {
    let mut pos = vec![0u32; L * 2];
    for i in PATCH_START..L {
        let j = i - PATCH_START;
        pos[i * 2] = (j / 37 + 1) as u32;
        pos[i * 2 + 1] = (j % 37 + 1) as u32;
    }
    pos
}

fn vgt_backbone(gpu: &Gpu, ws: &WeightStore, patch: &Buf, s: u32) -> Vec<Buf> {
    let cam = ws.get("visual_geometry_transformer.cam_token");
    let reg = ws.get("visual_geometry_transformer.reg_token");
    let tokens = gpu.buffer((s as usize * L * C * 4) as u64, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC);
    let vp = VgtParams { s, l: L as u32, c: C as u32, npatch: NPATCH as u32, nreg: 4, start: PATCH_START as u32, pad0: 0, pad1: 0 };
    let mut enc = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("vgt_prep") });
    gpu.dispatch(&mut enc, "vgt_prepare", &[patch, &cam, &reg, &tokens], bytemuck::bytes_of(&vp), (s * L as u32, 1, 1));
    gpu.finish(enc);

    let posv = vgt_positions();
    let pos = gpu.upload(bytemuck::cast_slice(&posv));
    let mut posg_v = Vec::with_capacity(s as usize * L * 2);
    for _ in 0..s { posg_v.extend_from_slice(&posv); }
    let posg = gpu.upload(bytemuck::cast_slice(&posg_v));
    let (cosv, sinv) = rope_table(64);
    let cos = gpu.upload(bytemuck::cast_slice(&cosv));
    let sin = gpu.upload(bytemuck::cast_slice(&sinv));
    let frame_cfg = AttnCfg { l: L as u32, c: C as u32, heads: 16, hd: 64, eps: 1e-5, qk_norm: true, rope: true };
    let global_cfg = AttnCfg { l: s * L as u32, c: C as u32, heads: 16, hd: 64, eps: 1e-5, qk_norm: true, rope: true };
    let mut x = tokens;
    let mut outputs = Vec::new();
    for i in 0..24 {
        let fw = BlockWeights::load(ws, &format!("visual_geometry_transformer.frame_blocks.{i}"));
        let frame_all = gpu.buffer((s as usize * L * C * 4) as u64, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST);
        let mut enc = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("frames") });
        for v in 0..s {
            let xs = x.slice((v as usize * L * C * 4) as u64, (L * C * 4) as u64);
            let f = block_forward(gpu, &mut enc, &xs, &fw, frame_cfg, &pos, &cos, &sin, 1e-5);
            enc.copy_buffer_to_buffer(
                &f.buffer, f.offset,
                &frame_all.buffer, frame_all.offset + (v as usize * L * C * 4) as u64,
                (L * C * 4) as u64,
            );
        }
        gpu.finish(enc);
        let gw = BlockWeights::load(ws, &format!("visual_geometry_transformer.global_blocks.{i}"));
        let mut enc = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("global") });
        let g = block_forward(gpu, &mut enc, &frame_all, &gw, global_cfg, &posg, &cos, &sin, 1e-5);
        gpu.finish(enc);
        if [4usize, 11, 17, 23].contains(&i) {
            let combined = gpu.buffer((s as usize * L * 2 * C * 4) as u64, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC);
            let cp = ConcatParams { rows: s * L as u32, c: C as u32, pad0: 0, pad1: 0 };
            let mut enc = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("concat") });
            gpu.dispatch(&mut enc, "concat2", &[&frame_all, &g, &combined], bytemuck::bytes_of(&cp), (s * L as u32, 1, 1));
            gpu.finish(enc);
            outputs.push(combined);
        }
        x = g;
        if i % 6 == 5 { gpu.wait(); }
    }
    outputs
}

fn quat_to_rot(q: [f32; 4]) -> [[f32; 3]; 3] {
    let [i, j, k, r] = q;
    let two_s = 2.0 / (i * i + j * j + k * k + r * r + 1e-12);
    [
        [1.0 - two_s * (j * j + k * k), two_s * (i * j - k * r), two_s * (i * k + j * r)],
        [two_s * (i * j + k * r), 1.0 - two_s * (i * i + k * k), two_s * (j * k - i * r)],
        [two_s * (i * k - j * r), two_s * (j * k + i * r), 1.0 - two_s * (i * i + j * j)],
    ]
}

fn camera_math(raw: &[f32]) -> ([f32; 9], [f32; 16], [f32; 9]) {
    let mut p = [0.0f32; 9];
    p.copy_from_slice(&raw[0..9]);
    p[7] = p[7].max(0.0);
    p[8] = p[8].max(0.0);
    let r = quat_to_rot([p[3], p[4], p[5], p[6]]);
    let t = [p[0], p[1], p[2]];
    let mut c2w = [0.0f32; 16];
    for i in 0..3 {
        for j in 0..3 { c2w[i * 4 + j] = r[j][i]; }
        c2w[i * 4 + 3] = -(r[0][i] * t[0] + r[1][i] * t[1] + r[2][i] * t[2]);
    }
    c2w[15] = 1.0;
    let fy = H as f32 * 0.5 / (p[7] * 0.5).tan();
    let fx = W as f32 * 0.5 / (p[8] * 0.5).tan();
    let intr = [fx, 0.0, W as f32 * 0.5, 0.0, fy, H as f32 * 0.5, 0.0, 0.0, 1.0];
    (p, c2w, intr)
}

fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 8);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            _ => out.push(c),
        }
    }
    out
}

fn quantile(v: &mut [f32], q: f32) -> f32 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    if v.is_empty() { return 0.0; }
    let pos = q * (v.len() - 1) as f32;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    let f = pos - lo as f32;
    v[lo] * (1.0 - f) + v[hi] * f
}

fn write_points_ply(path: &Path, pts: &[[f32; 3]], cols: &[[u8; 3]]) {
    let mut s = String::new();
    s.push_str("ply\nformat binary_little_endian 1.0\n");
    s.push_str(&format!("element vertex {}\n", pts.len()));
    s.push_str("property float x\nproperty float y\nproperty float z\n");
    s.push_str("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n");
    let mut data = Vec::with_capacity(pts.len() * 15);
    for (p, c) in pts.iter().zip(cols.iter()) {
        for v in p { data.extend_from_slice(&v.to_le_bytes()); }
        data.extend_from_slice(c);
    }
    let mut f = std::fs::File::create(path).unwrap();
    f.write_all(s.as_bytes()).unwrap();
    f.write_all(&data).unwrap();
}

fn write_gaussians_ply(path: &Path, s: &Splats) {
    let mut smax: Vec<f32> = s.scales.iter().map(|v| v[0].max(v[1]).max(v[2])).collect();
    let threshold = quantile(&mut smax, 0.95);
    let mut data: Vec<f32> = Vec::new();
    let mut n = 0usize;
    for i in 0..s.means.len() {
        let mx = s.scales[i][0].max(s.scales[i][1]).max(s.scales[i][2]);
        if mx > threshold { continue; }
        n += 1;
        data.extend_from_slice(&s.means[i]);
        data.extend_from_slice(&[0.0, 0.0, 0.0]);
        data.extend_from_slice(&s.sh[i]);
        data.push(s.opacities[i]);
        data.push(s.scales[i][0].ln());
        data.push(s.scales[i][1].ln());
        data.push(s.scales[i][2].ln());
        data.extend_from_slice(&s.quats[i]);
    }
    let mut header = String::new();
    header.push_str("ply\nformat binary_little_endian 1.0\n");
    header.push_str(&format!("element vertex {n}\n"));
    for name in ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
                 "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"] {
        header.push_str(&format!("property float {name}\n"));
    }
    header.push_str("end_header\n");
    let mut f = std::fs::File::create(path).unwrap();
    f.write_all(header.as_bytes()).unwrap();
    f.write_all(bytemuck::cast_slice(&data)).unwrap();
}

fn load_image(path: &Path, mode: &str) -> Vec<f32> {
    let img = image::open(path).unwrap_or_else(|e| panic!("open {path:?}: {e}")).to_rgb8();
    let (w, h) = img.dimensions();
    let mut nw = W as u32;
    let mut nh = ((h as f32 * (nw as f32 / w as f32) / 14.0).round() as u32) * 14;
    if mode == "pad" {
        if h >= w { nh = H as u32; nw = ((w as f32 * (nh as f32 / h as f32) / 14.0).round() as u32) * 14; }
        else { nw = W as u32; nh = ((h as f32 * (nw as f32 / w as f32) / 14.0).round() as u32) * 14; }
    } else if nh > H as u32 { nh = H as u32; }
    let resized = image::imageops::resize(&img, nw, nh, image::imageops::FilterType::CatmullRom);
    let mut out = vec![1.0f32; 3 * HW];
    let ox = (W as u32).saturating_sub(nw) / 2;
    let oy = (H as u32).saturating_sub(nh) / 2;
    for y in 0..nh.min(H as u32) {
        for x in 0..nw.min(W as u32) {
            let p = resized.get_pixel(x, y);
            let xx = (x + ox) as usize;
            let yy = (y + oy) as usize;
            let i = yy * W + xx;
            out[i] = p[0] as f32 / 255.0;
            out[HW + i] = p[1] as f32 / 255.0;
            out[2 * HW + i] = p[2] as f32 / 255.0;
        }
    }
    out
}

fn list_images(dir: &Path) -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("read_dir {dir:?}: {e}"))
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.extension()
                .and_then(|s| s.to_str())
                .map(|s| matches!(s.to_ascii_lowercase().as_str(), "png" | "jpg" | "jpeg" | "webp" | "bmp"))
                .unwrap_or(false)
        })
        .collect();
    v.sort();
    v
}

fn list_raw(dir: &Path) -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("read_dir {dir:?}: {e}"))
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("f32"))
        .collect();
    v.sort();
    v
}

fn find_python() -> String {
    for c in ["py", "python"] {
        if std::process::Command::new(c).arg("--version").output().is_ok() {
            return c.to_string();
        }
    }
    panic!("python not found; install Python for video/mesh helpers");
}

fn extract_video(video: &Path, frames: &Path, fps: f32, max_views: usize) -> Vec<PathBuf> {
    std::fs::create_dir_all(frames).unwrap();
    let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("scripts/extract_frames.py");
    let py = find_python();
    let mut cmd = std::process::Command::new(&py);
    if py == "py" {
        cmd.arg("-3");
    }
    let status = cmd
        .arg(&script)
        .arg(video)
        .arg(frames)
        .arg(fps.to_string())
        .arg(max_views.to_string())
        .status()
        .unwrap_or_else(|e| panic!("video extraction failed: {e}"));
    if !status.success() {
        panic!("video extraction failed (is opencv-python installed?)");
    }
    list_images(frames)
}

fn run_py(script_name: &str, args: &[String]) {
    let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("scripts").join(script_name);
    let py = find_python();
    let mut cmd = std::process::Command::new(&py);
    if py == "py" {
        cmd.arg("-3");
    }
    let status = cmd.arg(&script).args(args).status()
        .unwrap_or_else(|e| panic!("failed to run {script_name}: {e}"));
    if !status.success() {
        panic!("{script_name} failed");
    }
}

fn save_rgb(path: &Path, data: &[f32]) {    let mut b = vec![0u8; HW * 3];
    for p in 0..HW {
        b[p * 3] = (data[p] * 255.0) as u8;
        b[p * 3 + 1] = (data[HW + p] * 255.0) as u8;
        b[p * 3 + 2] = (data[2 * HW + p] * 255.0) as u8;
    }
    image::RgbImage::from_raw(W as u32, H as u32, b).unwrap().save(path).unwrap();
}

fn save_depth_png(path: &Path, depth: &[f32]) {
    let dmin = depth.iter().cloned().fold(f32::INFINITY, f32::min);
    let dmax = depth.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let b: Vec<u8> = depth.iter().map(|v| (((v - dmin) / (dmax - dmin + 1e-9)) * 255.0) as u8).collect();
    image::GrayImage::from_raw(W as u32, H as u32, b).unwrap().save(path).unwrap();
}

fn save_normal_png(path: &Path, normals: &[f32]) {
    let mut b = vec![0u8; HW * 3];
    for p in 0..HW {
        for c in 0..3 {
            b[p * 3 + c] = (((normals[c * HW + p] + 1.0) * 0.5).clamp(0.0, 1.0) * 255.0) as u8;
        }
    }
    image::RgbImage::from_raw(W as u32, H as u32, b).unwrap().save(path).unwrap();
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut image_paths: Vec<PathBuf> = Vec::new();
    let mut raw_paths: Vec<PathBuf> = Vec::new();
    let mut images_dir: Option<PathBuf> = None;
    let mut raw_dir: Option<PathBuf> = None;
    let mut video_path: Option<PathBuf> = None;
    let mut out_dir = PathBuf::from("out");
    let mut weights_dir = PathBuf::from("weights");
    let mut mode = "pad".to_string();
    let mut fps = 1.0f32;
    let mut max_views = 8usize;
    let mut do_render = true;
    let mut render_size = 1036usize;
    let mut turntable = 0usize;
    let mut mesh_method: Option<String> = None;
    let mut mesh_render = false;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--image" => { image_paths.push(PathBuf::from(&args[i + 1])); i += 2; }
            "--images" => { images_dir = Some(PathBuf::from(&args[i + 1])); i += 2; }
            "--raw" => { raw_paths.push(PathBuf::from(&args[i + 1])); i += 2; }
            "--raw-dir" => { raw_dir = Some(PathBuf::from(&args[i + 1])); i += 2; }
            "--video" => { video_path = Some(PathBuf::from(&args[i + 1])); i += 2; }
            "--fps" => { fps = args[i + 1].parse().expect("fps"); i += 2; }
            "--max-views" => { max_views = args[i + 1].parse().expect("max-views"); i += 2; }
            "--out" => { out_dir = PathBuf::from(&args[i + 1]); i += 2; }
            "--weights" => { weights_dir = PathBuf::from(&args[i + 1]); i += 2; }
            "--mode" => { mode = args[i + 1].clone(); i += 2; }
            "--no-render" => { do_render = false; i += 1; }
            "--render-size" => { render_size = args[i + 1].parse().expect("render-size"); i += 2; }
            "--turntable" => { turntable = args[i + 1].parse().expect("turntable"); i += 2; }
            "--mesh" => { mesh_method = Some(args[i + 1].clone()); i += 2; }
            "--mesh-render" => { mesh_render = true; i += 1; }
            _ => panic!("unknown arg {}", args[i]),
        }
    }
    std::fs::create_dir_all(&out_dir).unwrap();

    if let Some(v) = &video_path {
        if !image_paths.is_empty() || !raw_paths.is_empty() || images_dir.is_some() || raw_dir.is_some() {
            panic!("--video cannot be combined with image/raw inputs");
        }
        let frames = out_dir.join("_frames");
        image_paths = extract_video(v, &frames, fps, max_views);
        println!("video frames: {}", image_paths.len());
    }
    if let Some(d) = &images_dir {
        let mut v = list_images(d);
        image_paths.append(&mut v);
    }
    if let Some(d) = &raw_dir {
        let mut v = list_raw(d);
        raw_paths.append(&mut v);
    }
    if !image_paths.is_empty() && !raw_paths.is_empty() {
        panic!("use either images or raw f32 files, not both");
    }
    let total = image_paths.len() + raw_paths.len();
    if total == 0 {
        panic!("provide --image, --images, --raw, --raw-dir, or --video");
    }
    if total > max_views {
        println!("limiting {} views to {}", total, max_views);
        image_paths.truncate(max_views);
        raw_paths.truncate(max_views);
    }
    let s = image_paths.len().max(raw_paths.len());
    let mut input_cpu: Vec<f32> = Vec::with_capacity(s * 3 * HW);
    if !raw_paths.is_empty() {
        for p in &raw_paths {
            let v = rf(p);
            if v.len() < 3 * HW { panic!("raw input too short: {} ({})", p.display(), v.len()); }
            input_cpu.extend_from_slice(&v[..3 * HW]);
        }
    } else {
        for p in &image_paths {
            input_cpu.extend_from_slice(&load_image(p, &mode));
        }
    }
    let s = s as u32;
    if input_cpu.len() != s as usize * 3 * HW { panic!("input size mismatch"); }
    println!("views: {s}");

    let t_all = std::time::Instant::now();
    let mut gpu = Gpu::new();
    add_pipelines(&mut gpu);
    let ws = WeightStore::open(&weights_dir.join("weights.idx"), &weights_dir.join("model.safetensors"), &gpu, 512 * 1024 * 1024);
    let input_views: Vec<Buf> = (0..s as usize)
        .map(|v| gpu.upload(bytemuck::cast_slice(&input_cpu[v * 3 * HW..(v + 1) * 3 * HW])))
        .collect();

    let t0 = std::time::Instant::now();
    let patch = dino_multi(&gpu, &ws, &input_views, s);
    println!("DINOv2 backbone: {:.2}s", t0.elapsed().as_secs_f64());
    let t0 = std::time::Instant::now();
    let tokens = vgt_backbone(&gpu, &ws, &patch, s);
    println!("VGT backbone: {:.2}s", t0.elapsed().as_secs_f64());

    let t0 = std::time::Instant::now();
    let mut enc = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("cam") });
    let cam_raw = camera_head(&gpu, &mut enc, &tokens[3], s, L as u32, &ws);
    gpu.finish(enc);
    let camv = gpu.read_f32(&cam_raw);
    let mut cam_params = Vec::new();
    let mut c2ws = Vec::new();
    let mut intrs = Vec::new();
    for v in 0..s as usize {
        let (p, c, k) = camera_math(&camv[v * 9..v * 9 + 9]);
        cam_params.push(p);
        c2ws.push(c);
        intrs.push(k);
    }
    println!("camera head: {:.2}s", t0.elapsed().as_secs_f64());

    let pos = pos_tables(&gpu);
    let t0 = std::time::Instant::now();
    let run_head = |prefix: &str, out_dim: u32, is_gs: bool| -> (Vec<Vec<f32>>, Option<Vec<Vec<f32>>>) {
        let w = DptWeights::load(&ws, prefix, is_gs);
        let mut raws = Vec::new();
        let mut extras = Vec::new();
        for v in 0..s as usize {
            let toks: Vec<Buf> = (0..4)
                .map(|i| tokens[i].slice((v * L * 2 * C * 4) as u64, (L * 2 * C * 4) as u64))
                .collect();
            let img = &input_views[v];
            let mut enc = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(prefix) });
            let mut trace = Vec::new();
            let (raw, extra) = dpt_forward(&gpu, &mut enc, &toks, img, &w, &pos, 1, out_dim, is_gs, &mut trace);
            gpu.finish(enc);
            raws.push(gpu.read_f32(&raw));
            if let Some((_feat, gs_raw)) = extra {
                extras.push(gpu.read_f32(&gs_raw));
            }
        }
        (raws, if is_gs { Some(extras) } else { None })
    };
    let (depth_raws, _) = run_head("depth_head", 2, false);
    println!("depth head: {:.2}s", t0.elapsed().as_secs_f64());
    let t0 = std::time::Instant::now();
    let (pts_raws, _) = run_head("pts_head", 4, false);
    println!("pts head: {:.2}s", t0.elapsed().as_secs_f64());
    let t0 = std::time::Instant::now();
    let (norm_raws, _) = run_head("norm_head", 4, false);
    println!("norm head: {:.2}s", t0.elapsed().as_secs_f64());
    let t0 = std::time::Instant::now();
    let (gs_raws, gs_extras) = run_head("gs_head", 2, true);
    println!("gs head: {:.2}s", t0.elapsed().as_secs_f64());
    let gs12s = gs_extras.expect("gs params");

    let mut depths = Vec::new();
    let mut depth_confs = Vec::new();
    let mut pts_all = Vec::new();
    let mut pts_confs = Vec::new();
    let mut normals_all = Vec::new();
    let mut norm_confs = Vec::new();
    let mut gs_depths = Vec::new();
    let mut gs_confs = Vec::new();
    for v in 0..s as usize {
        let depth_raw = &depth_raws[v];
        let mut depth = vec![0.0f32; HW];
        let mut depth_conf = vec![0.0f32; HW];
        for p in 0..HW {
            depth[p] = depth_raw[p].exp();
            depth_conf[p] = 1.0 + depth_raw[HW + p].exp();
        }
        let mut pts = vec![0.0f32; HW * 3];
        let mut pts_conf = vec![0.0f32; HW];
        for p in 0..HW * 3 {
            let x = pts_raws[v][p];
            pts[p] = x.signum() * x.abs().exp_m1();
        }
        for p in 0..HW { pts_conf[p] = 1.0 + pts_raws[v][3 * HW + p].exp(); }
        let mut normals = vec![0.0f32; HW * 3];
        let mut norm_conf = vec![0.0f32; HW];
        for p in 0..HW {
            let x = norm_raws[v][p];
            let y = norm_raws[v][HW + p];
            let z = norm_raws[v][2 * HW + p];
            let n = (x * x + y * y + z * z).sqrt().max(1e-12);
            normals[p] = x / n;
            normals[HW + p] = y / n;
            normals[2 * HW + p] = z / n;
            norm_conf[p] = 1.0 + norm_raws[v][3 * HW + p].exp();
        }
        let mut gs_depth = vec![0.0f32; HW];
        let mut gs_conf = vec![0.0f32; HW];
        for p in 0..HW {
            gs_depth[p] = gs_raws[v][p].exp();
            gs_conf[p] = 1.0 + gs_raws[v][HW + p].exp();
        }
        depths.push(depth);
        depth_confs.push(depth_conf);
        pts_all.push(pts);
        pts_confs.push(pts_conf);
        normals_all.push(normals);
        norm_confs.push(norm_conf);
        gs_depths.push(gs_depth);
        gs_confs.push(gs_conf);
    }

    let t0 = std::time::Instant::now();
    let mut views = Vec::new();
    for v in 0..s as usize {
        views.push(hwm_vk::gs::SplatView {
            raw: &gs12s[v],
            depth: &gs_depths[v],
            c2w: &c2ws[v],
            intr: &intrs[v],
            image: &input_cpu[v * 3 * HW..(v + 1) * 3 * HW],
        });
    }
    let splats = hwm_vk::gs::build_splats_multi(&views, H, W, 0.002);
    println!("GS prune: {:.2}s, {} splats", t0.elapsed().as_secs_f64(), splats.means.len());

    // Per-view outputs.
    for v in 0..s as usize {
        let tag = format!("{v:04}");
        save_depth_png(&out_dir.join(format!("depth_{tag}.png")), &depths[v]);
        save_normal_png(&out_dir.join(format!("normals_{tag}.png")), &normals_all[v]);
        save_rgb(&out_dir.join(format!("input_{tag}.png")), &input_cpu[v * 3 * HW..(v + 1) * 3 * HW]);
        std::fs::write(out_dir.join(format!("depth_{tag}.f32")), bytemuck::cast_slice(&depths[v])).unwrap();
        std::fs::write(out_dir.join(format!("depth_conf_{tag}.f32")), bytemuck::cast_slice(&depth_confs[v])).unwrap();
        std::fs::write(out_dir.join(format!("pts3d_{tag}.f32")), bytemuck::cast_slice(&pts_all[v])).unwrap();
        std::fs::write(out_dir.join(format!("pts3d_conf_{tag}.f32")), bytemuck::cast_slice(&pts_confs[v])).unwrap();
        std::fs::write(out_dir.join(format!("normals_{tag}.f32")), bytemuck::cast_slice(&normals_all[v])).unwrap();
        std::fs::write(out_dir.join(format!("normals_conf_{tag}.f32")), bytemuck::cast_slice(&norm_confs[v])).unwrap();
        std::fs::write(out_dir.join(format!("gs_raw12_{tag}.f32")), bytemuck::cast_slice(&gs12s[v])).unwrap();
        std::fs::write(out_dir.join(format!("gs_depth_{tag}.f32")), bytemuck::cast_slice(&gs_depths[v])).unwrap();
        std::fs::write(out_dir.join(format!("gs_depth_conf_{tag}.f32")), bytemuck::cast_slice(&gs_confs[v])).unwrap();
    }
    if s == 1 {
        std::fs::write(out_dir.join("depth.f32"), bytemuck::cast_slice(&depths[0])).unwrap();
        std::fs::write(out_dir.join("depth_conf.f32"), bytemuck::cast_slice(&depth_confs[0])).unwrap();
        std::fs::write(out_dir.join("pts3d.f32"), bytemuck::cast_slice(&pts_all[0])).unwrap();
        std::fs::write(out_dir.join("pts3d_conf.f32"), bytemuck::cast_slice(&pts_confs[0])).unwrap();
        std::fs::write(out_dir.join("normals.f32"), bytemuck::cast_slice(&normals_all[0])).unwrap();
        std::fs::write(out_dir.join("normals_conf.f32"), bytemuck::cast_slice(&norm_confs[0])).unwrap();
        std::fs::write(out_dir.join("gs_raw12.f32"), bytemuck::cast_slice(&gs12s[0])).unwrap();
        std::fs::write(out_dir.join("input.f32"), bytemuck::cast_slice(&input_cpu)).unwrap();
        save_depth_png(&out_dir.join("depth.png"), &depths[0]);
        save_normal_png(&out_dir.join("normals.png"), &normals_all[0]);
        save_rgb(&out_dir.join("input.png"), &input_cpu);
    }

    let cam_flat: Vec<f32> = cam_params.iter().flatten().copied().collect();
    let c2w_flat: Vec<f32> = c2ws.iter().flatten().copied().collect();
    let intr_flat: Vec<f32> = intrs.iter().flatten().copied().collect();
    std::fs::write(out_dir.join("camera_params.f32"), bytemuck::cast_slice(&cam_flat)).unwrap();
    std::fs::write(out_dir.join("camera_poses.f32"), bytemuck::cast_slice(&c2w_flat)).unwrap();
    std::fs::write(out_dir.join("camera_intrs.f32"), bytemuck::cast_slice(&intr_flat)).unwrap();
    let (m, q, sc, op, sh) = flatten(&splats);
    std::fs::write(out_dir.join("splats_means.f32"), bytemuck::cast_slice(&m)).unwrap();
    std::fs::write(out_dir.join("splats_quats.f32"), bytemuck::cast_slice(&q)).unwrap();
    std::fs::write(out_dir.join("splats_scales.f32"), bytemuck::cast_slice(&sc)).unwrap();
    std::fs::write(out_dir.join("splats_opacities.f32"), bytemuck::cast_slice(&op)).unwrap();
    std::fs::write(out_dir.join("splats_sh.f32"), bytemuck::cast_slice(&sh)).unwrap();

    // Combined point cloud, confidence-filtered at the 10th percentile.
    let mut conf_sorted: Vec<f32> = pts_confs.iter().flatten().copied().collect();
    let cthr = quantile(&mut conf_sorted, 0.10);
    let mut pp = Vec::new();
    let mut pc = Vec::new();
    for v in 0..s as usize {
        for p in 0..HW {
            let x = pts_all[v][p];
            let y = pts_all[v][HW + p];
            let z = pts_all[v][2 * HW + p];
            if !(x.is_finite() && y.is_finite() && z.is_finite()) || pts_confs[v][p] < cthr { continue; }
            pp.push([x, y, z]);
            let img = &input_cpu[v * 3 * HW..(v + 1) * 3 * HW];
            pc.push([
                (img[p] * 255.0) as u8,
                (img[HW + p] * 255.0) as u8,
                (img[2 * HW + p] * 255.0) as u8,
            ]);
        }
    }
    write_points_ply(&out_dir.join("points.ply"), &pp, &pc);
    write_gaussians_ply(&out_dir.join("gaussians.ply"), &splats);

    if mesh_method.is_some() {
        // make_mesh.py needs the cameras; write a preliminary metadata file
        // before the optional mesh step, then overwrite it with the full result.
        let mut cams = String::from("[");
        for v in 0..s as usize {
            if v > 0 { cams.push(','); }
            cams.push_str(&format!(
                "{{\"params\":{:?},\"c2w\":{:?},\"intrinsics\":{:?}}}",
                cam_params[v], c2ws[v], intrs[v]
            ));
        }
        cams.push(']');
        std::fs::write(out_dir.join("result.json"),
            format!("{{\"views\":{},\"cameras\":{}}}\n", s, cams)).unwrap();
    }

    if do_render {
        let t0 = std::time::Instant::now();
        for v in 0..s as usize {
            let px = hwm_vk::render::render_view(&splats, &c2ws[v], &intrs[v], render_size, 1.0);
            image::RgbImage::from_raw(render_size as u32, render_size as u32, px).unwrap()
                .save(out_dir.join(format!("render_{v:04}.png"))).unwrap();
        }
        println!("3DGS render: {:.2}s", t0.elapsed().as_secs_f64());
    }
    if turntable > 0 {
        let n = turntable;
        let tile = 512usize;
        let cols = 4usize;
        let rows = (n + cols - 1) / cols;
        let mut sheet = vec![255u8; cols * tile * rows * tile * 3];
        for k in 0..n {
            let angle = std::f32::consts::TAU * k as f32 / n as f32;
            let cam = hwm_vk::render::orbit_camera(&splats, &c2ws[0], angle);
            let px = hwm_vk::render::render_view(&splats, &cam, &intrs[0], tile, 1.0);
            image::RgbImage::from_raw(tile as u32, tile as u32, px.clone()).unwrap()
                .save(out_dir.join(format!("turntable_{k:03}.png"))).unwrap();
            let ox = (k % cols) * tile;
            let oy = (k / cols) * tile;
            for y in 0..tile {
                let src = y * tile * 3;
                let dst = ((oy + y) * cols * tile + ox) * 3;
                sheet[dst..dst + tile * 3].copy_from_slice(&px[src..src + tile * 3]);
            }
        }
        image::RgbImage::from_raw((cols * tile) as u32, (rows * tile) as u32, sheet).unwrap()
            .save(out_dir.join("turntable.png")).unwrap();
        println!("turntable: {n} frames");
    }
    if let Some(method) = &mesh_method {
        run_py("make_mesh.py", &[
            "--dir".to_string(), out_dir.display().to_string(),
            "--method".to_string(), method.clone(),
        ]);
        if mesh_render {
            let names: Vec<&str> = match method.as_str() {
                "both" => vec!["tsdf", "poisson"],
                m => vec![m],
            };
            for name in names {
                let mesh = out_dir.join(format!("mesh-{name}.ply"));
                if mesh.exists() {
                    run_py("render_mesh.py", &[
                        "--mesh".to_string(), mesh.display().to_string(),
                        "--out".to_string(), out_dir.join(format!("mesh-{name}.png")).display().to_string(),
                    ]);
                }
            }
        }
    }

    let mut cams_json = String::from("[");
    for v in 0..s as usize {
        if v > 0 { cams_json.push(','); }
        cams_json.push_str(&format!(
            "{{\"params\":{:?},\"c2w\":{:?},\"intrinsics\":{:?}}}",
            cam_params[v], c2ws[v], intrs[v]
        ));
    }
    cams_json.push(']');
    let inputs_json: Vec<String> = image_paths.iter().map(|p| format!("\"{}\"", json_escape(&p.display().to_string()))).collect();
    let json = format!(
        "{{\n  \"inputs\": [{}],\n  \"views\": {},\n  \"width\": {},\n  \"height\": {},\n  \"points\": {},\n  \"splats\": {},\n  \"cameras\": {},\n  \"total_seconds\": {:.3}\n}}\n",
        inputs_json.join(","),
        s, W, H, pp.len(), splats.means.len(), cams_json,
        t_all.elapsed().as_secs_f64()
    );
    std::fs::write(out_dir.join("result.json"), json).unwrap();
    println!("wrote {} ({} views, points {} / splats {}) in {:.2}s",
             out_dir.display(), s, pp.len(), splats.means.len(), t_all.elapsed().as_secs_f64());
}




