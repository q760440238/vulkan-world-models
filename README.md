# Vulkan World Models / Vulkan 世界模型

**中文（主）** | [English](#english)

在 **AMD Radeon RX 6900 XT + Windows** 上，用 **Rust / wgpu / Vulkan** 跑世界模型和 3D Gaussian Splatting，不依赖 CUDA。

> 本项目是工程移植与验证仓库，不是 LingBot / VkSplat 官方仓库。

---

# 中文

## 项目简介

本仓库包含三条工作线：

1. **LingBot-World 1.3B causal-fast 推理移植**
   - Rust + wgpu/Vulkan
   - 30 层 DiT、UMT5-XXL 文本编码、Wan2.1 VAE
   - 4-step FlowUniPC 采样
   - 单帧 1560 token 路径已跑通

2. **3D Gaussian Splatting Vulkan 渲染器**
   - Rust + wgpu
   - 解析 14-float 紧凑 3DGS PLY
   - 3D 协方差 → EWA 2D 投影 → billboard splat
   - 直接输出 PNG
   - 浏览器 gallery / WebGL viewer

3. **VkSplat Vulkan 训练集成**
   - Vulkan compute tile rasterizer
   - per-tile sort
   - 可微 backward
   - SH 优化与 densification
   - 在 AMD RX 6900 XT 上实际训练 10k steps

## 结果

### LingBot 单帧推理

| 项目 | 结果 |
| --- | --- |
| 模型 | LingBot-World 2.0 1.3B causal-fast |
| 层数 | 30 |
| dim / FFN | 1536 / 8960 |
| 采样 | FlowUniPC, 4 steps |
| VAE | Wan2.1 VAE, z16 |
| 文本编码 | UMT5-XXL, CPU |
| 后端 | Rust + wgpu / Vulkan |
| 硬件 | AMD RX 6900 XT 16GB |

![LingBot frame](lingbot/first_frame_fix_final.png)

### 3DGS Vulkan 渲染

![3DGS preview](gaussian-splatting/vulkan-rs/preview.png)

13 视角 Chrome gallery：

![gallery](gaussian-splatting/vulkan-rs/gallery_shot.png)

### VkSplat 训练

14 视角合成 COLMAP 场景，10,000 steps：

| 指标 | 数值 |
| --- | --- |
| Splats | 319,073 |
| Train PSNR | 38.21 dB |
| Validation PSNR | 20.65 dB |
| 时间 | 49.7 s |
| VRAM | ~0.33 GiB |

Vulkan pass 实际执行：

```text
ComputeTileRanges                    10,000
SortRTS                              10,080
RasterizeForward                     10,000
RasterizeBackward                    10,000
FusedProjectionBackwardOptimizerStep 10,000
```

## 目录结构

```text
lingbot/
  FULL_FORWARD.md, PORTING.md, NUMERICS.md, ...
  wgpu-*/                  DiT / VAE / T5 的 Rust wgpu kernels
  glsl/                    shaders
  *.py                     CPU 驱动与参考脚本
  first_frame_fix_final.png

gaussian-splatting/
  vulkan-rs/               Rust wgpu 3DGS 渲染器 + 浏览器 viewer
  VKSPLAT.md               VkSplat Vulkan 训练集成说明
  VKSPLAT_RESULTS.md       10k 训练结果
```

## 快速开始

### 3DGS 渲染器

```powershell
cd gaussian-splatting/vulkan-rs
cargo build --release

.\target\release\vulkan-splat.exe photo.ply 0.0 out.png 0.10
```

`argv[2]` 是 yaw，`argv[3]` 是输出 PNG，`argv[4]` 是最小 splat sigma。

### LingBot wgpu kernels

每个 `lingbot/wgpu-*` 目录都是独立的 Rust crate，见各自的 `README.md` / `src/`。

```powershell
cd lingbot/wgpu/wgpu-dit-block
cargo build --release
```

LingBot 完整推理需要真实权重和 CPU driver 脚本，权重不包含在本仓库。

## 权重

**本仓库不包含模型权重。** 权重总量约 17.5GB（DiT 6.8GB + VAE 0.5GB + T5 10.6GB），另有 f32 分块约 10GB。

上游来源：

| 组件 | 上游 |
| --- | --- |
| 1.3B transformer | `Robbyant/lingbot-world-v2-1.3b-causal-fast` |
| VAE | `Robbyant/lingbot-world-v2-14b-causal-fast` 下的 `Wan2.1_VAE.pth` |
| T5 | 同一 14B repo 下的 `models_t5_umt5-xxl-enc-bf16.pth` |

如果要做“下载即跑”的镜像，建议只上传 **1.3B DiT + VAE（约 7.3GB）** 到 HuggingFace model repo；T5 可复用上游 `google/umt5-xxl`。

## 浏览器查看

- 13 视角 gallery：`gaussian-splatting/vulkan-rs/gaussian_gallery.html`
- 交互式中心点云：`gaussian-splatting/vulkan-rs/gaussian_viewer.html`
- 全量 1.18M WebGL splat viewer：`gaussian-splatting/vulkan-rs/gaussian_splat_full.html`
  - 约 82MB，未提交到 Git
- 在线 gallery：<https://huggingface.co/spaces/kylinfires/vulkan-world-models>

## 许可证与来源

- LingBot-World 上游代码/权重：**CC BY-NC-SA 4.0**
- VkSplat：Apache-2.0，<https://github.com/harry7557558/vksplat>
- SHARP / ml-sharp：Apple Machine Learning Research Model License
- 3D Gaussian Splatting：INRIA / MPII 非商业研究许可

由于包含 LingBot 衍生材料，**本仓库整体按 CC BY-NC-SA 4.0 发布，禁止商业使用**。第三方组件仍遵循各自许可证。

## 已知限制

- LingBot 是视频世界模型，不直接输出 mesh/depth。
- 单帧 4-step 已跑通；多帧 chunk 仍会退化。
- 3DGS 单图 orbit 到较大角度会露洞，这是单视角数据限制。
- 3DGS PLY 只有 DC 颜色，没有高阶 SH。
- VkSplat 只在小型合成多视角数据上验证，不是 Mip-NeRF 360 基准成绩。
- 不包含 18GB 权重、75MB 训练 PLY、82MB HTML。

---

# English

## Overview

Vulkan / wgpu experiments for running world models and 3D Gaussian Splatting on an **AMD Radeon RX 6900 XT under Windows**, without CUDA.

This is an engineering port and verification repository, not the official LingBot or VkSplat repository.

It contains:

1. **LingBot-World 1.3B causal-fast inference** ported to Rust + wgpu/Vulkan
   - 30-layer DiT, UMT5-XXL text encoder, Wan2.1 VAE
   - 4-step FlowUniPC
   - Single-frame 1560-token path working

2. **3D Gaussian Splatting renderer**
   - Rust + wgpu
   - Compact 14-float 3DGS PLY
   - 3D covariance → EWA projection → billboard splats
   - Direct PNG output
   - Chrome gallery and WebGL viewers

3. **VkSplat Vulkan training integration**
   - Vulkan compute tile rasterizer
   - Per-tile sort
   - Differentiable backward
   - SH optimization and densification
   - 10k-step run on AMD RX 6900 XT

## Results

### LingBot single-frame inference

| Item | Result |
| --- | --- |
| Model | LingBot-World 2.0 1.3B causal-fast |
| Layers | 30 |
| Dim / FFN | 1536 / 8960 |
| Sampler | FlowUniPC, 4 steps |
| VAE | Wan2.1 VAE, z16 |
| Text encoder | UMT5-XXL, CPU |
| Backend | Rust + wgpu / Vulkan |
| Hardware | AMD RX 6900 XT 16GB |

![LingBot frame](lingbot/first_frame_fix_final.png)

### 3DGS rendering

![3DGS preview](gaussian-splatting/vulkan-rs/preview.png)

![gallery](gaussian-splatting/vulkan-rs/gallery_shot.png)

### VkSplat training

| Metric | Value |
| --- | --- |
| Splats | 319,073 |
| Train PSNR | 38.21 dB |
| Validation PSNR | 20.65 dB |
| Time | 49.7 s |
| VRAM | ~0.33 GiB |

Measured Vulkan passes:

```text
ComputeTileRanges                    10,000
SortRTS                              10,080
RasterizeForward                     10,000
RasterizeBackward                    10,000
FusedProjectionBackwardOptimizerStep 10,000
```

## Build

```powershell
cd gaussian-splatting/vulkan-rs
cargo build --release

.\target\release\vulkan-splat.exe photo.ply 0.0 out.png 0.10
```

Each `lingbot/wgpu-*` directory is a standalone Rust crate.

## Weights

Model weights are **not included**. They are about 17.5GB total
(6.8GB DiT, 0.5GB VAE, 10.6GB T5), plus about 10GB of f32 block splits.

Upstream:

- `Robbyant/lingbot-world-v2-1.3b-causal-fast`
- `Robbyant/lingbot-world-v2-14b-causal-fast` (`Wan2.1_VAE.pth`, `models_t5_umt5-xxl-enc-bf16.pth`)

For a turnkey mirror, upload only **1.3B DiT + VAE (~7.3GB)** to a HuggingFace
model repo; reuse upstream `google/umt5-xxl` for T5.

## Viewers

- 13-view gallery: `gaussian-splatting/vulkan-rs/gaussian_gallery.html`
- Interactive center cloud: `gaussian-splatting/vulkan-rs/gaussian_viewer.html`
- Full 1.18M WebGL splat viewer: `gaussian-splatting/vulkan-rs/gaussian_splat_full.html`
  (about 82MB, not committed)
- Online gallery: <https://huggingface.co/spaces/kylinfires/vulkan-world-models>

## License and credits

- LingBot-World upstream code/weights: **CC BY-NC-SA 4.0**
- VkSplat: Apache-2.0, <https://github.com/harry7557558/vksplat>
- SHARP / ml-sharp: Apple Machine Learning Research Model License
- 3D Gaussian Splatting: INRIA / MPII non-commercial research license

Because this repository contains LingBot-derived material, the repository as a
whole is distributed under **CC BY-NC-SA 4.0**. Commercial use is not permitted.
Third-party components remain under their own licenses.

## Known limitations

- LingBot is a video world model, not a native mesh/depth generator.
- Single-frame 4-step inference works; multi-frame chunks still degrade.
- Single-image 3DGS shows holes at large orbit angles.
- The 3DGS PLY has DC color only, no higher-order SH.
- VkSplat was validated on a small synthetic multi-view set, not Mip-NeRF 360.
- 18GB weights, the 75MB trained PLY, and the 82MB HTML viewer are excluded.
