# 3D Gaussian Splatting Vulkan / Vulkan 3DGS

**中文（主） | [English](#english)**

在 **AMD Radeon RX 6900 XT + Windows** 上，用 **Rust + wgpu / Vulkan** 渲染、
查看和训练 3D Gaussian Splatting，不依赖 CUDA。

> 工程移植与验证仓库，不是 VkSplat / INRIA 3DGS 官方仓库。

| 3DGS preview |
| --- |
| ![3DGS preview](vulkan-rs/preview.png) |

| 13-view gallery |
| --- |
| ![gallery](vulkan-rs/gallery_shot.png) |

## 内容

### 1. Rust/wgpu 3DGS renderer

- 解析 ml-sharp 的 14-float 紧凑 3DGS PLY
- 3D 协方差 → EWA 2D 投影 → billboard splat
- 直接输出 PNG / PPM
- 浏览器 gallery / WebGL viewer
- 13 视角 Chrome gallery

构建：

```powershell
cd vulkan-rs
cargo build --release
.\target\release\vulkan-splat.exe photo.ply 0.35 splat.png 0.10
```

参数：`argv[2]` yaw，`argv[3]` 输出图，`argv[4]` 最小 splat sigma。

### 2. VkSplat Vulkan 训练集成

- Vulkan compute tile rasterizer
- per-tile sort
- 可微 backward
- SH 优化与 densification
- 10,000 steps on AMD RX 6900 XT

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

### 3. Space / gallery

- HF Space: <https://huggingface.co/spaces/kylinfires/vulkan-world-models>
- 交互式 13 视角 3DGS gallery

## 目录

```text
vulkan-rs/            Rust wgpu 3DGS renderer + viewers + 结果图
VKSPLAT.md            VkSplat Vulkan 训练集成说明
VKSPLAT_RESULTS.md    10k 训练结果
```

完整说明见 GitHub：
<https://github.com/q760440238/vulkan-world-models/tree/main/gaussian-splatting>

## 不包含

- 75MB 训练 PLY
- 82MB WebGL HTML
- 原始 ml-sharp 权重
- 多视角原始数据（只保留结果与源码）

## 限制

- 3DGS PLY 只有 DC 颜色，没有高阶 SH。
- 单图 orbit 到较大角度会露洞，这是单视角数据限制。
- VkSplat 只在小型合成多视角数据上验证，不是 Mip-NeRF 360 基准。
- 原 3DGS.cpp viewer 的 present bug 未修，训练/渲染路径用 VkSplat/vulkan-rs 替代。

## 许可

- VkSplat：Apache-2.0，<https://github.com/harry7557558/vksplat>
- 3D Gaussian Splatting：INRIA / MPII 非商业研究许可
- SHARP / ml-sharp：Apple Machine Learning Research Model License
- 本仓库整体：CC BY-NC-SA 4.0（因为包含 3DGS / LingBot 衍生材料）

---

# English

Rust + wgpu/Vulkan 3D Gaussian Splatting on AMD Windows, without CUDA.

- **Renderer**: compact 14-float 3DGS PLY, EWA projection, billboard splats,
  PNG/PPM output, Chrome gallery and WebGL viewers.
- **VkSplat training**: Vulkan compute tile rasterizer, per-tile sort,
  differentiable backward, SH optimization, densification; 10k steps on an
  RX 6900 XT, 319,073 splats, 38.21 dB train / 20.65 dB held-out PSNR in 49.7 s.

HF Space gallery:
<https://huggingface.co/spaces/kylinfires/vulkan-world-models>

Not included: 75MB trained PLY, 82MB WebGL HTML, upstream weights, raw
multi-view datasets. The PLY has DC color only. The original 3DGS.cpp viewer
present bug remains unfixed; this repo uses the VkSplat/vulkan-rs paths.

Licenses: VkSplat Apache-2.0; INRIA/MPII 3DGS non-commercial; SHARP Apple model
license; repository as a whole CC BY-NC-SA 4.0.
