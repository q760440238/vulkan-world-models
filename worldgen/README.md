---
license: apache-2.0
language:
  - zh
  - en
library_name: worldgen-vulkan
tags:
  - vulkan
  - amd
  - no-cuda
  - flux
  - worldgen
  - gaussian-splatting
  - 3d-generation
---

# WorldGen Vulkan

**中文** | [English](#english)

在 Windows + AMD RX 6900 XT 上，用 **Vulkan** 跑 WorldGen 的
text-to-360 / image-to-360 3DGS 流水线，不依赖 CUDA：

```text
prompt / image
  -> FLUX.1-dev / FLUX.1-Fill-dev（stable-diffusion.cpp, Vulkan）
  -> 1024x512 全景
  -> DA-2 depth（ncnn, Vulkan）
  -> Rust RGBD -> 3DGS PLY + mesh PLY
```

> 这不是官方 WorldGen 的逐字节复刻。默认 1024x512 路径已经跑通；
> 画质、1600x800、`use_sharp` / `inpaint_bg` 仍未完成。详见
> [docs/VALIDATION.md](docs/VALIDATION.md)。

## 性能（RX 6900 XT, 1024x512, 20 steps）

| 阶段 | 耗时 |
| --- | ---: |
| t2s 全流程 | 56.7 s |
| i2s 全流程 | 61.5 s |
| FLUX sampling | GPU 3D ~100% |
| VAE decode（tiling 512） | 1.35 s（未 tiling 9.0 s） |
| DA-2 Vulkan | 3.05 s（f32） |
| Rust RGBD→3DGS | 0.29 s |

DA-2 与 PyTorch 对拍（1024x512）：

```text
max abs   0.0161
mean abs  0.0027
cosine    0.9999994
```

## 环境

- Windows 11
- AMD RX 6900 XT 16GB（或其他支持 Vulkan 的 GPU）
- Vulkan 1.4 驱动
- Visual Studio 2022 + CMake/Ninja
- Rust stable
- Python 3.11+，仅用于编排/图像 I/O

不需要 CUDA，不需要 PyTorch 推理。

## 模型文件

把模型放到 `models/`（不要提交到 git）：

```text
models/
  flux-dev/
    flux1-dev-q4_k.gguf              # SHA256 962d536a...
    ae.safetensors
    clip_l.safetensors
    t5xxl_fp8_e4m3fn.safetensors
    worldgen_text2scene.safetensors
    worldgen_img2scene.safetensors
  flux-fill/
    flux-fill-official-q4.gguf       # 由官方 FLUX.1-Fill-dev 转换
  da2/
    da2_trace_1024.ncnn.param
    da2_trace_1024.ncnn.bin
```

FLUX.1-dev / FLUX.1-Fill-dev 权重受 Black Forest Labs 非商业许可证约束，
**本仓库不附带权重**。DA-2、WorldGen LoRA 请遵循各自上游许可证。

可用 `scripts/download_models.py` 查看下载地址和校验值；gated 模型需要
`HF_TOKEN`。

## 构建

```powershell
# 1. stable-diffusion.cpp（Vulkan）
scripts/build_sd.ps1

# 2. ncnn + DA-2 runner
scripts/build_da2.ps1

# 3. Rust 后处理
scripts/build_rust.ps1
```

## 运行

```powershell
# text -> 360 3DGS
py -3 worldgen\worldgen_t2s.py `
  --prompt "a cozy wooden bedroom with warm lamps" `
  --out output\bedroom --width 1024 --height 512 --steps 20

# image -> 360 3DGS
py -3 worldgen\worldgen_i2s.py `
  --image path\to\photo.jpg `
  --prompt "a lush rainforest with gorillas" `
  --out output\rainforest --width 1024 --height 512 --steps 20
```

输出：

```text
out/
  pano.png
  pano_fill.png        # i2s
  depth.f32
  scene/
    gaussians.ply      # 3DGS
    mesh.ply           # mesh
```

用任意 3DGS 查看器打开 `gaussians.ply`；mesh 可导入 Blender。

## DA-2 转换

DA-2 原始模型是 PyTorch。仓库提供：

- `patches/da2-export.patch`：去掉 `quantile`、分离 QKV，使 pnnx 可导出
- `scripts/export_da2.py`：trace + pnnx + ncnn
- `da2/main.cpp`：ncnn Vulkan runner

关键点：runner 必须用 `-DNCNN_VULKAN` 编译，并调用
`net.set_vulkan_device(0)`，否则 ncnn 会回退 CPU。

## 限制

- Fill 会重复物体，竖带边界可见，mesh 有拉伸
- 默认只验证 1024x512；1600x800 需重新 trace DA-2 或接受 resample
- `use_sharp=True`、`inpaint_bg=True` 未接
- 没有官方 WorldGen 全链路视觉对拍
- 不是官方实现，不代表 BFL / WorldGen 作者背书

## License

代码：Apache-2.0，见 [LICENSE](LICENSE) 与 [NOTICE](NOTICE)。
模型权重：遵循各自上游许可证，不随仓库分发。

---

## English

A Vulkan-only WorldGen pipeline for Windows + AMD RX 6900 XT. No CUDA,
no PyTorch inference:

```text
prompt/image
  -> FLUX.1-dev / FLUX.1-Fill-dev (stable-diffusion.cpp, Vulkan)
  -> 1024x512 panorama
  -> DA-2 depth (ncnn, Vulkan)
  -> Rust RGBD -> 3DGS/mesh PLY
```

This is a functional port of the default WorldGen path, **not** a
quality-equivalent reproduction. See [docs/VALIDATION.md](docs/VALIDATION.md).

### Quick start

```powershell
scripts/build_sd.ps1
scripts/build_da2.ps1
scripts/build_rust.ps1
py -3 worldgen\worldgen_t2s.py --prompt "a cozy bedroom" --out output\t2s
py -3 worldgen\worldgen_i2s.py --image path\to\photo.jpg --prompt "a rainforest" --out output\i2s
```

Weights are not bundled. FLUX.1-dev / FLUX.1-Fill-dev are subject to the
Black Forest Labs non-commercial license; DA-2 and the WorldGen LoRAs follow
their upstream licenses.
