# HunyuanWorld-Mirror Vulkan 移植

## 结论

**单图 / 多图 / 视频推理已完整移植到 Vulkan，不再依赖 CUDA / PyTorch / gsplat。**

本机（Windows 11 + AMD RX 6900 XT 16GB）上，原生 Rust/wgpu 程序一次前向即可输出：

- 深度图 + 置信度
- 表面法线 + 置信度
- 相机参数 / 相机到世界矩阵 / 内参
- 世界坐标点云
- 3D Gaussian Splatting 参数（`gaussians.ply`）
- 可选 TSDF / Poisson mesh
- 内置 3DGS 渲染 + 环绕 turntable

518×518 单图端到端 **约 11.1 秒**；2 视图 **约 21.5 秒**；权重加载 2.5 秒。

```text
Vulkan device : AMD Radeon RX 6900 XT
backend       : wgpu / Vulkan 1.4
weights       : 1.26 B params, 5.05 GB f32, 1544 tensors
input         : S × 518 × 518
1 view        : 11.14 s
2 views       : 21.5 s
output        : depth, normals, camera, points.ply, gaussians.ply, mesh, render
```

## 模型结构

HunyuanWorld-Mirror 不是单纯的 ViT + DPT，实际是 **72 个 transformer block**：

| 部分 | 层数 | 说明 |
| --- | --- | --- |
| DINOv2 ViT-L/14 backbone | 24 | 1 cls + 4 register + 1369 patch token；无 RoPE、无 QK norm，LayerNorm eps=1e-6 |
| VGT frame blocks | 24 | 每帧内部 attention，QK norm + 2D RoPE，LayerNorm eps=1e-5 |
| VGT global blocks | 24 | 跨帧 attention，QK norm + 2D RoPE |
| Camera head | 4 × 4 | 2048 维、16 head、4 次迭代精化 |
| DPT heads | 4 | depth / pts / normal / gs，各自 256 通道 refinenet |

关键常量：

```text
patch grid        : 37 × 37 = 1369
DINOv2 tokens     : 1374 (1 cls + 4 reg + 1369 patch)
VGT tokens        : 1376 (1 cam + 4 reg + 1 pose + 1 ray + 1369 patch)
patch_start_idx   : 7
embed dim         : 1024
frame+global dim  : 2048
```

无条件推理时 pose/ray token 是零向量，但仍作为 token 参与 attention，不能省略。

## Vulkan 实现

代码：`port/hwm-vk/`，纯 Rust + WGSL，`wgpu` 走 Vulkan backend。

| 模块 | 文件 | 说明 |
| --- | --- | --- |
| GEMM | `shaders/gemm.wgsl` | 64×64 tile，16×16 workgroup，f32 accumulate；支持 bias / GELU / ReLU / SiLU / LayerScale / residual / 转置输出 |
| LayerNorm | `shaders/layernorm.wgsl` | 两遍均值/方差，支持 affine / 非 affine |
| Attention | `attn.rs` + `qkv_prepare.wgsl` + `softmax.wgsl` | QKV 拆分、QK LayerNorm、2D RoPE、V 转置、分块 QK^T → softmax → PV |
| Conv2d | `shaders/conv2d.wgsl` | 通用 direct conv，支持 stride/pad/act；按输出通道分块规避 65535 workgroup 限制 |
| ConvTranspose2d | `shaders/deconv.wgsl` | 4×4/2×2 stride=kernel 的 pixel-shuffle 形式 |
| Bilinear | `shaders/interpolate.wgsl` | align_corners=True |
| 位置编码 | `shaders/pos_add.wgsl` + `dpt.rs` | DPT 的 UV sinusoidal embedding，CPU f64 生成后上传 |
| DINOv2 patch embed | `im2col_patch.wgsl` + GEMM | 14×14 strided conv 转为 im2col GEMM |
| DPT | `dpt.rs` | 4 级 project/resize、refinenet1–4、output_conv、input_merger、gs_head |
| Camera | `camera.rs` | 4 步迭代、adaptive norm、4 个 2048 维 block、9 维参数 |
| 3DGS | `gs.rs` | 激活、depth→world、0.002 voxel 加权合并、PLY 导出 |

所有中间激活保持 f32；权重也是 f32。没有 CPU fallback 参与模型计算，CPU 只做最终的小规模参数激活、相机矩阵、点云/PLY 写出。

## 权重

官方 `model.safetensors` 5,053,546,720 字节，1544 个 tensor。

```text
ai-lab/hunyuanworld/weights/model.safetensors
ai-lab/hunyuanworld/weights/config.json
ai-lab/hunyuanworld/weights/weights.idx      # name abs_offset size ndim dims...
```

`weights.idx` 由 `tools/make_index.py` 从 safetensors header 生成；Rust 直接按绝对偏移读取，不需要 Python/JSON 解析。

重新下载/准备：

```powershell
cd D:\3DWord\itseez3d\ai-lab\hunyuanworld
py -3 -m pip install huggingface_hub
huggingface-cli download tencent/HunyuanWorld-Mirror --local-dir weights
py -3 tools\make_index.py
```

参考/分阶段测试还需要官方源码：

```powershell
git clone https://github.com/Tencent-Hunyuan/HunyuanWorld-Mirror src/HunyuanWorld-Mirror-main
```

## 数值验证

参考实现：官方 PyTorch 代码 + 官方权重，CPU f32，同一张 518×518 输入。

| 阶段 | 最大绝对误差 | cosine |
| --- | ---: | ---: |
| patch conv (im2col+GEMM) | 5.66e-6 | 1.000000 |
| DINOv2 24 blocks + final norm | 2.46e-5 | 1.000000 |
| VGT frame\_0 | 1.41e-5 | 1.000000 |
| VGT global\_23 | 6.10e-5 | 1.000000 |
| camera params | 1.12e-7 | 1.000000 |
| depth | 3.22e-6 | 0.99999988 |
| depth confidence | 1.47e-4 | 0.99999988 |
| pts3d | 8.34e-5 | 1.000000 |
| normals | 6.08e-5 | 1.000000 |
| gs raw 12 channels | 8.20e-5 | 0.99999994 |

3DGS 合并：

```text
reference splats : 41,386
vulkan splats    : 41,390
means NN         : mean 4.7e-7, p99 8.4e-7, max 4.4e-4
```

差 4 个 splat 的原因是上游 depth/camera 有 ~1e-5 级差异，0.002 的 voxel 边界上会有极少数点落到相邻 voxel；这是不同 GPU kernel 数值顺序导致的正常现象，不是模型缺失。

多视图（`Bright_Room`，2 视图，官方 `prepare_images_to_tensor(..., resize_strategy="pad")` 的同一输入）：

| 输出 | 最大绝对误差 | cosine |
| --- | ---: | ---: |
| camera params | 2.98e-7 | 1.00000000 |
| camera c2w | 1.75e-7 | 1.00000000 |
| depth | 5.91e-5 | 1.00000012 |
| pts3d | 1.24e-4 | 1.00000012 |
| normals | 5.00e-4 | 1.00000000 |

全局 3DGS voxel 合并把两视图的 2×268,324 个候选点合并为 **343,922 splats**。

## 性能（RX 6900 XT，518×518）

```text
1 view:
weights load      2.50 s
DINOv2 24 blocks  2.07 s
VGT 48 blocks     3.57 s
camera head       0.21 s
depth head        0.57 s
pts head          0.57 s
norm head         0.57 s
gs head           0.85 s
GS prune          0.02 s
total            11.14 s

2 views:
DINOv2 backbone   3.92 s
VGT backbone      9.20 s
camera head       0.20 s
4 DPT heads       5.09 s
GS prune          0.04 s
total            21.5 s

4 views:
DINOv2 backbone   7.58 s
VGT backbone     27.77 s
camera head       0.18 s
4 DPT heads      10.42 s
GS prune          0.09 s
total            49.1 s
```

显存峰值低于 16 GB（权重 5.05 GB + 位置编码表 0.15 GB + DPT 临时特征图）。

## 运行

```powershell
cd D:\3DWord\itseez3d\ai-lab\hunyuanworld\port\hwm-vk
cargo build --release --bin hwm-vk

# 真实图片（默认 pad 成 518×518，保留完整画面）
.\target\release\hwm-vk.exe `
  --image D:\3DWord\itseez3d\assets\rainforest\photo.jpg `
  --out ..\..\output\vulkan-photo

# 已预处理好的 [3,518,518] f32
.\target\release\hwm-vk.exe --raw testdata\model\input.f32 --out ..\..\output\vulkan-test

# 多图
.\target\release\hwm-vk.exe `
  --images .\frames --max-views 8 `
  --mesh tsdf --mesh-render --turntable 8 `
  --out ..\..\output\vulkan-multi

# 视频（OpenCV 抽帧）
.\target\release\hwm-vk.exe `
  --video .\clip.mp4 --fps 1 --max-views 8 `
  --out ..\..\output\vulkan-video
```

或直接：

```powershell
.\ai-lab\hunyuanworld\port\hwm-vk\run.ps1 -Image .\assets\rainforest\photo.jpg
```

权重目录默认是 `../../weights`（相对 `port/hwm-vk`）。也可 `--weights <dir>` 指定。

## 输出

```text
out/
  input_0000.png / depth_0000.png / normals_0000.png     # 每视图
  depth_0000.f32 / depth_conf_0000.f32 / pts3d_0000.f32
  normals_0000.f32 / gs_raw12_0000.f32
  camera_params.f32 / camera_poses.f32 / camera_intrs.f32 # [S,...]
  points.ply              # 全视图合并，按 pts3d_conf 10% 分位过滤
  gaussians.ply           # 标准 3DGS 17 float 顶点，log scale / f_dc
  splats_*.f32            # 合并后的 means/quats/scales/opacities/sh
  render_0000.png         # 内置 3DGS 渲染
  turntable.png           # --turntable N
  mesh-tsdf.ply / mesh-tsdf.png
  mesh-poisson.ply / mesh-poisson.png
  result.json
```

单视图还会额外写一份不带后缀的 `depth.f32`、`input.png` 等，方便旧脚本。

`gaussians.ply` 可以直接给任意 3DGS viewer / gsplat / VkSplat 使用。内置 `render_0000.png` 使用 **isotropic Gaussian** 的软件渲染，速度快、稳定；需要各向异性 gsplat 画质时请用外部 gsplat/VkSplat。

## Mesh

`--mesh tsdf|poisson|both` 会调用 `scripts/make_mesh.py`（Open3D）：

- **TSDF**：`UniformTSDFVolume`，voxel ≈ bbox/256，sdf_trunc=4×voxel；多视图融合效果最好。
- **Poisson**：从 `points.ply` + 法线估计重建，单视图也能出完整表面，但可能出现过平滑/膨胀。

```powershell
py -3 -m pip install open3d
.\target\release\hwm-vk.exe --images frames --mesh both --mesh-render --out out
```

`--mesh-render` 使用 Open3D 的 Vulkan offscreen renderer 输出 `mesh-*.png`。

## 限制

- 默认最多 8 视图（`--max-views`）；global attention 对视图数是 O(S²)，16GB 显存下 8 视图是安全上限。
- 输入会被 resize/pad 到 518×518。官方 DPT 的 `_apply_pos_embed` 在非方形特征图上存在维度错位，所以方形输入是安全路径。
- 模型本身输出点云/深度/3DGS，不输出三角网格；mesh 是 Open3D 后处理，不是模型原生输出。
- 只移植推理。训练、3DGS 后优化、gsplat CUDA rasterizer 不在范围内。
- 内置 3DGS renderer 是 isotropic 软件实现，用于快速检查和 turntable；不追求 gsplat 的各向异性画质。

## 复现验证

```powershell
# CPU 参考（单图）
py -3 ai-lab\hunyuanworld\ref\run_ref.py

# CPU 参考（2 视图，Bright_Room）
py -3 ai-lab\hunyuanworld\ref\make_multi_ref.py

# 分阶段测试
cd ai-lab\hunyuanworld\port\hwm-vk
cargo run --release --bin patch_test
cargo run --release --bin dino_full_test
cargo run --release --bin vgt_full_test
cargo run --release --bin camera_test
cargo run --release --bin dpt_test depth
cargo run --release --bin dpt_test pts
cargo run --release --bin dpt_test norm
cargo run --release --bin dpt_test gs
cargo run --release --bin gs_test
```

测试数据在 `port/hwm-vk/testdata/`，由 `ref/make_stage_data.py` 等脚本生成。多视图验证数据在 `ref/multi/`，用 `--raw-dir` 跑同一输入。

## 开源打包

可公开发布的独立仓库已经准备好：

```text
D:\3DWord\hunyuanworld-vulkan\
  src/                     # 生产代码（不含开发期 test binaries）
  docs/                    # depth / normals / points / 3DGS / turntable / mesh 预览
  scripts/download_weights.py
  scripts/extract_frames.py
  scripts/make_mesh.py
  scripts/render_mesh.py
  scripts/publish_github.ps1
  scripts/publish_hf.py
  requirements.txt         # huggingface_hub + opencv + open3d
  LICENSE                  # Apache-2.0，仅覆盖本仓库 Rust/WGSL 代码
  LICENSE-TENCENT-HUNYUANWORLD.txt
  NOTICE / NOTICE-TENCENT.txt
  MODEL_CARD.md            # Hugging Face 模型卡
  .github/workflows/build.yml
```

本地已 `git init` + commit，且确认只占 1.6MB（不含权重）。发布前需要：

1. GitHub：创建空仓库，把 `~/.ssh/id_ed25519_github.pub` 加到账号，或使用 HTTPS PAT；把 `YOUR_GITHUB_USER` 替换成实际用户名；运行 `scripts\publish_github.ps1 -RepoUrl ...`。
2. Hugging Face：设置 `HF_TOKEN`（write token），运行 `python scripts\publish_hf.py --repo <user>/hunyuanworld-vulkan`。

**不要上传 `weights/`**：官方权重受 Tencent Community License 的地域限制（不含 EU/UK/韩国），公开托管在全球可访问的 HF 上存在合规风险。仓库只放代码、模型卡和权重下载脚本。

