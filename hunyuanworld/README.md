# HunyuanWorld-Mirror Vulkan 推理 / Vulkan inference

**中文（主） | [English](#english)**

用 **Rust + wgpu / Vulkan** 原生运行 **HunyuanWorld-Mirror**，支持
**单图 / 多图 / 视频**，不依赖 CUDA、PyTorch 或 gsplat。

单次前向输出：

- 深度 + 置信度
- 表面法线 + 置信度
- 相机参数 / c2w / 内参
- 世界坐标点云
- 3D Gaussian Splatting
- 可选 TSDF / Poisson mesh
- 内置 3DGS 渲染与 turntable

| 项目 | 结果 |
| --- | --- |
| 模型 | HunyuanWorld-Mirror, 1.26B params, 5.05GB f32 |
| 硬件 | AMD Radeon RX 6900 XT 16GB |
| 单图 518×518 | **11.1 s** |
| 2 视图 | **21.5 s** |
| 4 视图 | **49.1 s** |
| 后端 | Rust + wgpu / Vulkan |

![3DGS render](docs/render.png)

![turntable](docs/turntable.png)

![TSDF mesh](docs/mesh.png)

## 模型结构

不是单个 ViT，而是 **72 个 transformer block**：

| 部分 | 层数 | 说明 |
| --- | ---: | --- |
| DINOv2 ViT-L/14 | 24 | 1 cls + 4 register + 1369 patch；无 RoPE/QK-norm，LN eps 1e-6 |
| VGT frame blocks | 24 | 帧内 attention，QK norm + 2D RoPE |
| VGT global blocks | 24 | 跨帧 attention，QK norm + 2D RoPE |
| Camera head | 4 × 4 | 2048 维、16 head、4 次迭代 |
| DPT heads | 4 | depth / pts / normal / gs |

Token 布局：

```text
patch grid      37 × 37 = 1369
DINOv2          1 cls + 4 reg + 1369 = 1374
VGT             1 cam + 4 reg + 1 pose + 1 ray + 1369 = 1376
patch_start_idx 7
embed dim       1024
```

全部推理算子为 WGSL compute shader：GEMM、LayerNorm、QKV/QK-norm/2D RoPE、
attention、Conv2d/ConvTranspose2d、bilinear、DPT UV embedding、im2col、
camera head、3DGS activation/voxel merge。

## 数值验证

官方 PyTorch CPU f32 为参考：

| 输出 | 单图 max abs | 2 视图 max abs |
| --- | ---: | ---: |
| camera | 1.12e-7 | 2.98e-7 |
| depth | 3.22e-6 | 5.91e-5 |
| pts3d | 8.34e-5 | 1.24e-4 |
| normals | 6.08e-5 | 5.00e-4 |
| 3DGS raw | 8.20e-5 | — |

单图 3DGS：41,390 vs 41,386 splats；2 视图合并 343,922 splats。

## 运行

```powershell
git clone https://github.com/q760440238/vulkan-world-models.git
cd vulkan-world-models/hunyuanworld
py -3 -m pip install -r requirements.txt
py -3 scripts/download_weights.py
cargo build --release --bin hwm-vk

# 单图 + mesh + 3DGS render + turntable
.\target\release\hwm-vk.exe --image photo.jpg --mesh poisson --mesh-render --turntable 8 --out out

# 多图
.\target\release\hwm-vk.exe --images frames --max-views 8 --out out

# 视频
.\target\release\hwm-vk.exe --video clip.mp4 --fps 1 --max-views 8 --out out
```

## 不包含权重

本仓库只包含推理引擎、文档和预览图，**不包含 5.05GB 官方权重**。
权重从官方仓库获取：

<https://huggingface.co/tencent/HunyuanWorld-Mirror>

权重受 **Tencent HunyuanWorld-Mirror Community License** 约束（不含
EU / UK / 韩国），并有可接受使用条款。

## 限制

- 默认最多 8 视图；global attention 对视图数是 O(S²)。
- 输入 resize/pad 到 518×518。
- 模型输出点云/3DGS，mesh 是 Open3D 后处理。
- 内置 3DGS renderer 是 isotropic 软件实现；各向异性画质用外部 gsplat。
- 只做推理，不含训练或 gsplat CUDA rasterizer。

## 许可

- 本仓库 Rust/WGSL 原创代码：**Apache-2.0**
- HunyuanWorld-Mirror 模型/权重：**Tencent HunyuanWorld-Mirror Community License**
- 不包含 Tencent 权重或源码；与 Tencent 无隶属关系。

---

# English

Native **Rust + wgpu/Vulkan** inference for
[HunyuanWorld-Mirror](https://github.com/Tencent-Hunyuan/HunyuanWorld-Mirror),
with **single image / multi-view / video** input, no CUDA/PyTorch/gsplat.

Outputs: depth, normals, camera poses/intrinsics, point cloud, 3D Gaussian
Splatting, optional TSDF/Poisson mesh, built-in 3DGS render and turntable.

| Item | Result |
| --- | --- |
| Model | HunyuanWorld-Mirror, 1.26B params, 5.05GB f32 |
| 1 view 518×518 | 11.1 s |
| 2 views | 21.5 s |
| 4 views | 49.1 s |

Validation against official PyTorch CPU f32: camera 3e-7, depth 5.9e-5,
pts3d 1.2e-4, normals 5.0e-4 (2 views).

The engine implements 72 transformer blocks (24 DINOv2 + 24 frame + 24 global),
camera head, four DPT heads, and 3DGS in WGSL compute shaders.

Weights are **not included**; download from
[`tencent/HunyuanWorld-Mirror`](https://huggingface.co/tencent/HunyuanWorld-Mirror).
They are governed by the Tencent HunyuanWorld-Mirror Community License
(territory restrictions: not EU/UK/South Korea).

Source: <https://github.com/q760440238/vulkan-world-models/tree/main/hunyuanworld>
