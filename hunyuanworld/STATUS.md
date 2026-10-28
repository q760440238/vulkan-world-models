# HunyuanWorld-Mirror

- 状态：**VULKAN 单图 / 多图 / 视频推理完成**
- 来源：https://github.com/Tencent-Hunyuan/HunyuanWorld-Mirror
- 能力：单图/多图/视频 → 深度、法线、相机、点云、3DGS、mesh、3DGS 渲染
- 本机：Windows 11、AMD RX 6900 XT 16GB、无 CUDA
- 实现：**Rust + wgpu/Vulkan，纯 GPU 前向；不依赖 CUDA / PyTorch / gsplat**
- 模型：1.26B 参数，5.05GB f32，1544 个 tensor
- 输入：S × 518 × 518，默认最多 8 视图
- 端到端：
  - 1 视图：**11.14s**
  - 2 视图：**21.5s**
  - 4 视图：**49.1s**
  - 视频：OpenCV 抽帧后走多视图
- 输出：`output/vulkan-2view-full/`
  - `depth_0000.png` / `normals_0000.png` / `input_0000.png`
  - `points.ply`（482,983 点）
  - `gaussians.ply`（343,922 splats，标准 17-float 3DGS PLY）
  - `render_0000.png`（内置 3DGS 渲染）
  - `turntable.png`（环绕视图）
  - `mesh-tsdf.ply` / `mesh-poisson.ply` / `mesh-*.png`
  - `camera_*.f32`、`result.json`

## 数值验证

官方 PyTorch CPU f32 为参考：

| 阶段 | 单图 max abs | 2 视图 max abs |
| --- | ---: | ---: |
| DINOv2 24 blocks | 2.46e-5 | — |
| VGT 48 blocks | 6.10e-5 | — |
| camera | 1.12e-7 | 2.98e-7 |
| depth | 3.22e-6 | 5.91e-5 |
| pts3d | 8.34e-5 | 1.24e-4 |
| normals | 6.08e-5 | 5.00e-4 |
| gs raw | 8.20e-5 | — |
| splats | 41,390 vs 41,386 | 343,922（合并后） |

## 运行

```powershell
cd ai-lab\hunyuanworld\port\hwm-vk
cargo build --release --bin hwm-vk

# 单图
.\target\release\hwm-vk.exe `
  --image D:\3DWord\itseez3d\assets\rainforest\photo.jpg `
  --out ..\..\output\vulkan-photo

# 多图 + mesh + turntable
.\target\release\hwm-vk.exe `
  --images .\frames --max-views 8 `
  --mesh tsdf --mesh-render --turntable 8 `
  --out ..\..\output\vulkan-multi

# 视频
.\target\release\hwm-vk.exe --video .\clip.mp4 --fps 1 --max-views 8 --out ..\..\output\vulkan-video
```

详细架构、算子、验证和限制见 [PORTING.md](PORTING.md)。

## 限制

- 默认最多 8 视图；global attention O(S²)，16GB 显存下 8 视图是安全上限。
- 模型输出点云/3DGS，mesh 是 Open3D TSDF/Poisson 后处理，不是模型原生输出。
- 内置 3DGS renderer 是 isotropic 软件实现；各向异性画质请用外部 gsplat/VkSplat。
- 只做推理，不含训练或 gsplat CUDA rasterizer。
