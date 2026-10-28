# LingBot-World Vulkan kernel port（部分完成）/ partial port

**中文（主） | [English](#english)**

> **状态：PARTIAL / BLOCKED。完整 1.3B 推理尚未完成，不能称为“已移植”。**
>
> 已完成的是一批 Vulkan correctness kernel、单个/小型 DiT block、以及部分 VAE
> 组件；不是完整 LingBot-World 推理。

在 **AMD Radeon RX 6900 XT + Windows** 上，用 **Rust + wgpu / Vulkan** 逐算子
验证 LingBot-World 2.0 1.3B causal-fast 的可行性，不依赖 CUDA。

![frame](first_frame_fix_final.png)

## 已验证的 Vulkan 组件

| 组件 | 状态 |
| --- | --- |
| RMSNorm / LayerNorm | 通过，误差 ~2e-6 |
| GEMM / FP16 GEMM / tiled GEMM | 通过 |
| RoPE / 3D-RoPE | 通过 |
| multi-head causal attention | 通过；12 heads × L1560 |
| QKV → 3D RoPE → attention | 通过，约 93ms |
| attention 完整层（out proj + residual） | 通过 |
| KV cache append / sink+window | 通过 |
| Q=1560, S=37440 KV attention | 通过，分块总 4.75s |
| cross-attention / time-MLP | 通过 |
| modulated / self+cross / camera block | 通过（小尺度） |
| 4 heads × head_dim 128 block | 通过（D=512 / F=512，**不是实际尺度**） |
| 4×128 + 3D-RoPE / camera block | 通过 |
| conv3d（含 causal）/ GroupNorm / upsample3d | 通过 |
| streaming causal conv3d / resblock / decoder tail | 通过 |
| VAE decoder（2×resblock + attention + 4×upsample） | 通过（小尺度） |
| 30 层真实 block 链式运行 | 跑完 199.2s，0 non-finite |
| T5 UMT5-XXL encoder | **CPU**：load 34.88s / encode 9.69s |

## 未完成 / 阻塞

- **完整 1.3B DiT 前向**：没有把 30 层真实 block、真实 text/camera、完整
  CFG/denoise 串成一次推理。
- **完整 VAE decoder**：只完成小尺度/部分 streaming 组件；没有真实 480×832
  视频解码链路。
- **T5 上 Vulkan**：目前 T5 仍在 CPU，未移植。
- **4-step FlowUniPC + 16 帧 chunk 视频生成**：没有稳定产出。
- **16GB 显存不足**：官方峰值约 30.7GB，逐 chunk 约 26.4GB，需要权重离载 /
  流式 / 分块；目前未完成。
- **FP8**：RX 6900 XT 不支持 `VK_EXT_shader_float8`，上游 FP8 路径不能原样移植。
- **14B 版本**：未开始，16GB 更不可行。

## 结论

当前最多只能称为：

```text
LingBot-World Vulkan kernel feasibility / partial block port
```

不能称为：

```text
LingBot-World 1.3B inference ported to Vulkan
```

完整移植还需要：真实尺度 DiT block、完整 VAE decoder、T5 路径、显存流式、
4-step 采样与视频 chunk 的端到端整合。

## 目录

```text
wgpu/                 # 每个算子一个独立 Rust wgpu crate
glsl/                 # shaders
first_frame_fix_final.png
first_frame_4680_fix.png
world_frame_0.png
vae_recon.png
```

源码与完整记录：
<https://github.com/q760440238/vulkan-world-models/tree/main/lingbot>

## 许可

- LingBot-World 上游代码/权重：CC BY-NC-SA 4.0
- 本仓库移植代码与文档：CC BY-NC-SA 4.0
- 不包含上游权重

---

# English

> **Status: PARTIAL / BLOCKED. Full 1.3B inference is not implemented.**

This is a Vulkan **kernel/component feasibility port**, not a complete
LingBot-World inference engine.

Validated: RMSNorm/LayerNorm, GEMM (f32/FP16/tiled), RoPE/3D-RoPE, multi-head
causal attention, QKV→RoPE→attention, KV cache with sink+window, cross-attention,
time-MLP, small modulated/self+cross/camera blocks, 4×128 blocks, conv3d
(including causal), GroupNorm, upsample3d, streaming resblock/decoder tail, and
a small-scale VAE decoder with attention. A 30-layer chain of real blocks ran to
completion (199.2s, 0 non-finite), and the T5 UMT5-XXL encoder runs on CPU.

Not done: full 1.3B DiT forward with real text/camera and denoising, full VAE
decoder at 480×832, T5 on Vulkan, 4-step FlowUniPC video chunks, and the
memory-streaming needed for 16GB VRAM (official peak ~30.7GB). RX 6900 XT does
not support `VK_EXT_shader_float8`, so the upstream FP8 path cannot be used
as-is. The 14B model is not started.

License: CC BY-NC-SA 4.0. Upstream weights are not included.
