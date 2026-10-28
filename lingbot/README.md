# LingBot-World Vulkan 推理 / Vulkan inference

**中文（主） | [English](#english)**

在 **AMD Radeon RX 6900 XT + Windows** 上，用 **Rust + wgpu / Vulkan** 跑
LingBot-World 2.0 1.3B causal-fast，不依赖 CUDA。

> 这是工程移植与数值验证仓库，不是 LingBot 官方仓库。权重不包含在本仓库。

![LingBot frame](first_frame_fix_final.png)

## 已完成

| 项目 | 结果 |
| --- | --- |
| 模型 | LingBot-World 2.0 1.3B causal-fast |
| 层数 | 30 |
| dim / FFN | 1536 / 8960 |
| 采样 | FlowUniPC，4 steps |
| VAE | Wan2.1 VAE，z16 |
| 文本编码 | UMT5-XXL，CPU |
| 后端 | Rust + wgpu / Vulkan |
| 硬件 | AMD RX 6900 XT 16GB |
| 单帧 token | 1560 |
| 验证 | 各 wgpu kernel 与 CPU/PyTorch 参考对齐 |

已落地的 Vulkan 组件：

```text
RMSNorm / LayerNorm
GEMM（f32 / FP16）
RoPE / 3D-RoPE
QKV / attention / KV cache / sink+window
12-head L1560 batch
DiT block（self + cross + camera）
camera MLP / camera patch / plucker
time-MLP
conv3d / causal conv3d / stream conv3d
GroupNorm + SiLU
VAE resblock / attention / up2d / up3d / decoder
```

## 目录

```text
wgpu/                 # 每个算子一个独立 Rust wgpu crate
glsl/                 # shaders
first_frame_fix_final.png
first_frame_4680_fix.png
world_frame_0.png
vae_recon.png
```

完整数值记录、benchmark、权重说明见 GitHub：
<https://github.com/q760440238/vulkan-world-models/tree/main/lingbot>

## 运行

每个 `wgpu/wgpu-*` 都是独立 crate：

```powershell
cd wgpu/wgpu-dit-block
cargo build --release
```

完整 1.3B 推理需要：

- 真实权重（DiT 6.8GB + VAE 0.5GB + T5 10.6GB，约 18GB）
- CPU driver / 参考脚本
- 本仓库不包含权重

权重来源：

- `Robbyant/lingbot-world-v2-1.3b-causal-fast`
- `Robbyant/lingbot-world-v2-14b-causal-fast` 下的 `Wan2.1_VAE.pth`、`models_t5_umt5-xxl-enc-bf16.pth`

## 限制

- 单帧 4-step 已跑通；多帧 4680-token chunk 仍会退化。
- 14B 版本没有移植。
- 完整交互式连续视频生成还没有稳定产出。
- 权重许可为 CC BY-NC-SA 4.0，禁止商业使用。

## 许可

- LingBot-World 上游代码/权重：**CC BY-NC-SA 4.0**
- 本仓库的移植代码与文档：CC BY-NC-SA 4.0
- 不包含上游权重

---

# English

Rust + wgpu/Vulkan port of **LingBot-World 2.0 1.3B causal-fast** for AMD
Windows, without CUDA.

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

Implemented kernels: RMSNorm/LayerNorm, f32/FP16 GEMM, RoPE/3D-RoPE, QKV,
attention, KV cache, sink+window, 12-head L1560, DiT self/cross/camera blocks,
camera MLP/patch/plucker, time-MLP, conv3d/causal/streaming conv3d,
GroupNorm+SiLU, VAE resblock/attention/up2d/up3d/decoder.

Weights (about 18GB) are **not included**. Upstream:
`Robbyant/lingbot-world-v2-1.3b-causal-fast` and
`Robbyant/lingbot-world-v2-14b-causal-fast`.

Limitations: single-frame 4-step works; multi-frame 4680-token chunks degrade;
14B is not ported; full interactive continuous video is not stable.

License: CC BY-NC-SA 4.0 (non-commercial). Not affiliated with the LingBot team.
