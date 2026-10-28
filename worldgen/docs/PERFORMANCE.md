# Performance notes

Tested: Windows 11, AMD RX 6900 XT 16GB, Vulkan 1.4, 1024x512, 20 steps.

| Stage | Before | After |
| --- | ---: | ---: |
| VAE decode | 9.0 s | 1.35 s (`--vae-tiling --vae-tile-size 512x512`) |
| DA-2 | 8.7 s (CPU fallback) | 3.05 s (Vulkan f32) |
| t2s total | ~75 s | 56.7 s |
| i2s total | — | 61.5 s |

## What helped

- VAE tiling: the untiled decode is memory-bound and leaves the GPU idle.
- ncnn `-DNCNN_VULKAN` + `net.set_vulkan_device(0)`: without both, ncnn
  silently runs DA-2 on CPU.
- f32 Vulkan for DA-2 is required: fp16 gives 2.5 s but max error 2.74 and
  cosine 0.972, which is not acceptable for depth.
- FLUX sampling already reaches ~100% GPU 3D utilization at 1024x512.

## What did not help

- Forcing `--flash-attn on`: the probe reports it slower than the normal path
  on this RDNA2 GPU (11.68 s vs 4.05 s for one step).
- fp16 VAE: decode remained ~9.1 s untiled.
- `torch.amax` inside the DA-2 graph: broke the ncnn export; normalization is
  done by the caller instead.

## Remaining headroom

- 1600x800 needs a separate DA-2 trace or resampling.
- T5 prompt encoding is ~3.4 s and not cached.
- Optional `use_sharp` / `inpaint_bg` branches are not wired.
