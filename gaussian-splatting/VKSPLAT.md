# VkSplat integration (Vulkan tile rasterizer + backward)

Source: https://github.com/harry7557558/vksplat (paper: VkSplat, Eurographics 2026 short paper)

## What was built

- Windows Python extension built against a local Vulkan header/import-lib shim
  (`vulkan-sdk/include`, `vulkan-sdk/lib/vulkan-1.lib` generated from system `vulkan-1.dll`).
- Slang 2026.2.1 downloaded, shaders recompiled with `USE_EMULATED_F32_ATOMIC=1`.
- `vksplat` imports and runs on AMD RX 6900 XT.
- Tile rasterizer present: `compute_tile_ranges.spv`, `rasterize_forward.spv`, `sort`.
- Backward present: `rasterize_backward_0..4.spv`,
  `fused_projection_backward_optimizer.spv`.
- A 30k-step run completed: 271,499 splats, ~139 s, ~0.31 GiB VRAM.

## Dataset

Synthetic COLMAP dataset built from 13 orbit renders of the SHARP point cloud:

- `vksplat_dataset4/images/*.png`
- `vksplat_dataset4/sparse/0/{cameras,images,points3D}.txt`

## Evidence

- `vksplat_dataset4/train30.log`
- `vksplat_dataset4/out30/train_00006.png`
- `vksplat/splat.ply` (trained PLY)
- `vulkan-rs/trained_orbit.png`

## Status

Tile rasterizer and backward are running, but the synthetic dataset / camera
normalization still gives poor visual convergence. Do not claim quality parity yet.
The next step is to validate on a real COLMAP scene and measure PSNR, then tune
the synthetic camera normalization.
