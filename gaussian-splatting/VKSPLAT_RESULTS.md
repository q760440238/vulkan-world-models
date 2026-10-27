# VkSplat Vulkan training result

## Pipeline

- Source: https://github.com/harry7557558/vksplat
- Windows Python extension built locally against Vulkan headers/import lib.
- Slang shaders recompiled with `USE_EMULATED_F32_ATOMIC=1`.
- AMD Radeon RX 6900 XT, Vulkan backend.

## Dataset

14-view cropped COLMAP dataset:

- 13 orbit views
- 1 canonical view (`zzz.png`, kept in train)
- 550x330
- 60k initial points

`vksplat_dataset6/`

## Training evidence

- 10,000 steps
- 49.7 s
- 319,073 splats
- `ComputeTileRanges` 10,000 calls
- `SortRTS` 10,080 calls
- `RasterizeForward` 10,000 calls
- `RasterizeBackward` 10,000 calls
- `FusedProjectionBackwardOptimizerStep` 10,000 calls

This is a Vulkan tile rasterizer with per-tile sorting and a differentiable backward pass.

## Metrics

- Train reconstruction PSNR: **38.21 dB** (canonical image `zzz.png`)
- Validation PSNR: **20.65 dB** (one orbit view held out)
- SH rest coefficients: mean abs **0.0357**, max **0.8263** (non-zero, optimized)
- Splats grew from 60,000 initial points to **319,073** via densification

## Files

- `vksplat_dataset6/train.log`
- `vksplat_dataset6/out/train_00012.png` (canonical render)
- `vksplat_dataset6/images/zzz.png` (target)
- `vksplat_dataset6/out/val_00000.png`
- `vksplat/splat.ply` (trained PLY, 75 MB, not committed)

## Caveats

- Small synthetic multi-view set; train PSNR is much higher than held-out PSNR.
- Not a quality benchmark against Mip-NeRF 360.
- The trained PLY is in VkSplat's normalized world; the existing `vulkan-splat`
  viewer needs a coordinate transform before it can display it correctly.

## Conclusion

Vulkan tile binning, per-tile sorting, backward, SH optimization, and
densification all executed and were measured on AMD hardware.
