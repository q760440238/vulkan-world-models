# vulkan-splat

Standalone **wgpu / Vulkan** Gaussian-splat renderer for the compact 14-float PLY produced by `ml-sharp`.

## Build

```powershell
cd ai-lab/gaussian-splatting/vulkan-rs
cargo build --release --offline
```

## Run

```powershell
.\target\release\vulkan-splat.exe `
  ..\..\ml-sharp\output\photo.ply 0.35 splat.ppm
```

`argv[2]` is camera yaw in radians, `argv[3]` is output PNG; optional `argv[4]` is minimum splat sigma in pixels (default 0.10).

## Results

Canonical camera uses the PLY `intrinsic` (fx=1629.27, cx=1024, cy=576 for 2048x1152) and identity `extrinsic`, scaled to 832x480. This renders the source view instead of an arbitrary orbit.

```text
1,179,648 Gaussians
832x480
~421 ms including process startup
```

- Input: `ml-sharp/output/photo.ply`
- 1,179,648 Gaussians
- Output: 832x480 Vulkan render
- `preview.png`
- `turntable.gif` (8 yaw views)

## Implementation

- Parses binary little-endian PLY: `x y z f_dc_0..2 opacity scale_0..2 rot_0..3`
- `scale = exp(log_scale)`, `opacity = sigmoid`
- SH DC -> RGB: `0.5 + 0.2820948 * f_dc`
- 3D covariance `R S S^T R^T`
- EWA projection to 2D covariance
- Per-Gaussian screen-space quad, Gaussian alpha falloff
- Back-to-front CPU sort
- wgpu render pass, alpha blending, 832x480
- Direct PNG output (uncompressed zlib/deflate writer)

## Known issues / limits

- DC color only, no view-dependent SH
- Billboard splats, not tile-based 3DGS rasterizer
- No anti-aliasing / proper mip filtering
- Camera is a simple orbit, not the original COLMAP extrinsic


