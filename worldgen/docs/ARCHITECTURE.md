# Architecture

```text
worldgen/worldgen_t2s.py     prompt -> panorama -> depth -> scene
worldgen/worldgen_i2s.py     image -> Fill panorama -> depth -> scene
worldgen/worldgen_i2s_prep.py 90-degree band cond/mask
worldgen/worldgen_da2.py     ncnn Vulkan DA-2 bridge

tools/stable-diffusion.cpp   FLUX.1-dev / FLUX.1-Fill-dev (C++/Vulkan)
tools/da2-ncnn/da2ncnn.exe   DA-2 (C++/ncnn/Vulkan)
rust/                        worldgen-vk (Rust): RGBD -> 3DGS/mesh PLY
da2/main.cpp                 ncnn runner source
patches/da2/                 DA-2 export fixes
```

## FLUX stage

`stable-diffusion.cpp` with `SD_VULKAN=ON`. Diffusion, T5, CLIP and VAE all
run on Vulkan0 (the VAE uses `--vae-tiling --vae-tile-size 512x512` to avoid
a 9 s untiled decode).

## DA-2 stage

DA-2 is a DINOv2 ViT-L/14 + SphereViT head. The export path:

1. Apply `patches/da2/` (remove `quantile`, separate QKV projections).
2. Trace at 1x3x512x1024.
3. pnnx -> ncnn (0 custom layers).
4. `da2ncnn.exe` runs the graph on Vulkan.

The runner must be built with `-DNCNN_VULKAN` and call
`net.set_vulkan_device(0)`; otherwise ncnn silently uses CPU (8.7 s vs 3.05 s).

## Rust stage

`worldgen-vk both` converts pano+depth to:

- standard 17-float 3DGS PLY
- vertex-colored mesh PLY with a panorama UV seam fix

It is CPU geometry, 0.29 s for 524,288 splats / 1,046,528 triangles.

## Coordinate conventions

WorldGen uses the same normalized camera/center formula as the Python
reference; validation of center/scale/quaternion is ~1e-7.
