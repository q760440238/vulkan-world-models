# WorldGen Vulkan validation

## Environment

```text
Windows 11
AMD Radeon RX 6900 XT 16GB
Vulkan 1.4 (AMD proprietary driver)
No CUDA
```

## Pipeline

```text
t2s:
  prompt
   -> FLUX.1-dev Q4_K_S + WorldGen text2scene LoRA
      (stable-diffusion.cpp, SD_VULKAN=ON)
   -> 1024x512 equirectangular panorama
   -> DA-2 depth (ncnn Vulkan, fp32)
   -> worldgen-vk (Rust): RGBD -> 3DGS PLY + mesh PLY

i2s:
  image
   -> 90-degree full-height band -> cond/mask
   -> FLUX.1-Fill-dev Q4_K (official safetensors converted with sd-cli)
      + WorldGen img2scene LoRA
   -> Fill pano, composite original image region back
   -> DA-2 depth (ncnn Vulkan)
   -> worldgen-vk (Rust): RGBD -> 3DGS PLY + mesh PLY
```

## Verified runs

### t2s

```text
prompt: a cozy wooden bedroom with warm lamps
1024x512, 20 steps
output: output/t2s-ncnn-clean/
  524,288 splats
  1,046,528 triangles
```

### i2s

```text
input: assets/rainforest/photo.jpg
1024x512, 20 steps
output: output/i2s-ncnn-clean/
  524,288 splats
  1,046,528 triangles
```

### i2s seam blending

The Fill output and the original input band are composited with
`cv2.seamlessClone` (NORMAL_CLONE). Without OpenCV, a 24 px feathered alpha
blend is used. This removes the visible vertical band boundary; it does not
remove Fill object repetition.

### 1600x800

```text
t2s, 1600x800, 10 steps: 739 s
output: 1,280,000 splats, 1600x800 pano

DA-2 native 1600x800 trace:
  graph      da2_trace_1600.ncnn.param/.bin
  time       6.29 s (Vulkan f32)
  max abs    0.144
  mean abs   0.0041
  cosine     0.99954

full 50 steps: roughly 1 hour on this GPU
```

### inpaint_bg (simplified)

`worldgen_i2s.py --inpaint-bg`:

```text
GrabCut foreground mask
-> OpenCV Telea background inpaint
-> DA-2 depth on background
-> worldgen-vk splat
-> merge foreground splats + background splats in the foreground mask
```

Test: 1024x512, 5 steps, 524,288 base splats + 20,529 background splats
= 544,817 merged vertices. This is not the official OneFormer + FLUX Fill
branch; it is a CPU/OpenCV approximation.

### Clean builds from source

Tested with local source checkouts (same scripts also clone when `-Source` is
omitted):

```text
build_sd.ps1 -Source <stable-diffusion.cpp>   -> build-sd/bin/Release/sd-cli.exe
build_da2.ps1 -Source <ncnn>                  -> tools/da2-ncnn/da2ncnn.exe + ncnn.dll
build_rust.ps1                                -> tools/worldgen-vk/worldgen-vk.exe
```

`sd-cli.exe --list-devices` reports `Vulkan0  AMD Radeon RX 6900 XT`; the
freshly built DA-2 runner reproduces the 1024x512 output on the same GPU.

### DA-2 numeric and performance

ncnn Vulkan vs PyTorch, 1024x512:

```text
max abs   0.0161
mean abs  0.0027
cosine    0.9999994
time      3.05 s (f32 Vulkan)
```

The ncnn graph has **0 custom layers**. The runner must be compiled with
`-DNCNN_VULKAN` and must call `net.set_vulkan_device(0)`; otherwise ncnn
silently falls back to CPU (the earlier 8.7 s result). The fixed trace
resolution is 1024x512; other pano sizes are resampled.

### Latency (1024x512, 20 steps, RX 6900 XT)

```text
t2s total          56.7 s
i2s total          61.5 s
FLUX sampling      ~100% GPU 3D utilization
VAE decode         9.0 s untiled -> 1.35 s with --vae-tiling --vae-tile-size 512x512
DA-2               3.05 s
Rust RGBD->3DGS    0.29 s
```

### RGBD -> 3DGS math

Compared with the WorldGen Python formula on synthetic panorama/depth:

```text
center      ~1e-7
scale       ~1e-7
quaternion  ~1e-7
```

### FLUX

- stable-diffusion.cpp built with `SD_VULKAN=ON`
- `Vulkan0 = AMD Radeon RX 6900 XT`
- FLUX.1-dev Q4_K_S SHA256 `962d536a...` matches
- FLUX.1-Fill-dev official safetensors SHA256 `03e289f5...` matches
- converted q4_K: 780 tensors
- text2scene LoRA: 312/312 applied
- img2scene LoRA: 456/684 applied (228 incompatible keys skipped)

## No PyTorch at inference

`worldgen_t2s.py`, `worldgen_i2s.py`, `worldgen_i2s_prep.py` and
`worldgen_da2.py` do not import torch. Model inference is:

```text
FLUX/Fill -> sd-cli (C++/Vulkan)
DA-2      -> ncnn (C++/Vulkan)
3DGS/mesh -> worldgen-vk (Rust, CPU geometry + PLY)
```

## Known gaps

- Not byte/quality-equivalent to official WorldGen:
  - Fill repeats objects and has a visible band boundary.
  - Mesh has some stretching from depth discontinuities.
- Optional `use_sharp=True` and `inpaint_bg=True` branches are not wired.
- Official WorldGen visual/numeric comparison of the full FLUX stage is not done;
  only component-level validation (DA-2, 3DGS math, weights, pipeline runs).
- Default validation resolution is 1024x512; 1600x800 is supported by
  resampling but is much slower (~1 h per 50-step panorama).
- ncnn/pnnx toolchain is C++, not Rust; Rust covers RGBD->3DGS/mesh.
