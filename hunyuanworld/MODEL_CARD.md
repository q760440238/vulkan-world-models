---
license: apache-2.0
library_name: rust
tags:
  - vulkan
  - wgpu
  - 3d-reconstruction
  - depth-estimation
  - surface-normals
  - gaussian-splatting
  - mesh
  - hunyuanworld
  - amd
---

# HunyuanWorld-Mirror 鈥?Vulkan inference

Native **Rust + wgpu/Vulkan** inference for
[HunyuanWorld-Mirror](https://github.com/Tencent-Hunyuan/HunyuanWorld-Mirror).
No CUDA or PyTorch at inference time.

- single image / multi-view images / video frames
- depth, normals, camera poses/intrinsics
- point cloud, 3D Gaussian Splatting
- optional TSDF/Poisson mesh
- built-in 3DGS render + turntable

Performance: 518脳518 single view 鈮?**11.2 s** on an AMD RX 6900 XT; two views
鈮?21.5 s (1.26 B params, 5.05 GB f32).

## Contents

- `src/` 鈥?Rust + WGSL engine (72 transformer blocks, camera head, 4 DPT heads, 3DGS)
- `scripts/download_weights.py` 鈥?download official weights + build `weights.idx`
- `scripts/make_mesh.py` 鈥?Open3D TSDF/Poisson mesh
- `scripts/extract_frames.py` 鈥?OpenCV video frame extraction
- `docs/` 鈥?outputs

**Weights are not included.** Download them from
[`tencent/HunyuanWorld-Mirror`](https://huggingface.co/tencent/HunyuanWorld-Mirror);
they are governed by the Tencent HunyuanWorld-Mirror Community License
Agreement, not Apache-2.0.

## Quick start

```bash
git clone https://github.com/q760440238/vulkan-world-models.git
cd vulkan-world-models/hunyuanworld
python scripts/download_weights.py
cargo build --release --bin hwm-vk
./target/release/hwm-vk --images frames --mesh tsdf --mesh-render --turntable 8 --out out
```

## Validation

| Output | single view | two views |
| --- | ---: | ---: |
| camera | 1.12e-7 | 2.98e-7 |
| depth | 3.22e-6 | 5.91e-5 |
| pts3d | 8.34e-5 | 1.24e-4 |
| normals | 6.08e-5 | 5.00e-4 |

## License

Source: Apache-2.0. Model/weights: Tencent HunyuanWorld-Mirror Community License;
territory restrictions apply. Not affiliated with Tencent.

