# Vulkan-only path around CUDA and multi-view capture

## CUDA limitation

Current pipeline needs no CUDA for inference or rendering:

```text
single image
  -> ml-sharp (CPU) -> 3DGS PLY
  -> vulkan-splat (Rust/wgpu, Vulkan) -> PNG
```

`vulkan-splat` does not use `diff-gaussian-rasterization` or any CUDA extension.

## Multi-view limitation

Instead of capturing multiple photos, one image is lifted to 3DGS and the Vulkan renderer synthesizes a multi-view set:

```text
photo.ply
  -> vulkan-splat with yaw sweep
  -> multiview/000.png ... 012.png
  -> multiview/cameras.json (intrinsics + w2c)
```

This is a **pseudo multi-view** dataset for novel-view tests / initialization, not true parallax capture.

## Files

- `make_multiview.py`
- `multiview/*.png`
- `multiview/cameras.json`

## Remaining gap

A full trainable 3DGS still needs a **differentiable Vulkan rasterizer** (backward pass). The current Vulkan renderer is forward-only. Training/fine-tuning is therefore not yet CUDA-free; it is avoided by using the training-free SHARP model.
