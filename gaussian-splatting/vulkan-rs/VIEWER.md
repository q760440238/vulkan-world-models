# Chrome 3DGS viewers

## High-quality 13-view gallery

`gaussian_gallery.html` — 2.4 MB, no server needed.

- 13 Vulkan-rendered 3DGS views
- drag / slider to orbit
- wheel zoom
- space auto-play
- initial view = canonical (index 6)

Open:

```powershell
chrome "file:///D:/3DWord/itseez3d/ai-lab/gaussian-splatting/vulkan-rs/gaussian_gallery.html"
```

## Interactive Gaussian-center cloud

`gaussian_viewer.html` — 23.6 MB

- all 1,179,648 Gaussian centers
- WebGL2 point cloud
- drag orbit / wheel zoom / point-size slider
- not a full splat rasterizer

## Full Vulkan splat renderer

`vulkan-splat.exe` renders real 3DGS billboards:

```powershell
.\target\release\vulkan-splat.exe ..\..\ml-sharp\output\photo.ply 0.0 out.png 0.10
```

## Limitation

The gallery is a set of Vulkan renders, not a live continuous splat viewer.
The point cloud viewer is live but only shows centers.
A live continuous splat viewer needs the WebGL splat shader or a streaming server.

## Full 1.18M WebGL splat

`gaussian_splat_full.html` — 81.8 MB, local only (not committed).

- all 1,179,648 Gaussians
- 3D covariance -> EWA 2D
- instanced billboard quads
- canonical pre-sort
- URL `?yaw=...` for screenshots
- multi-angle screenshots: `splat_full_2_75.png`, `splat_full_3_14.png`, `splat_full_3_55.png`

Known issue: sort order is canonical only; large orbit angles show blending ghosting.

## Full splat viewer sorting fix

`gaussian_splat_full.html` now:

- 1,179,648 Gaussians
- all instances in one WebGL2 instanced draw
- per-view re-sort by **camera distance** on load and pointer-up
- depth test + alpha blend
- screenshots:
  - `sorted_3_14.png` canonical
  - `sorted_3_55.png` side view

The earlier ghosting came from sorting by view-space z, which flips sign when the camera crosses the scene. Sorting by Euclidean camera distance fixes canonical and large yaw views.

## Frustum culling + LOD + visible-only sort

`gaussian_splat_full.html` now:

- projects every center, culls behind-camera and offscreen splats
- drops sub-pixel/low-opacity splats
- sorts only the visible subset by camera distance
- uploads a compact visible instance buffer
- canonical view: 891,593 / 1,179,648 visible
- HUD shows live visible count

This is optimization items 2 and 3. Item 4 (WebGPU tile rasterizer) and item 5 (Vulkan backward) are still open.
