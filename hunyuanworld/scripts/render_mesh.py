#!/usr/bin/env python3
"""Render a mesh to PNG with Open3D's Vulkan-backed offscreen renderer."""
import argparse
from pathlib import Path

import numpy as np
import open3d as o3d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=768)
    args = ap.parse_args()

    mesh = o3d.io.read_triangle_mesh(str(args.mesh))
    mesh.compute_vertex_normals()
    bbox = mesh.get_axis_aligned_bounding_box()
    center = bbox.get_center()
    diag = float(np.linalg.norm(bbox.get_max_bound() - bbox.get_min_bound()))
    eye = center + np.array([0.9, 0.45, 0.9]) * max(diag, 1e-6)

    renderer = o3d.visualization.rendering.OffscreenRenderer(args.width, args.height)
    renderer.scene.set_background([0.96, 0.96, 0.96, 1.0])
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultLit"
    mat.base_color = [0.75, 0.78, 0.82, 1.0]
    renderer.scene.add_geometry("mesh", mesh, mat)
    renderer.setup_camera(
        50.0,
        center.astype(np.float32),
        eye.astype(np.float32),
        np.array([0.0, 1.0, 0.0], np.float32),
    )
    img = renderer.render_to_image()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_image(str(args.out), img)
    print(f"wrote {args.out} ({args.width}x{args.height}) vertices={len(mesh.vertices)}")


if __name__ == "__main__":
    main()
