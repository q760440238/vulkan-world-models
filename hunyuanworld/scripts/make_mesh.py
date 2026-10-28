#!/usr/bin/env python3
"""Build a mesh from HunyuanWorld Vulkan outputs using Open3D.

Examples:
    python make_mesh.py --dir out --method tsdf
    python make_mesh.py --dir out --method poisson
    python make_mesh.py --dir out --method both --voxel-size 0.003

The model itself produces depth/point clouds, not meshes; this is a separate
CPU post-process (TSDF fusion or Poisson reconstruction).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image


def load_cameras(d: Path):
    try:
        meta = json.loads((d / "result.json").read_text())
        views = int(meta["views"])
        cams = meta["cameras"]
        out = []
        for i in range(views):
            c2w = np.asarray(cams[i]["c2w"], np.float64).reshape(4, 4)
            k = np.asarray(cams[i]["intrinsics"], np.float64).reshape(3, 3)
            out.append((c2w, k))
        return views, out
    except Exception:
        # Fallback for older / partially written result.json
        params = np.fromfile(d / "camera_params.f32", np.float32)
        views = params.size // 9
        poses = np.fromfile(d / "camera_poses.f32", np.float32).reshape(views, 4, 4)
        intrs = np.fromfile(d / "camera_intrs.f32", np.float32).reshape(views, 3, 3)
        return views, [(poses[i].astype(np.float64), intrs[i].astype(np.float64)) for i in range(views)]


def load_view(d: Path, i: int):
    depth = np.fromfile(d / f"depth_{i:04d}.f32", np.float32).reshape(518, 518)
    color = np.asarray(Image.open(d / f"input_{i:04d}.png").convert("RGB"))
    return depth, color


def tsdf(d: Path, resolution: int, voxel_hint: float):
    views, cams = load_cameras(d)
    pcd = o3d.io.read_point_cloud(str(d / "points.ply"))
    if len(pcd.points):
        lo, hi = pcd.get_min_bound(), pcd.get_max_bound()
    else:
        lo, hi = np.array([-0.5] * 3), np.array([0.5] * 3)
    extent = float(np.max(hi - lo))
    margin = max(0.05 * extent, 1e-3)
    origin = lo - margin
    length = extent + 2 * margin
    res = resolution
    voxel = length / res
    if voxel_hint > 0:
        res = max(64, min(512, int(np.ceil(length / voxel_hint))))
        voxel = length / res
    trunc = 4.0 * voxel
    depth_trunc = max(4.0 * extent, 1.0)
    print(f"tsdf: bounds={length:.4f} res={res} voxel={voxel:.5f} trunc={trunc:.5f}")

    vol = o3d.pipelines.integration.UniformTSDFVolume(
        length=length,
        resolution=res,
        sdf_trunc=trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )
    # UniformTSDFVolume spans [0, length]^3; shift the world so the scene
    # bounding box starts at the origin.
    shift = np.eye(4)
    shift[:3, 3] = origin
    for i, (c2w, k) in enumerate(cams):
        depth, color = load_view(d, i)
        intr = o3d.camera.PinholeCameraIntrinsic(518, 518, k[0, 0], k[1, 1], k[0, 2], k[1, 2])
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(color),
            o3d.geometry.Image(depth),
            depth_scale=1.0,
            depth_trunc=depth_trunc,
            convert_rgb_to_intensity=False,
        )
        # world' = world - origin, so w2c' = w2c @ T(origin)
        vol.integrate(rgbd, intr, np.linalg.inv(c2w) @ shift)
    mesh = vol.extract_triangle_mesh()
    mesh.translate(origin)
    mesh.compute_vertex_normals()
    return mesh


def poisson(d: Path):
    pcd = o3d.io.read_point_cloud(str(d / "points.ply"))
    if len(pcd.points) == 0:
        raise SystemExit("points.ply is empty")
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30))
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
    densities = np.asarray(densities)
    if densities.size:
        mesh.remove_vertices_by_mask(densities < np.quantile(densities, 0.02))
    mesh.compute_vertex_normals()
    return mesh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--method", choices=["tsdf", "poisson", "both"], default="tsdf")
    ap.add_argument("--resolution", type=int, default=256, help="TSDF volume resolution")
    ap.add_argument("--voxel-size", type=float, default=0.0, help="0 = length/resolution")
    ap.add_argument("--trunc", type=float, default=0.0, help="0 = 4*voxel")
    args = ap.parse_args()
    d = args.dir

    if args.method in ("tsdf", "both"):
        m = tsdf(d, args.resolution, args.voxel_size)
        p = d / "mesh-tsdf.ply"
        o3d.io.write_triangle_mesh(str(p), m)
        print(f"tsdf: {p} vertices={len(m.vertices)} triangles={len(m.triangles)}")
    if args.method in ("poisson", "both"):
        m = poisson(d)
        p = d / "mesh-poisson.ply"
        o3d.io.write_triangle_mesh(str(p), m)
        print(f"poisson: {p} vertices={len(m.vertices)} triangles={len(m.triangles)}")


if __name__ == "__main__":
    main()
