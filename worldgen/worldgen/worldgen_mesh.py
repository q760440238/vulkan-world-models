"""Open3D Poisson reconstruction for a WorldGen panorama + depth map."""

import argparse
import time
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image
from scipy.spatial import cKDTree


def rays(w: int, h: int) -> np.ndarray:
    u = (np.arange(w, dtype=np.float32) + 0.5) / w
    v = (np.arange(h, dtype=np.float32) + 0.5) / h
    phi = u * (2.0 * np.pi) - np.pi
    theta = v * np.pi - np.pi / 2.0
    PHI, THETA = np.meshgrid(phi, theta)
    x = np.cos(THETA) * np.sin(PHI)
    y = np.sin(THETA)
    z = np.cos(THETA) * np.cos(PHI)
    return np.stack([x, y, z], axis=-1).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pano", required=True)
    ap.add_argument("--depth", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--poisson-depth", type=int, default=9)
    ap.add_argument("--density-trim", type=float, default=0.05)
    ap.add_argument("--estimate-normals", action="store_true")
    ap.add_argument("--voxel", type=float, default=0.0)
    ap.add_argument("--method", choices=["poisson", "alpha", "ball"], default="poisson")
    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--largest-component", action="store_true")
    ap.add_argument("--normal-outward", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    w, h = args.width, args.height
    rgb = np.asarray(Image.open(args.pano).convert("RGB").resize((w, h))).astype(np.float32) / 255.0
    depth = np.fromfile(args.depth, "<f4").reshape(h, w).astype(np.float32)
    rs = rays(w, h)
    pts = (rs * depth[..., None]).reshape(-1, 3)
    colors = rgb.reshape(-1, 3)
    valid = np.isfinite(pts).all(axis=1) & (depth.reshape(-1) > 1e-4)
    pts = pts[valid]
    colors = colors[valid]

    # Analytic normals from neighboring surface points; degenerate/pole rows
    # are skipped rather than given a random normal.
    p = (rs * depth[..., None]).astype(np.float32)
    tx = np.roll(p, -1, axis=1) - np.roll(p, 1, axis=1)
    ty = np.zeros_like(p)
    ty[1:-1] = p[2:] - p[:-2]
    ty[0] = p[1] - p[0]
    ty[-1] = p[-1] - p[-2]
    n = np.cross(tx, ty)
    ln = np.linalg.norm(n, axis=-1, keepdims=True)
    good = (ln[..., 0] > 1e-8) & valid.reshape(h, w)
    n = np.divide(n, ln, out=np.zeros_like(n), where=ln > 1e-8)
    # Panorama camera is inside the scene: orient normals toward the origin.
    inward = (n * (-p)).sum(axis=-1, keepdims=True) < 0
    if not args.normal_outward:
        n = np.where(inward, -n, n)
    n = n.reshape(-1, 3)[valid]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.normals = o3d.utility.Vector3dVector(n)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    if args.voxel > 0:
        pcd = pcd.voxel_down_sample(args.voxel)
    if args.estimate_normals:
        pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=30)
        )
        pcd.orient_normals_towards_camera_location(np.zeros(3, dtype=np.float64))
    pts = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    if args.method == "alpha":
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
            pcd, args.alpha
        )
        dens = np.zeros(0)
    elif args.method == "ball":
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, o3d.utility.DoubleVector([0.02, 0.05, 0.1])
        )
        dens = np.zeros(0)
    else:
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=args.poisson_depth, width=0, scale=1.1, linear_fit=False
        )
        dens = np.asarray(densities)
    if args.density_trim > 0 and len(dens):
        mesh.remove_vertices_by_mask(dens < np.quantile(dens, args.density_trim))
    if args.largest_component:
        clusters, counts, _ = mesh.cluster_connected_triangles()
        clusters = np.asarray(clusters)
        keep = int(np.argmax(np.asarray(counts)))
        mesh.remove_triangles_by_mask(clusters != keep)
        mesh.remove_unreferenced_vertices()

    # Transfer colors from the nearest input point.
    v = np.asarray(mesh.vertices)
    tree = cKDTree(pts)
    _, idx = tree.query(v, k=1)
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors[idx])
    mesh.compute_vertex_normals()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(str(out), mesh, write_ascii=False, compressed=False)
    print(
        f"poisson: {len(pts)} points -> {len(mesh.vertices)} verts / "
        f"{len(mesh.triangles)} tris, {time.time() - t0:.1f}s -> {out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
