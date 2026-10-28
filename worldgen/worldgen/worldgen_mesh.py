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


def tsdf_mesh(rgb: np.ndarray, depth: np.ndarray, face: int, voxel: float, trunc: float):
    """Integrate six 90-degree cube faces into a TSDF and extract a mesh."""
    from scipy.ndimage import map_coordinates

    h, w = depth.shape
    n = face
    u = (np.arange(n, dtype=np.float32) + 0.5) / n * 2.0 - 1.0
    uu, vv = np.meshgrid(u, u)
    faces = [
        ((0, 0, 1), (1, 0, 0), (0, 1, 0)),
        ((0, 0, -1), (-1, 0, 0), (0, 1, 0)),
        ((1, 0, 0), (0, 0, -1), (0, 1, 0)),
        ((-1, 0, 0), (0, 0, 1), (0, 1, 0)),
        ((0, 1, 0), (1, 0, 0), (0, 0, -1)),
        ((0, -1, 0), (1, 0, 0), (0, 0, 1)),
    ]
    vol = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel,
        sdf_trunc=trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )
    intr = o3d.camera.PinholeCameraIntrinsic(n, n, n / 2, n / 2, n / 2, n / 2)
    for f, r, d in faces:
        f = np.asarray(f, np.float32)
        r = np.asarray(r, np.float32)
        up = np.asarray(d, np.float32)
        dirs = f[None, None, :] + uu[..., None] * r - vv[..., None] * up
        dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)
        lon = np.arctan2(dirs[..., 0], dirs[..., 2])
        lat = np.arcsin(np.clip(dirs[..., 1], -1, 1))
        x = (lon + np.pi) / (2 * np.pi) * w - 0.5
        y = (lat + np.pi / 2) / np.pi * h - 0.5
        x = np.mod(x, w)
        coords = np.stack([y, x], axis=0)
        face_rgb = np.stack(
            [map_coordinates(rgb[..., c], coords, order=1, mode="nearest") for c in range(3)],
            axis=-1,
        ).clip(0, 1)
        rad = map_coordinates(depth, coords, order=1, mode="nearest")
        z = rad * np.maximum(dirs @ f, 1e-4)
        z[~np.isfinite(z)] = 0
        z[z < 0.1] = 0
        print("face", f, "z", float(z.min()), float(z.max()), float(z.mean()), flush=True)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image((face_rgb * 255).astype(np.uint8)),
            o3d.geometry.Image((z * 1000.0).astype(np.float32)),
            depth_scale=1000.0,
            depth_trunc=50.0,
            convert_rgb_to_intensity=False,
        )
        ext = np.array(
            [[*r, 0.0], [*up, 0.0], [*f, 0.0], [0.0, 0.0, 0.0, 1.0]],
            np.float64,
        )
        vol.integrate(rgbd, intr, ext)
    print("voxels", len(vol.extract_voxel_point_cloud().points), flush=True)
    mesh = vol.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    return mesh


def voxel_tsdf(points: np.ndarray, normals: np.ndarray, colors: np.ndarray,
               res: float, trunc: float, smooth: int = 1):
    """Signed-distance voxel field + marching cubes (a simple TSDF)."""
    from scipy.ndimage import gaussian_filter
    from skimage.measure import marching_cubes

    lo = points.min(axis=0) - res * 2
    hi = points.max(axis=0) + res * 2
    dims = np.maximum(np.ceil((hi - lo) / res).astype(int), 4)
    xs = lo[0] + np.arange(dims[0]) * res
    ys = lo[1] + np.arange(dims[1]) * res
    zs = lo[2] + np.arange(dims[2]) * res
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    centers = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)

    tree = cKDTree(points)
    dist, idx = tree.query(centers, k=1, workers=-1)
    nearest = points[idx]
    nrm = normals[idx]
    side = np.einsum("ij,ij->i", centers - nearest, nrm)
    field = np.sign(side) * np.minimum(dist / trunc, 1.0)
    field = field.reshape(dims).astype(np.float32)
    if smooth > 0:
        field = gaussian_filter(field, smooth).astype(np.float32)

    verts, faces, _, _ = marching_cubes(field, level=0.0, spacing=(res, res, res))
    verts = verts + lo
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(verts),
        o3d.utility.Vector3iVector(faces.astype(np.int32)),
    )
    _, cidx = tree.query(verts, k=1, workers=-1)
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors[cidx])
    mesh.compute_vertex_normals()
    print(f"voxel grid {dims.tolist()}, {len(verts)} verts / {len(faces)} tris", flush=True)
    return mesh


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
    ap.add_argument("--method", choices=["poisson", "alpha", "ball", "tsdf", "voxel"], default="poisson")
    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--voxel-res", type=float, default=0.2)
    ap.add_argument("--voxel-trunc", type=float, default=0.4)
    ap.add_argument("--voxel-stride", type=int, default=1)
    ap.add_argument("--face-size", type=int, default=512)
    ap.add_argument("--voxel-size", type=float, default=0.1)
    ap.add_argument("--sdf-trunc", type=float, default=0.2)
    ap.add_argument("--largest-component", action="store_true")
    ap.add_argument("--normal-outward", action="store_true")
    ap.add_argument("--median", type=int, default=0)
    args = ap.parse_args()

    t0 = time.time()
    w, h = args.width, args.height
    rgb = np.asarray(Image.open(args.pano).convert("RGB").resize((w, h))).astype(np.float32) / 255.0
    depth = np.fromfile(args.depth, "<f4").reshape(h, w).astype(np.float32)
    if args.median > 1:
        from scipy.ndimage import median_filter
        depth = median_filter(depth, size=args.median, mode="nearest")
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

    if args.method == "voxel":
        s = max(1, args.voxel_stride)
        mesh = voxel_tsdf(pts[::s], n[::s], colors[::s], args.voxel_res, args.voxel_trunc)
        dens = np.zeros(0)
    elif args.method == "tsdf":
        mesh = tsdf_mesh(rgb, depth, args.face_size, args.voxel_size, args.sdf_trunc)
        dens = np.zeros(0)
    elif args.method == "alpha":
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
        f"{args.method}: {len(pts)} points -> {len(mesh.vertices)} verts / "
        f"{len(mesh.triangles)} tris, {time.time() - t0:.1f}s -> {out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
