"""Minimal point-based 3DGS preview (no splat covariance) for validation."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def load_ply(path: Path):
    b = path.read_bytes()
    hs = b.index(b"end_header\n") + 11
    head = b[:hs].decode()
    n = int([l for l in head.splitlines() if l.startswith("element vertex")][0].split()[2])
    v = np.frombuffer(b[hs:hs + n * 17 * 4], dtype="<f4").reshape(n, 17)
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--yaw", type=float, default=0.0)
    ap.add_argument("--pitch", type=float, default=0.0)
    ap.add_argument("--dist", type=float, default=2.2)
    ap.add_argument("--width", type=int, default=832)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--inside", action="store_true")
    ap.add_argument("--equirect", action="store_true")
    ap.add_argument("--radius", type=int, default=1)
    ap.add_argument("--fov", type=float, default=90.0)
    args = ap.parse_args()

    v = load_ply(Path(args.ply))
    c = v[:, :3].astype(np.float64)
    col = np.clip(0.5 + 0.2820947918 * v[:, 6:9], 0, 1)
    alpha = 1.0 / (1.0 + np.exp(-v[:, 16]))
    keep = alpha > 0.1
    c, col = c[keep], col[keep]

    lo, hi = np.percentile(c, 2, axis=0), np.percentile(c, 98, axis=0)
    ctr = (lo + hi) / 2
    rad = float(np.max(np.linalg.norm(c - ctr, axis=1)))
    yaw, pitch = np.radians(args.yaw), np.radians(args.pitch)
    if args.inside:
        eye = np.zeros(3)
        f = np.array([np.sin(yaw) * np.cos(pitch), np.sin(pitch), np.cos(yaw) * np.cos(pitch)])
        f /= np.linalg.norm(f)
        r = np.cross(f, np.array([0.0, 1.0, 0.0]))
        r /= np.linalg.norm(r)
        u = np.cross(r, f)
    else:
        eye = ctr + args.dist * rad * np.array([
            np.sin(yaw) * np.cos(pitch), -np.sin(pitch), np.cos(yaw) * np.cos(pitch)
        ])
        f = ctr - eye
        f /= np.linalg.norm(f)
        r = np.cross(f, np.array([0.0, 1.0, 0.0]))
        r /= np.linalg.norm(r)
        u = np.cross(r, f)
    W, H = args.width, args.height
    cc = col
    if args.equirect:
        depth = np.linalg.norm(c, axis=1)
        lon = np.arctan2(c[:, 0], c[:, 2])
        lat = np.arcsin(np.clip(c[:, 1] / np.maximum(depth, 1e-6), -1, 1))
        uu = ((lon + np.pi) / (2 * np.pi) * W).astype(np.int32) % W
        vv = ((lat + np.pi / 2) / np.pi * H).astype(np.int32).clip(0, H - 1)
        zz = depth
    else:
        fx = (W / 2) / np.tan(np.radians(args.fov) / 2)
        d = c - eye
        xc, yc, zc = d @ r, d @ u, d @ f
        ok = zc > 0.05
        uu = (fx * xc[ok] / zc[ok] + W / 2).astype(np.int32)
        vv = (fx * yc[ok] / zc[ok] + H / 2).astype(np.int32)
        zz = zc[ok]
        cc = cc[ok]
        inside = (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H)
        uu, vv, zz, cc = uu[inside], vv[inside], zz[inside], cc[inside]

    img = np.ones((H, W, 3), np.float32) * np.array([0.78, 0.87, 0.93])
    zbuf = np.full((H, W), np.inf, np.float32)
    r = max(0, args.radius)
    order = np.argsort(-zz)
    for i in order:
        x, y, z = uu[i], vv[i], zz[i]
        if z >= zbuf[y, x]:
            continue
        x0, x1 = max(0, x - r), min(W, x + r + 1)
        y0, y1 = max(0, y - r), min(H, y + r + 1)
        patch = zbuf[y0:y1, x0:x1]
        m = z < patch
        patch[m] = z
        img[y0:y1, x0:x1][m] = cc[i]
    Image.fromarray((img * 255).astype(np.uint8)).save(args.out)
    print("points", len(v), "projected", len(uu), "radius", rad, "->", args.out)


if __name__ == "__main__":
    main()
