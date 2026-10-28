"""Equirectangular 3DGS renderer with per-Gaussian covariance (numba)."""

import argparse
from pathlib import Path

import numpy as np
from numba import njit, prange
from PIL import Image

SH_C0 = 0.28209479177387814


def load_ply(path: Path):
    b = path.read_bytes()
    hs = b.index(b"end_header\n") + 11
    head = b[:hs].decode()
    n = int([l for l in head.splitlines() if l.startswith("element vertex")][0].split()[2])
    return np.frombuffer(b[hs:hs + n * 17 * 4], dtype="<f4").reshape(n, 17)


@njit(parallel=True)
def render(centers, scales, quats, colors, opacity, W, H, max_radius, max_sigma, max_lat):
    img = np.zeros((H, W, 3), np.float32)
    trans = np.zeros((H, W), np.float32)
    n = centers.shape[0]
    for i in range(n):
        x, y, z = centers[i, 0], centers[i, 1], centers[i, 2]
        r = np.sqrt(x * x + y * y + z * z)
        if r < 1e-4:
            continue
        lon = np.arctan2(x, z)
        lat = np.arcsin(max(-1.0, min(1.0, y / r)))
        if abs(lat) > np.radians(max_lat):
            continue
        u = (lon + np.pi) / (2 * np.pi) * W
        v = (lat + np.pi / 2) / np.pi * H
        # quaternion -> rotation matrix
        w, qx, qy, qz = quats[i, 0], quats[i, 1], quats[i, 2], quats[i, 3]
        qn = np.sqrt(w * w + qx * qx + qy * qy + qz * qz)
        if qn < 1e-8:
            w, qx, qy, qz = 1.0, 0.0, 0.0, 0.0
        else:
            w, qx, qy, qz = w / qn, qx / qn, qy / qn, qz / qn
        R = np.empty((3, 3), np.float32)
        R[0, 0] = 1 - 2 * (qy * qy + qz * qz)
        R[0, 1] = 2 * (qx * qy - w * qz)
        R[0, 2] = 2 * (qx * qz + w * qy)
        R[1, 0] = 2 * (qx * qy + w * qz)
        R[1, 1] = 1 - 2 * (qx * qx + qz * qz)
        R[1, 2] = 2 * (qy * qz - w * qx)
        R[2, 0] = 2 * (qx * qz - w * qy)
        R[2, 1] = 2 * (qy * qz + w * qx)
        R[2, 2] = 1 - 2 * (qx * qx + qy * qy)
        s = np.exp(scales[i])
        C = np.empty((3, 3), np.float32)
        for a in range(3):
            for b in range(3):
                t = 0.0
                for k in range(3):
                    t += R[a, k] * s[k] * s[k] * R[b, k]
                C[a, b] = t
        sl, cl = np.sin(lon), np.cos(lon)
        st, ct = np.sin(lat), np.cos(lat)
        er = np.array([-sl, 0.0, cl], np.float32)
        et = np.array([-st * cl, ct, -st * sl], np.float32)
        crr = er[0] * C[0, 0] * er[0] + er[0] * C[0, 1] * er[1] + er[0] * C[0, 2] * er[2] + \
              er[1] * C[1, 0] * er[0] + er[1] * C[1, 1] * er[1] + er[1] * C[1, 2] * er[2] + \
              er[2] * C[2, 0] * er[0] + er[2] * C[2, 1] * er[1] + er[2] * C[2, 2] * er[2]
        ctt = et[0] * C[0, 0] * et[0] + et[0] * C[0, 1] * et[1] + et[0] * C[0, 2] * et[2] + \
              et[1] * C[1, 0] * et[0] + et[1] * C[1, 1] * et[1] + et[1] * C[1, 2] * et[2] + \
              et[2] * C[2, 0] * et[0] + et[2] * C[2, 1] * et[1] + et[2] * C[2, 2] * et[2]
        crt = er[0] * C[0, 0] * et[0] + er[0] * C[0, 1] * et[1] + er[0] * C[0, 2] * et[2] + \
              er[1] * C[1, 0] * et[0] + er[1] * C[1, 1] * et[1] + er[1] * C[1, 2] * et[2] + \
              er[2] * C[2, 0] * et[0] + er[2] * C[2, 1] * et[1] + er[2] * C[2, 2] * et[2]
        r2 = r * r
        su = W / (2 * np.pi)
        sv = H / np.pi
        a11 = crr / r2 * su * su
        a22 = ctt / r2 * sv * sv
        a12 = crt / r2 * su * sv
        tr = a11 + a22
        det = a11 * a22 - a12 * a12
        if det <= 1e-12:
            continue
        l1 = 0.5 * tr + np.sqrt(max(0.25 * tr * tr - det, 0.0))
        if max_sigma > 0 and np.sqrt(l1) > max_sigma:
            continue
        rad = int(min(max_radius, max(1.0, 3.0 * np.sqrt(l1))))
        inv_det = 1.0 / det
        u0 = int(np.floor(u))
        v0 = int(np.floor(v))
        op = opacity[i]
        for yy in range(v0 - rad, v0 + rad + 1):
            if yy < 0 or yy >= H:
                continue
            dy = yy + 0.5 - v
            for xx in range(u0 - rad, u0 + rad + 1):
                dx = xx + 0.5 - u
                while dx > W / 2:
                    dx -= W
                while dx < -W / 2:
                    dx += W
                xw = xx % W
                q = a22 * dx * dx - 2 * a12 * dx * dy + a11 * dy * dy
                g = np.exp(-0.5 * q * inv_det)
                a = op * g
                if a < 0.003:
                    continue
                img[yy, xw, 0] = img[yy, xw, 0] * (1 - a) + colors[i, 0] * a
                img[yy, xw, 1] = img[yy, xw, 1] * (1 - a) + colors[i, 1] * a
                img[yy, xw, 2] = img[yy, xw, 2] * (1 - a) + colors[i, 2] * a
                trans[yy, xw] = trans[yy, xw] * (1 - a) + a
    for yy in range(H):
        for xx in range(W):
            t = trans[yy, xx]
            img[yy, xx] = img[yy, xx] + (1 - t) * np.array([0.78, 0.87, 0.93])
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--max-radius", type=int, default=8)
    ap.add_argument("--scale-factor", type=float, default=1.0)
    ap.add_argument("--max-sigma", type=float, default=0.0)
    ap.add_argument("--max-lat", type=float, default=90.0)
    args = ap.parse_args()

    v = load_ply(Path(args.ply))
    order = np.argsort(-np.linalg.norm(v[:, :3], axis=1))
    v = v[order]
    centers = v[:, :3].astype(np.float32)
    colors = np.clip(0.5 + SH_C0 * v[:, 6:9], 0, 1).astype(np.float32)
    opacity = (1.0 / (1.0 + np.exp(-v[:, 16]))).astype(np.float32)
    img = render(centers, (v[:, 10:13] + np.log(args.scale_factor)).astype(np.float32), v[:, 13:17].astype(np.float32),
                 colors, opacity, args.width, args.height, args.max_radius, args.max_sigma, args.max_lat)
    Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).save(args.out)
    print("rendered", len(v), "->", args.out)


if __name__ == "__main__":
    main()
