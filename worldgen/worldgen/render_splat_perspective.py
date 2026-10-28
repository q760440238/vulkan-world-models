"""Perspective 3DGS renderer with covariance projection (numba)."""

import argparse
from pathlib import Path

import numpy as np
from numba import njit
from PIL import Image

SH_C0 = 0.28209479177387814


def load_ply(path: Path):
    b = path.read_bytes()
    hs = b.index(b"end_header\n") + 11
    head = b[:hs].decode()
    n = int([l for l in head.splitlines() if l.startswith("element vertex")][0].split()[2])
    return np.frombuffer(b[hs:hs + n * 17 * 4], dtype="<f4").reshape(n, 17)


@njit
def render_persp(xc, yc, zc, scales, quats, colors, opacity, W, H, fx, fy, cyaw, syaw, max_radius, boost):
    img = np.zeros((H, W, 3), np.float32)
    trans = np.zeros((H, W), np.float32)
    n = xc.shape[0]
    for i in range(n):
        z = zc[i]
        if z < 0.05:
            continue
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
        # camera-space covariance (yaw about Y)
        cxx = C[0, 0]; cxz = C[0, 2]; czz = C[2, 2]
        cc = np.empty((3, 3), np.float32)
        cc[0, 0] = cyaw * cyaw * cxx - 2 * cyaw * syaw * cxz + syaw * syaw * czz
        cc[0, 2] = cyaw * syaw * (cxx - czz) + (cyaw * cyaw - syaw * syaw) * cxz
        cc[2, 0] = cc[0, 2]
        cc[2, 2] = syaw * syaw * cxx + 2 * syaw * cyaw * cxz + cyaw * cyaw * czz
        cc[1, 1] = C[1, 1]
        cc[0, 1] = cyaw * C[0, 1] - syaw * C[2, 1]
        cc[1, 0] = cc[0, 1]
        cc[1, 2] = syaw * C[0, 1] + cyaw * C[2, 1]
        cc[2, 1] = cc[1, 2]
        # perspective Jacobian
        iz = 1.0 / z
        j00 = fx * iz
        j02 = -fx * xc[i] * iz * iz
        j11 = fy * iz
        j12 = -fy * yc[i] * iz * iz
        a11 = j00 * j00 * cc[0, 0] + 2 * j00 * j02 * cc[0, 2] + j02 * j02 * cc[2, 2]
        a22 = j11 * j11 * cc[1, 1] + 2 * j11 * j12 * cc[1, 2] + j12 * j12 * cc[2, 2]
        a12 = j00 * j11 * cc[0, 1] + j00 * j12 * cc[0, 2] + j02 * j11 * cc[2, 1] + j02 * j12 * cc[2, 2]
        tr = a11 + a22
        det = a11 * a22 - a12 * a12
        if det <= 1e-12:
            continue
        l1 = 0.5 * tr + np.sqrt(max(0.25 * tr * tr - det, 0.0))
        rad = int(min(max_radius, max(1.0, 3.0 * np.sqrt(l1))))
        u = fx * xc[i] * iz + W / 2
        v = fy * yc[i] * iz + H / 2
        u0 = int(np.floor(u))
        v0 = int(np.floor(v))
        inv_det = 1.0 / det
        op = min(1.0, opacity[i] * boost)
        for yy in range(v0 - rad, v0 + rad + 1):
            if yy < 0 or yy >= H:
                continue
            dy = yy + 0.5 - v
            for xx in range(u0 - rad, u0 + rad + 1):
                if xx < 0 or xx >= W:
                    continue
                dx = xx + 0.5 - u
                q = a22 * dx * dx - 2 * a12 * dx * dy + a11 * dy * dy
                a = op * np.exp(-0.5 * q * inv_det)
                if a < 0.003:
                    continue
                img[yy, xx, 0] = img[yy, xx, 0] * (1 - a) + colors[i, 0] * a
                img[yy, xx, 1] = img[yy, xx, 1] * (1 - a) + colors[i, 1] * a
                img[yy, xx, 2] = img[yy, xx, 2] * (1 - a) + colors[i, 2] * a
                trans[yy, xx] = trans[yy, xx] * (1 - a) + a
    for yy in range(H):
        for xx in range(W):
            t = trans[yy, xx]
            img[yy, xx] = img[yy, xx] + (1 - t) * np.array([0.78, 0.87, 0.93])
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--width", type=int, default=832)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fov", type=float, default=90.0)
    ap.add_argument("--yaw", type=float, default=0.0)
    ap.add_argument("--scale-factor", type=float, default=1.0)
    ap.add_argument("--max-radius", type=int, default=8)
    ap.add_argument("--opacity-boost", type=float, default=1.0)
    args = ap.parse_args()

    v = load_ply(Path(args.ply))
    yaw = np.radians(args.yaw)
    cy, sy = np.cos(yaw), np.sin(yaw)
    c = v[:, :3].astype(np.float32)
    xc = cy * c[:, 0] - sy * c[:, 2]
    zc = sy * c[:, 0] + cy * c[:, 2]
    yc = c[:, 1].copy()
    order = np.argsort(-zc)
    v = v[order]
    xc, yc, zc = xc[order], yc[order], zc[order]
    colors = np.clip(0.5 + SH_C0 * v[:, 6:9], 0, 1).astype(np.float32)
    opacity = (1.0 / (1.0 + np.exp(-v[:, 16]))).astype(np.float32)
    fx = (args.width / 2) / np.tan(np.radians(args.fov) / 2)
    fy = fx
    img = render_persp(xc, yc, zc, (v[:, 10:13] + np.log(args.scale_factor)).astype(np.float32),
                       v[:, 13:17].astype(np.float32), colors, opacity,
                       args.width, args.height, fx, fy, cy, sy, args.max_radius, args.opacity_boost)
    Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).save(args.out)
    print("perspective", len(v), "->", args.out)


if __name__ == "__main__":
    main()
