import argparse
import struct
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--rgb", action="store_true")
    ap.add_argument("--pitch", type=float, default=0.0, help="rotate world about X for the GPU camera")
    args = ap.parse_args()
    b = Path(args.ply).read_bytes()
    hs = b.index(b"end_header\n") + 11
    head = b[:hs].decode()
    n = int([l for l in head.splitlines() if l.startswith("element vertex")][0].split()[2])
    v = np.frombuffer(b[hs:hs + n * 17 * 4], dtype="<f4").reshape(n, 17)
    out = np.empty((n, 14), "<f4")
    out[:, :3] = v[:, :3]
    if args.pitch:
        a = np.radians(args.pitch)
        c, s = np.cos(a), np.sin(a)
        R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]], "<f4")
        out[:, :3] = v[:, :3] @ R.T
        w, x, y, z = v[:, 13], v[:, 14], v[:, 15], v[:, 16]
        qw, qx, qy, qz = c, s, 0.0, 0.0
        out[:, 10] = qw * w - qx * x - qy * y - qz * z
        out[:, 11] = qw * x + qx * w + qy * z - qz * y
        out[:, 12] = qw * y - qy * w + qz * x - qx * z
        out[:, 13] = qw * z + qz * w + qx * y - qy * x
    else:
        out[:, 10:14] = v[:, 13:17]
    out[:, 3:6] = (
        np.clip(0.5 + 0.28209479177387814 * v[:, 6:9], 0, 1)
        if args.rgb else v[:, 6:9]
    )
    out[:, 6] = v[:, 9]
    out[:, 7:10] = v[:, 10:13]
    w, h = args.width, args.height
    intr = np.zeros(9, "<f4")
    # The viewer scales intrinsics from a 2048x1152 reference frame.
    intr[0] = 1024.0
    intr[2] = 1024.0
    intr[4] = 1024.0
    intr[5] = 576.0
    intr[8] = 1.0
    props = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
             "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    header = "ply\nformat binary_little_endian 1.0\nelement vertex %d\n" % n
    header += "".join("property float %s\n" % p for p in props) + "end_header\n"
    Path(args.out).write_bytes(header.encode() + out.tobytes() + b"\0" * 64 + intr.tobytes())
    print(args.out, n)


if __name__ == "__main__":
    main()
