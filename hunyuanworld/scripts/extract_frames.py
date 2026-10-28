#!/usr/bin/env python3
"""Extract frames from a video for hwm-vk --video.

Usage:
    python extract_frames.py input.mp4 out_frames 1.0 8
"""
import sys
from pathlib import Path

import cv2


def main() -> None:
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2])
    fps = float(sys.argv[3])
    max_views = int(sys.argv[4])
    dst.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {src}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(src_fps / fps)))
    idx = 0
    saved = 0
    while saved < max_views:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            cv2.imwrite(str(dst / f"frame_{saved:04d}.png"), frame)
            saved += 1
        idx += 1
    cap.release()
    print(f"extracted {saved} frames to {dst}")


if __name__ == "__main__":
    main()
