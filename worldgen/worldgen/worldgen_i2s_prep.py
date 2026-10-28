"""Build the Fill condition/mask: a 90-degree full-height band in a pano."""

import argparse
from pathlib import Path

from PIL import Image, ImageOps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--width", type=int, default=1024)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    W, H = args.width, args.height
    pw = W // 4
    x0 = (W - pw) // 2

    src = Image.open(args.image).convert("RGB")
    img = ImageOps.fit(src, (pw, H), method=Image.LANCZOS)
    cond = Image.new("RGB", (W, H), (0, 0, 0))
    cond.paste(img, (x0, 0))
    mask = Image.new("L", (W, H), 255)
    mask.paste(0, (x0, 0, x0 + pw, H))
    cond.save(out / "cond.png")
    mask.save(out / "mask.png")

    try:
        import cv2
        import numpy as np

        im = np.asarray(src)[:, :, ::-1].copy()
        h, w = im.shape[:2]
        gc = np.zeros((h, w), np.uint8)
        rect = (int(w * 0.25), int(h * 0.1), int(w * 0.5), int(h * 0.8))
        bgd = np.zeros((1, 65), np.float64)
        fgd = np.zeros((1, 65), np.float64)
        cv2.grabCut(im, gc, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
        fg = np.where((gc == 1) | (gc == 3), 255, 0).astype(np.uint8)
        band = cv2.resize(im, (pw, H))
        fg = cv2.resize(fg, (pw, H), interpolation=cv2.INTER_NEAREST)
        clean = cv2.inpaint(band, fg, 5, cv2.INPAINT_TELEA)
        cond_clean = Image.new("RGB", (W, H), (0, 0, 0))
        cond_clean.paste(Image.fromarray(clean[:, :, ::-1]), (x0, 0))
        cond_clean.save(out / "cond_clean.png")
    except Exception:
        cond.save(out / "cond_clean.png")
    print("cond", cond.size, "mask", mask.size, flush=True)


if __name__ == "__main__":
    main()
