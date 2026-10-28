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

    img = ImageOps.fit(
        Image.open(args.image).convert("RGB"), (pw, H), method=Image.LANCZOS
    )
    cond = Image.new("RGB", (W, H), (0, 0, 0))
    cond.paste(img, (x0, 0))
    mask = Image.new("L", (W, H), 255)
    mask.paste(0, (x0, 0, x0 + pw, H))
    cond.save(out / "cond.png")
    mask.save(out / "mask.png")
    print("cond", cond.size, "mask", mask.size, flush=True)


if __name__ == "__main__":
    main()
