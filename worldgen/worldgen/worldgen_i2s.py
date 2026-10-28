"""Single image -> FLUX Fill panorama -> DA-2 depth -> Rust 3DGS/mesh."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from worldgen_da2 import depth as da2_depth

ROOT = Path(__file__).resolve().parents[1]
MODELS = Path(os.environ.get("WORLDGEN_MODELS", ROOT / "models"))
FLUX = MODELS / "flux-dev"
SD = Path(os.environ.get("SD_CLI", ROOT / "tools/sd-cli.exe"))
WG = Path(os.environ.get("WORLDGEN_VK", ROOT / "tools/worldgen-vk/worldgen-vk.exe"))


def run(cmd):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gguf", default=str(MODELS / "flux-fill/flux-fill-official-q4.gguf"))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    prep = out / "prep"
    run([
        sys.executable, str(ROOT / "worldgen/worldgen_i2s_prep.py"),
        "--image", args.image, "--out", str(prep),
        "--width", str(args.width), "--height", str(args.height),
    ])

    prompt = (
        "A high quality 360 panorama photo of, "
        + args.prompt
        + ", HDR, RAW, 360 consistent, omnidirectional"
        + " <lora:worldgen_img2scene:1>"
    )
    run([
        str(SD),
        "--diffusion-model", args.gguf,
        "--vae", str(FLUX / "ae.safetensors"),
        "--clip_l", str(FLUX / "clip_l.safetensors"),
        "--t5xxl", str(FLUX / "t5xxl_fp8_e4m3fn.safetensors"),
        "-i", str(prep / "cond.png"),
        "--mask", str(prep / "mask.png"),
        "-p", prompt,
        "-n", "blurry, low quality",
        "--lora-model-dir", str(FLUX),
        "-W", str(args.width), "-H", str(args.height),
        "--steps", str(args.steps),
        "--cfg-scale", "1.0",
        "--guidance", "30.0",
        "--sampling-method", "euler",
        "--strength", "1.0",
        "--seed", str(args.seed),
        "--vae-tiling",
        "--vae-tile-size", "512x512",
        "--backend", "diffusion=vulkan0",
        "-o", str(out / "pano_fill.png"),
    ])

    g = np.asarray(Image.open(out / "pano_fill.png").convert("RGB")).astype(np.float32)
    c = np.asarray(Image.open(prep / "cond.png").convert("RGB")).astype(np.float32)
    m = Image.open(prep / "mask.png").convert("L").filter(ImageFilter.GaussianBlur(8))
    valid = (1.0 - np.asarray(m).astype(np.float32) / 255.0)[..., None]
    pano = (g * (1.0 - valid) + c * valid).clip(0, 255).astype(np.uint8)
    Image.fromarray(pano).save(out / "pano.png")

    d = da2_depth(Image.fromarray(pano), args.width, args.height)
    d.astype("<f4").tofile(out / "depth.f32")
    run([
        str(WG), "both",
        "--pano", str(out / "pano.png"),
        "--depth", str(out / "depth.f32"),
        "--width", str(args.width),
        "--height", str(args.height),
        "--out", str(out / "scene"),
    ])
    print("done", out)


if __name__ == "__main__":
    main()
