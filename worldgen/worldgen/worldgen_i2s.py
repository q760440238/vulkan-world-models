"""Single image -> FLUX Fill panorama -> DA-2 depth -> Rust 3DGS/mesh."""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
from plyfile import PlyData, PlyElement

from worldgen_da2 import depth as da2_depth

ROOT = Path(__file__).resolve().parents[1]
MODELS = Path(os.environ.get("WORLDGEN_MODELS", ROOT / "models"))
FLUX = MODELS / "flux-dev"
SD = Path(os.environ.get("SD_CLI", ROOT / "tools/sd-cli.exe"))
WG = Path(os.environ.get("WORLDGEN_VK", ROOT / "tools/worldgen-vk/worldgen-vk.exe"))


def run(cmd):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run(cmd, check=True)


def foreground_mask(image_path: str, width: int, height: int) -> np.ndarray:
    import cv2

    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    gc = np.zeros((h, w), np.uint8)
    rect = (int(w * 0.25), int(h * 0.1), int(w * 0.5), int(h * 0.8))
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    cv2.grabCut(img, gc, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
    fg = np.where((gc == 1) | (gc == 3), 255, 0).astype(np.uint8)
    pw = width // 4
    fg = cv2.resize(fg, (pw, height), interpolation=cv2.INTER_NEAREST)
    full = np.zeros((height, width), np.uint8)
    x0 = (width - pw) // 2
    full[:, x0:x0 + pw] = fg
    return cv2.dilate(full, np.ones((5, 5), np.uint8), iterations=2)


def merge_splat(base_path: Path, bg_path: Path, mask: np.ndarray, out_path: Path) -> int:
    base = PlyData.read(str(base_path))
    bg = PlyData.read(str(bg_path))
    bv = base["vertex"].data
    gv = bg["vertex"].data
    if len(gv) != mask.size or len(bv) != mask.size:
        raise RuntimeError(f"splat vertex count mismatch: base={len(bv)} bg={len(gv)} mask={mask.size}")
    merged = np.concatenate([bv, gv[mask.reshape(-1) > 0]], axis=0)
    PlyData([PlyElement.describe(merged, "vertex")]).write(str(out_path))
    return len(merged)


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
    ap.add_argument("--inpaint-bg", action="store_true")
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

    g_img = Image.open(out / "pano_fill.png").convert("RGB")
    c_img = Image.open(prep / "cond.png").convert("RGB")
    mask_img = Image.open(prep / "mask.png").convert("L")
    try:
        import cv2
        g_bgr = cv2.cvtColor(np.asarray(g_img), cv2.COLOR_RGB2BGR)
        c_bgr = cv2.cvtColor(np.asarray(c_img), cv2.COLOR_RGB2BGR)
        valid_mask = 255 - np.asarray(mask_img)
        ys, xs = np.where(valid_mask > 128)
        center = (int((xs.min() + xs.max()) / 2), int((ys.min() + ys.max()) / 2))
        pano = cv2.seamlessClone(c_bgr, g_bgr, valid_mask, center, cv2.NORMAL_CLONE)
        pano = cv2.cvtColor(pano, cv2.COLOR_BGR2RGB)
    except Exception:
        g = np.asarray(g_img).astype(np.float32)
        c = np.asarray(c_img).astype(np.float32)
        m = mask_img.filter(ImageFilter.GaussianBlur(24))
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

    if args.inpaint_bg:
        import cv2
        fg = foreground_mask(args.image, args.width, args.height)
        pano_bg = cv2.inpaint(np.asarray(pano)[:, :, ::-1], fg, 3, cv2.INPAINT_TELEA)[:, :, ::-1]
        Image.fromarray(pano_bg).save(out / "pano_bg.png")
        bg_depth = da2_depth(Image.fromarray(pano_bg), args.width, args.height)
        bg_depth.astype("<f4").tofile(out / "depth_bg.f32")
        run([str(WG), "splat", "--pano", str(out / "pano_bg.png"),
             "--depth", str(out / "depth_bg.f32"),
             "--width", str(args.width), "--height", str(args.height),
             "--out", str(out / "scene_bg")])
        shutil.copy2(out / "scene/gaussians.ply", out / "scene/gaussians_base.ply")
        n = merge_splat(out / "scene/gaussians_base.ply",
                        out / "scene_bg/gaussians.ply", fg,
                        out / "scene/gaussians.ply")
        print("inpaint_bg splats", n, flush=True)

    print("done", out)


if __name__ == "__main__":
    main()
