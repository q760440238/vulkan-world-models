"""Text -> 360 panorama -> DA-2 depth -> Rust 3DGS/mesh."""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image

from worldgen_da2 import depth as da2_depth

ROOT = Path(__file__).resolve().parents[1]
MODELS = Path(os.environ.get("WORLDGEN_MODELS", ROOT / "models"))
FLUX = MODELS / "flux-dev"
SD = Path(os.environ.get("SD_CLI", ROOT / "tools/sd-cli.exe"))
WG = Path(os.environ.get("WORLDGEN_VK", ROOT / "tools/worldgen-vk/worldgen-vk.exe"))
GPU_SPLAT = Path(os.environ.get("GPU_SPLAT", ROOT / "tools/gpu-splat/gpu-splat.exe"))


def run(cmd):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gguf", default=str(FLUX / "flux1-dev-q4_k.gguf"))
    ap.add_argument("--no-lora", action="store_true")
    ap.add_argument("--mesh-shell", action="store_true")
    ap.add_argument("--cpu-mesh", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    prompt = (
        "A high quality 360 panorama photo of "
        + args.prompt
        + ", HDR, RAW, 360 consistent, omnidirectional"
    )
    if not args.no_lora:
        prompt += " <lora:worldgen_text2scene:1>"

    run([
        str(SD),
        "--diffusion-model", args.gguf,
        "--vae", str(FLUX / "ae.safetensors"),
        "--clip_l", str(FLUX / "clip_l.safetensors"),
        "--t5xxl", str(FLUX / "t5xxl_fp8_e4m3fn.safetensors"),
        "-p", prompt,
        "--lora-model-dir", str(FLUX),
        "--width", str(args.width),
        "--height", str(args.height),
        "--steps", str(args.steps),
        "--cfg-scale", "1.0",
        "--guidance", "3.5",
        "--sampling-method", "euler",
        "--seed", str(args.seed),
        "--circularx",
        ["--vae-tiling", "--vae-tile-size", ("512x512")],
        "--backend", "diffusion=vulkan0",
        "-o", str(out / "pano.png"),
    ])

    pano = Image.open(out / "pano.png").convert("RGB")
    d = da2_depth(pano, args.width, args.height)
    d.astype("<f4").tofile(out / "depth.f32")
    print("depth", d.shape, float(d.min()), float(d.max()), flush=True)

    cpu_mesh = args.mesh_shell or args.cpu_mesh
    run([
        str(WG), "both" if cpu_mesh else "splat",
        "--pano", str(out / "pano.png"),
        "--depth", str(out / "depth.f32"),
        "--width", str(args.width),
        "--height", str(args.height),
        "--out", str(out / "scene"),
    ])
    if cpu_mesh:
        if not args.mesh_shell:
            shutil.copy2(out / "scene/mesh.ply", out / "scene/mesh_shell.ply")
            run([
                sys.executable, str(ROOT / "worldgen/worldgen_mesh.py"),
                "--pano", str(out / "pano.png"),
                "--depth", str(out / "depth.f32"),
                "--out", str(out / "scene/mesh.ply"),
                "--width", str(args.width), "--height", str(args.height),
                "--method", "voxel", "--voxel-res", "0.25", "--voxel-stride", "4",
                "--median", "5", "--trim-poles", "0.02", "--voxel-smooth", "2.5",
                "--largest-component",
            ])
    else:
        ply14 = out / "scene/gaussians14.ply"
        run([
            sys.executable, str(ROOT / "worldgen/worldgen_make14.py"),
            "--ply", str(out / "scene/gaussians.ply"), "--out", str(ply14),
            "--width", str(args.width), "--height", str(args.height), "--rgb", "--pitch", "-15",
        ])
        run([str(GPU_SPLAT), str(ply14), "0", str(out / "splat.png"), "0.2"])
    print("done", out)


if __name__ == "__main__":
    main()



