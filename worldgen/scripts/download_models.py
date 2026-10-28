"""Download or verify WorldGen Vulkan model assets.

Gated FLUX files require a Hugging Face token with access to
black-forest-labs/FLUX.1-dev and FLUX.1-Fill-dev.
"""

import argparse
import hashlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = Path(os.environ.get("WORLDGEN_MODELS", ROOT / "models"))

ASSETS = {
    "flux-dev/ae.safetensors": (
        "https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/ae.safetensors",
        335304388,
        None,
    ),
    "flux-dev/clip_l.safetensors": (
        "https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/clip_l.safetensors",
        None,
        None,
    ),
    "flux-dev/t5xxl_fp8_e4m3fn.safetensors": (
        "https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/t5xxl_fp8_e4m3fn.safetensors",
        None,
        None,
    ),
    "flux-fill/flux1-fill-dev.safetensors": (
        "https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev/resolve/main/flux1-fill-dev.safetensors",
        23804922408,
        "03e289f530df51d014f48e675a9ffa2141bc003259bf5f25d75b957e920a41ca",
    ),
    "flux-dev/worldgen_text2scene.safetensors": (
        "https://huggingface.co/LeoXie/WorldGen/resolve/main/models--WorldGen-Flux-Lora/worldgen_text2scene.safetensors",
        34647632,
        None,
    ),
    "flux-dev/worldgen_img2scene.safetensors": (
        "https://huggingface.co/LeoXie/WorldGen/resolve/main/models--WorldGen-Flux-Lora/worldgen_img2scene.safetensors",
        44917720,
        None,
    ),
    "da2/model.safetensors": (
        "https://huggingface.co/haodongli/DA-2/resolve/main/model.safetensors",
        1378513064,
        None,
    ),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    args = ap.parse_args()
    MODELS.mkdir(parents=True, exist_ok=True)

    missing = []
    for rel, (url, size, digest) in ASSETS.items():
        path = MODELS / rel
        ok = path.exists()
        if ok and size and path.stat().st_size != size:
            ok = False
        if ok and args.check and digest and sha256(path) != digest:
            ok = False
        print(("OK   " if ok else "MISS ") + str(path))
        if not ok:
            missing.append((rel, url))

    if missing:
        print("\nMissing:")
        for rel, url in missing:
            print(f"  {rel}\n    {url}")
        print("\nFLUX.1-dev / Fill are gated; accept the license and set HF_TOKEN.")
        print("The FLUX dev q4 GGUF is user-supplied; expected SHA256:")
        print("  962d536a2a51342b4793e46cb2c491f3ea5fa62a66fbfc38684638fa7bff646b")
        print("Convert the official Fill safetensors with:")
        print("  sd-cli --mode convert -m models/flux-fill/flux1-fill-dev.safetensors \\")
        print("    -o models/flux-fill/flux-fill-official-q4.gguf --type q4_K")
        sys.exit(1)

    if args.download:
        print("automatic download of gated files is intentionally not implemented;")
        print("use scripts/download_one.py or huggingface-cli with HF_TOKEN.")


if __name__ == "__main__":
    main()
