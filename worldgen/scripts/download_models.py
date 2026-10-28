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
    "flux-dev/flux1-dev-q4_k.gguf": (
        None,
        6925526176,
        "962d536a2a51342b4793e46cb2c491f3ea5fa62a66fbfc38684638fa7bff646b",
    ),
    "flux-dev/ae.safetensors": (
        "https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/ae.safetensors",
        335304388,
        "afc8e28272cd15db3919bacdb6918ce9c1ed22e96cb12c4d5ed0fba823529e38",
    ),
    "flux-dev/clip_l.safetensors": (
        "https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/clip_l.safetensors",
        246144152,
        "660c6f5b1abae9dc498ac2d21e1347d2abdb0cf6c0c0c8576cd796491d9a6cdd",
    ),
    "flux-dev/t5xxl_fp8_e4m3fn.safetensors": (
        "https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/t5xxl_fp8_e4m3fn.safetensors",
        4893934904,
        "315fadec5f39e44cdaa20af56446fcb1c7c9b17091209ee7ba0f03d47160518f",
    ),
    "flux-fill/flux1-fill-dev.safetensors": (
        "https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev/resolve/main/flux1-fill-dev.safetensors",
        23804922408,
        "03e289f530df51d014f48e675a9ffa2141bc003259bf5f25d75b957e920a41ca",
    ),
    "flux-fill/flux-fill-official-q4.gguf": (
        None,
        6793133088,
        "096c26692c17a99a8e86961bcc3aafe1de01af7ee7a1b56218a01f6c48624c5d",
    ),
    "flux-dev/worldgen_text2scene.safetensors": (
        "https://huggingface.co/LeoXie/WorldGen/resolve/main/models--WorldGen-Flux-Lora/worldgen_text2scene.safetensors",
        34647632,
        "5349c785650892914b32e74d88d16baa5863981a63a8747b70332e67e0feb10b",
    ),
    "flux-dev/worldgen_img2scene.safetensors": (
        "https://huggingface.co/LeoXie/WorldGen/resolve/main/models--WorldGen-Flux-Lora/worldgen_img2scene.safetensors",
        44917720,
        "b10a1fa212fcff58c87a8264b1b9a8c01853febcd4d3f5c239ab7d7fd35d17ef",
    ),
    "da2/model.safetensors": (
        "https://huggingface.co/haodongli/DA-2/resolve/main/model.safetensors",
        1378513064,
        "d8ea568fc3dfb7d7432e5b763de499dd03fb6dc1c2020d84639f35e2dfa4f78e",
    ),
    "da2/da2_trace_1024.ncnn.param": (
        None,
        46271,
        "cd0a3fcb3d8e143e7b65e6706a15d60467bdf6e49bc584c4351bfbcd9436ac86",
    ),
    "da2/da2_trace_1024.ncnn.bin": (
        None,
        744102084,
        "e7773c23b2d117ac9f5085a92e8df06b46abba05011ac90da33c65b53ea479d9",
    ),
    "da2/da2_trace_1600.ncnn.param": (
        None,
        46271,
        "9ffbbc8e4348bc712c92a348a86f9b4dc5279e7ca574ef143c2b772173e23d33",
    ),
    "da2/da2_trace_1600.ncnn.bin": (
        None,
        748136644,
        "6bd2c96f93258cc5ff8821ce9db7780d331f11902aa4a4ab187f2fbcb1e0e5ba",
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
