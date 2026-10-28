"""DA-2 depth inference through ncnn Vulkan."""

import os
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
MODELS = Path(os.environ.get("WORLDGEN_MODELS", ROOT / "models"))
EXE = Path(os.environ.get("DA2_NCNN_EXE", ROOT / "tools/da2-ncnn/da2ncnn.exe"))
PARAM = MODELS / "da2/da2_trace_1024.ncnn.param"
BIN = MODELS / "da2/da2_trace_1024.ncnn.bin"
NCNN_DLL = Path(os.environ.get("NCNN_DLL_DIR", ROOT / "tools/ncnn/src"))


def depth(pano: Image.Image, width: int = 1024, height: int = 512) -> np.ndarray:
    """Run DA-2 at its fixed 1024x512 trace size; resample other sizes."""
    rw, rh = 1024, 512
    arr = np.asarray(pano.convert("RGB").resize((rw, rh))).astype("<f4") / 255.0
    nchw = np.ascontiguousarray(arr.transpose(2, 0, 1))
    tmp_in = ROOT / "output" / "_da2_in.f32"
    tmp_out = ROOT / "output" / "_da2_out.f32"
    tmp_in.parent.mkdir(parents=True, exist_ok=True)
    nchw.tofile(tmp_in)
    env = dict(os.environ)
    env["PATH"] = str(NCNN_DLL) + os.pathsep + env.get("PATH", "")
    subprocess.run(
        [str(EXE), str(PARAM), str(BIN), str(tmp_in), str(tmp_out),
         str(rw), str(rh), "vulkan"],
        check=True,
        env=env,
    )
    d = np.fromfile(tmp_out, "<f4").reshape(rh, rw).astype(np.float32)
    if width != rw or height != rh:
        d = np.asarray(
            Image.fromarray(d, mode="F").resize((width, height), Image.BILINEAR)
        ).astype(np.float32)
    d = d / d.max() * 20.0
    return d
