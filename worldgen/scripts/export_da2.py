"""Export the patched DA-2 SphereViT to ncnn for the Vulkan runner.

This is a one-time developer step; end users only need the resulting
da2_trace_1024.ncnn.param/.bin. Requires torch, safetensors and pnnx.exe.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

CONFIG = {
    "inference": {"min_pixels": 580000, "max_pixels": 620000},
    "spherevit": {
        "vit_w_esphere": {
            "input_dims": [1024, 1024, 1024, 1024],
            "hidden_dim": 512,
            "num_heads": 8,
            "expansion": 4,
            "num_layers_head": [2, 2, 2],
            "dropout": 0.0,
            "layer_scale": 0.0001,
            "out_dim": 64,
            "kernel_size": 3,
            "num_prompt_blocks": 1,
            "use_norm": False,
        },
        "sphere": {"width": 1092, "height": 546, "hfov": 6.2832, "vfov": 3.1416},
    },
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--da2-src", required=True, help="DA-2/src directory with patches applied")
    ap.add_argument("--weights", required=True, help="models/da2/model.safetensors")
    ap.add_argument("--pnnx", required=True, help="pnnx.exe")
    ap.add_argument("--out", default="models/da2")
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=512)
    args = ap.parse_args()

    sys.path.insert(0, str(Path(args.da2_src).resolve()))
    from da2.model.spherevit import SphereViT

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model = SphereViT(CONFIG)
    model.load_state_dict(load_file(args.weights))
    model.eval()

    x = torch.randn(1, 3, args.height, args.width)
    with torch.no_grad():
        y = model(x)
    print("out", y.shape)
    pt = out / "da2_trace.pt"
    torch.jit.trace(model, x, strict=False).save(pt)

    subprocess.run(
        [args.pnnx, str(pt), f"inputshape=[1,3,{args.height},{args.width}]", "device=cpu"],
        check=True,
    )
    for suffix in (".ncnn.param", ".ncnn.bin"):
        # pnnx writes da2_trace.ncnn.param next to da2_trace.pt
        src = out / ("da2_trace" + suffix)
        shutil.copy2(src, out / ("da2_trace_%d%s" % (args.width, suffix)))
    print("written to", out)


if __name__ == "__main__":
    main()
