#!/usr/bin/env python3
"""Download the official HunyuanWorld-Mirror weights and build weights.idx.

The weights are NOT part of this repository. They are governed by the
Tencent HunyuanWorld-Mirror Community License Agreement; see
LICENSE-TENCENT-HUNYUANWORLD.txt and NOTICE.
"""
import json
import os
import struct
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "weights"
REPO = "tencent/HunyuanWorld-Mirror"


def build_index(model: Path) -> None:
    with open(model, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    data_start = 8 + n
    lines = []
    total = 0
    for name, v in header.items():
        if name == "__metadata__":
            continue
        if v["dtype"] != "F32":
            raise SystemExit(f"unsupported dtype {v['dtype']} for {name}")
        start, end = v["data_offsets"]
        shape = v["shape"]
        lines.append(
            "{} {} {} {} {}".format(
                name, data_start + start, end - start, len(shape), " ".join(map(str, shape))
            )
        )
        total += end - start
    (WEIGHTS / "weights.idx").write_text("\n".join(lines) + "\n")
    print(f"wrote {WEIGHTS / 'weights.idx'} ({len(lines)} tensors, {total / 1e9:.2f} GB)")


def main() -> None:
    WEIGHTS.mkdir(exist_ok=True)
    for name in ("config.json", "model.safetensors"):
        print(f"downloading {REPO}/{name} ...")
        hf_hub_download(repo_id=REPO, filename=name, local_dir=str(WEIGHTS))
    build_index(WEIGHTS / "model.safetensors")


if __name__ == "__main__":
    main()
