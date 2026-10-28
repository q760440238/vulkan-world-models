#!/usr/bin/env python3
"""Publish the engine + model card to Hugging Face.

Usage:
    $env:HF_TOKEN = "hf_..."
    python scripts/publish_hf.py --repo YOUR_HF_USER/hunyuanworld-vulkan

Weights are deliberately NOT uploaded: they are governed by the Tencent
HunyuanWorld-Mirror Community License and are downloaded separately.
"""
import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="e.g. user/hunyuanworld-vulkan")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        sys.exit("Set HF_TOKEN first (write token from https://huggingface.co/settings/tokens)")

    api = HfApi(token=token)
    api.create_repo(repo_id=args.repo, repo_type="model", private=args.private, exist_ok=True)

    # HF renders README.md as the model card; keep the GitHub README separate.
    api.upload_file(
        path_or_fileobj=str(ROOT / "MODEL_CARD.md"),
        path_in_repo="README.md",
        repo_id=args.repo,
        repo_type="model",
        commit_message="Add HunyuanWorld-Mirror Vulkan model card",
    )
    api.upload_folder(
        folder_path=str(ROOT),
        repo_id=args.repo,
        repo_type="model",
        allow_patterns=[
            "src/**",
            "docs/**",
            "scripts/download_weights.py",
            "Cargo.toml",
            "Cargo.lock",
            "LICENSE",
            "LICENSE-TENCENT-HUNYUANWORLD.txt",
            "NOTICE",
            "NOTICE-TENCENT.txt",
            "run.ps1",
        ],
        commit_message="Add Vulkan inference engine",
    )
    print(f"published https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
