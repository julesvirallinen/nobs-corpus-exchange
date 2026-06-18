#!/usr/bin/env python3
"""Resolve cached Unsloth Qwen GGUF + mmproj paths for llama-server."""

from __future__ import annotations

import glob
import os
import sys

REPO = os.environ.get("QWEN_GGUF_REPO", "unsloth/Qwen3.5-0.8B-GGUF")
QUANT = os.environ.get("QWEN_GGUF_QUANT", "UD-Q4_K_XL")


def main() -> int:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Install huggingface_hub", file=sys.stderr)
        return 1

    root = snapshot_download(
        repo_id=REPO,
        allow_patterns=[f"*{QUANT}*", "*mmproj-F16*"],
    )
    model = sorted(glob.glob(os.path.join(root, f"*{QUANT}*.gguf")))
    mmproj = sorted(glob.glob(os.path.join(root, "mmproj*.gguf")))
    if not model:
        model = sorted(glob.glob(os.path.join(root, "*.gguf")))
        model = [p for p in model if "mmproj" not in os.path.basename(p)]
    if not model or not mmproj:
        print(f"GGUF files not found under {root}", file=sys.stderr)
        return 1
    print(model[0])
    print(mmproj[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
