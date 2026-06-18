#!/usr/bin/env python3
"""Download Unsloth Qwen3.5-0.8B-GGUF + mmproj for llama.cpp (not FastLanguageModel)."""

from __future__ import annotations

import sys

QWEN_GGUF = "unsloth/Qwen3.5-0.8B-GGUF"
DEFAULT_QUANT = "UD-Q4_K_XL"


def main() -> int:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Install: pip install huggingface_hub", file=sys.stderr)
        return 1

    print(f"Downloading {QWEN_GGUF} ({DEFAULT_QUANT} + mmproj)...")
    path = snapshot_download(
        repo_id=QWEN_GGUF,
        allow_patterns=[f"*{DEFAULT_QUANT}*", "*mmproj-F16*"],
    )
    print(f"Cached at: {path}")
    print("Start server: bash scripts/start_llama_server.sh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
