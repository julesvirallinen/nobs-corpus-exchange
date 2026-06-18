"""Pre-download models for EM-HSD 2.0 + SPINE harness."""

from __future__ import annotations

import sys
from pathlib import Path

MLM = "distilroberta-base"
HATE = "unitary/unbiased-toxic-roberta"
EMBED = "sentence-transformers/all-MiniLM-L6-v2"
QWEN_GGUF = "unsloth/Qwen3.5-0.8B-GGUF"
QWEN_SAFETENSORS = "Qwen/Qwen3.5-0.8B"
QWEN_QUANT = "UD-Q4_K_XL"


def main() -> int:
    try:
        from transformers import (
            AutoModelForMaskedLM,
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )
    except ImportError:
        print("Install hf extras: pip install -e '.[hf]'", file=sys.stderr)
        return 1

    print(f"caching MLM: {MLM}")
    AutoTokenizer.from_pretrained(MLM)
    AutoModelForMaskedLM.from_pretrained(MLM)

    print(f"caching hate classifier: {HATE}")
    AutoTokenizer.from_pretrained(HATE)
    AutoModelForSequenceClassification.from_pretrained(HATE)

    try:
        from sentence_transformers import SentenceTransformer
        print(f"caching embedder: {EMBED}")
        SentenceTransformer(EMBED)
    except ImportError:
        print("WARN: sentence-transformers not installed", file=sys.stderr)

    try:
        from huggingface_hub import snapshot_download
        print(f"caching GGUF for llama.cpp: {QWEN_GGUF} ({QWEN_QUANT})")
        snapshot_download(
            repo_id=QWEN_GGUF,
            allow_patterns=[f"*{QWEN_QUANT}*", "*mmproj-F16*"],
        )
    except ImportError:
        print("WARN: huggingface_hub not installed — skip GGUF download", file=sys.stderr)
    except Exception as exc:
        print(f"WARN: could not cache GGUF: {exc}", file=sys.stderr)

    try:
        from unsloth import FastLanguageModel
        print(f"caching safetensors for Unsloth backend: {QWEN_SAFETENSORS}")
        FastLanguageModel.from_pretrained(
            model_name=QWEN_SAFETENSORS,
            max_seq_length=2048,
            load_in_4bit=True,
            fast_inference=False,
        )
    except ImportError:
        print("WARN: unsloth not installed — skip safetensors download", file=sys.stderr)
    except Exception as exc:
        print(f"WARN: could not cache safetensors Qwen: {exc}", file=sys.stderr)

    print("done.")
    print("For GGUF inference: bash scripts/start_llama_server.sh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
