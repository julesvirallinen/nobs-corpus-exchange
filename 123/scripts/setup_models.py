"""Pre-download models for EM-HSD 2.0 + SPINE harness."""

from __future__ import annotations

import sys

MLM = "distilroberta-base"
# Multi-label English hate classifier: toxicity (flag), severity-type labels,
# and identity/target-group labels (drives the Corpus Exchange review categories).
HATE = "unitary/unbiased-toxic-roberta"
EMBED = "sentence-transformers/all-MiniLM-L6-v2"
QWEN = "unsloth/Qwen3.5-2B"


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
        from unsloth import FastLanguageModel
        print(f"caching Qwen via Unsloth: {QWEN}")
        FastLanguageModel.from_pretrained(
            model_name=QWEN,
            max_seq_length=2048,
            load_in_4bit=True,
        )
    except ImportError:
        print("WARN: unsloth not installed — skip Qwen download", file=sys.stderr)
    except Exception as exc:
        print(f"WARN: could not cache Qwen: {exc}", file=sys.stderr)

    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
