"""Pre-download the open-weight models used by the `hf` paths (optional).

Requires requirements-hf.txt. Models are cached by HuggingFace under your HF
cache and are NOT committed. Everything runs on CPU.

    python scripts/setup_models.py

Downloads:
  * MLM for the DP rewrite (default: distilroberta-base)
  * the harness hate-classifier ensemble (open-weight English models)
"""

from __future__ import annotations

import sys

MLM = "distilroberta-base"
CLASSIFIERS = [
    "cardiffnlp/twitter-roberta-base-hate-latest",
    "Hate-speech-CNERG/dehatebert-mono-english",
]


def main() -> int:
    try:
        from transformers import (AutoModelForMaskedLM,
                                  AutoModelForSequenceClassification,
                                  AutoTokenizer)
    except ImportError:
        print("transformers not installed. Run: pip install -r requirements-hf.txt",
              file=sys.stderr)
        return 1

    print(f"caching MLM: {MLM}")
    AutoTokenizer.from_pretrained(MLM)
    AutoModelForMaskedLM.from_pretrained(MLM)

    for name in CLASSIFIERS:
        try:
            print(f"caching classifier: {name}")
            AutoTokenizer.from_pretrained(name)
            AutoModelForSequenceClassification.from_pretrained(name)
        except Exception as exc:  # network/model availability
            print(f"  WARN: could not cache {name}: {exc}", file=sys.stderr)

    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
