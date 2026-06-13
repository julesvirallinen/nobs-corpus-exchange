"""Masked-language-model backends that supply candidate logits for the DP step.

The DP exponential mechanism (dp.py) is identical no matter where the logits
come from; only the *source* differs:

* ``HFMaskedLM`` -- a real open-weight masked LM (default: distilroberta-base)
  from HuggingFace. This is the intended evaluation path. Requires
  ``requirements-hf.txt`` and a one-time model download (CPU is fine).

* ``HashMLM`` -- a deterministic, dependency-free stand-in. It produces
  reproducible, context-dependent pseudo-logits over a fixed neutral vocabulary
  so the entire DP machinery (clipping, exponential mechanism, per-row RNG) is
  exercised with NO torch and NO downloads. It is NOT a language model and has
  no semantic quality; it exists for the test suite, plumbing, and the offline
  verify.sh fallback. Never ship a submission produced with the hash backend.

Both return RAW (unclipped) logits; clipping happens in dp.py.
"""

from __future__ import annotations

import hashlib
from typing import List, Sequence, Tuple

import numpy as np

# A small, deliberately neutral vocabulary for the deterministic backend.
_HASH_VOCAB = [
    "the", "a", "this", "that", "people", "person", "thing", "things", "time",
    "way", "day", "world", "place", "man", "woman", "group", "country", "city",
    "good", "bad", "great", "small", "big", "new", "old", "right", "wrong",
    "happy", "sad", "angry", "calm", "real", "fake", "true", "false", "hard",
    "easy", "common", "rare", "simple", "strange", "clear", "odd", "kind",
    "go", "going", "went", "come", "came", "make", "made", "think", "thought",
    "know", "knew", "see", "saw", "say", "said", "feel", "felt", "want", "need",
    "like", "love", "hate", "find", "found", "give", "take", "use", "used",
    "very", "really", "quite", "so", "too", "just", "even", "still", "much",
    "always", "never", "often", "maybe", "perhaps", "here", "there", "now",
    "today", "again", "more", "less", "most", "some", "many", "few", "every",
    "and", "but", "or", "because", "though", "while", "when", "where", "what",
    "talk", "talks", "post", "posts", "comment", "story", "idea", "point",
    "issue", "problem", "view", "opinion", "fact", "reason", "matter",
]


class HashMLM:
    """Deterministic, dependency-free pseudo-MLM. See module docstring."""

    name = "hash"

    def __init__(self, spread: float = 6.0, vocab: Sequence[str] = None):
        self.spread = float(spread)
        self.vocab = list(vocab) if vocab is not None else list(_HASH_VOCAB)

    def _logit(self, ctx: str, word: str) -> float:
        h = hashlib.sha256((ctx + "\x1f" + word).encode("utf-8")).digest()
        frac = int.from_bytes(h[:4], "big") / 2 ** 32  # [0, 1)
        return (frac - 0.5) * 2.0 * self.spread          # [-spread, spread)

    def score(self, left: str, right: str, original: str, top_k: int,
              include_original: bool) -> Tuple[List[str], np.ndarray]:
        # Tiny, order-stable context key from the immediate neighbours.
        lwords = left.split()
        rwords = right.split()
        ctx = (lwords[-1] if lwords else "") + "|" + (rwords[0] if rwords else "")
        scored = [(w, self._logit(ctx, w)) for w in self.vocab]
        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[: max(1, top_k)]
        cands = [w for w, _ in top]
        scores = [s for _, s in top]
        if include_original and original not in cands:
            cands.append(original)
            scores.append(self._logit(ctx, original))
        return cands, np.asarray(scores, dtype=np.float64)


class HFMaskedLM:
    """Real masked-LM backend (transformers + torch). Lazily imported."""

    name = "hf"

    def __init__(self, model_name: str = "distilroberta-base", max_length: int = 256):
        import torch  # noqa: F401  (lazy: only needed for this backend)
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        self._torch = torch
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.model.eval()
        self.mask_token = self.tokenizer.mask_token
        if self.mask_token is None:
            raise ValueError(
                f"{model_name!r} has no mask token; pick an AutoModelForMaskedLM "
                "checkpoint (e.g. distilroberta-base, bert-base-uncased)."
            )

    def score(self, left: str, right: str, original: str, top_k: int,
              include_original: bool) -> Tuple[List[str], np.ndarray]:
        torch = self._torch
        text = f"{left} {self.mask_token} {right}".strip()
        enc = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=self.max_length
        )
        with torch.no_grad():
            logits = self.model(**enc).logits[0]  # (seq_len, vocab)
        ids = enc["input_ids"][0]
        mask_positions = (ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
        if len(mask_positions) == 0:
            # Mask got truncated away; fall back to keeping the original.
            return [original], np.asarray([1.0], dtype=np.float64)
        pos = int(mask_positions[0])
        row = logits[pos]
        k = min(max(1, top_k), row.shape[-1])
        top = torch.topk(row, k)
        cands: List[str] = []
        scores: List[float] = []
        seen = set()
        for score_val, idx in zip(top.values.tolist(), top.indices.tolist()):
            # decode() maps byte-level BPE back to real text; require an ASCII
            # alphabetic whole word so mojibake subword fragments are dropped.
            word = self.tokenizer.decode([idx]).strip()
            if not word or not word.isascii() or not word.isalpha():
                continue
            low = word.lower()
            if low in seen:
                continue
            seen.add(low)
            cands.append(low)
            scores.append(float(score_val))
        if include_original and original.lower() not in seen:
            cands.append(original.lower())
            # Use the original's own logit if it is a single token, else the
            # minimum top-k score so it remains a low-but-present option.
            oid = self.tokenizer.encode(" " + original, add_special_tokens=False)
            if len(oid) == 1:
                scores.append(float(row[oid[0]]))
            else:
                scores.append(float(min(scores)) if scores else 0.0)
        if not cands:
            return [original], np.asarray([1.0], dtype=np.float64)
        return cands, np.asarray(scores, dtype=np.float64)


def make_backend(cfg):
    """Construct the MLM backend named by ``cfg.mlm.backend``."""
    backend = cfg.mlm.backend.lower()
    if backend == "hash":
        return HashMLM()
    if backend == "hf":
        return HFMaskedLM(cfg.mlm.model)
    raise ValueError(f"Unknown mlm.backend {backend!r} (expected 'hf' or 'hash')")
