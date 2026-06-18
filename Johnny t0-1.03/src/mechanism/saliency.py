"""Optional classifier-based saliency (config flag ``saliency.enabled``).

Marks additional hate-carrying *content* tokens that the lexicon missed, by
occlusion: drop each content token and measure the fall in a hate classifier's
score; tokens whose removal drops the score by more than ``threshold`` are
protected in place (kept, not DP-rewritten) so genuine hate signal is preserved.

This is OFF by default and only does anything when the HF extra is installed.
It loads its OWN classifier (the mechanism never imports the harness). It is
slower (one forward pass per content token) and is intended as a refinement, not
a requirement. Failures degrade gracefully: a warning, and the lexicon-only
behaviour is used.
"""

from __future__ import annotations

import threading
from typing import List, Optional

from .config import Config
from .tokenize import PROTECTED, WORD, Segment
from .canonicalize import canonicalize_protected

_HATE_KEYS = ("hate", "offensive", "abusive", "toxic", "hateful", "label_1")

# Process-level cache so the same checkpoint is not loaded twice.
_oc_lock = threading.Lock()
_oc_cache: dict = {}  # model_name -> OcclusionSaliency


class OcclusionSaliency:
    def __init__(self, model_name: str):
        import torch  # noqa: F401
        from transformers import (AutoModelForSequenceClassification,
                                   AutoTokenizer)

        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        id2label = {int(k): v.lower() for k, v in self.model.config.id2label.items()}
        self.hate_ids = [i for i, lab in id2label.items()
                         if any(k in lab for k in _HATE_KEYS)]
        if not self.hate_ids:  # fall back to the last label
            self.hate_ids = [max(id2label)]

    def hate_prob(self, text: str) -> float:
        torch = self._torch
        if not text.strip():
            return 0.0
        enc = self.tokenizer(text, return_tensors="pt", truncation=True,
                             max_length=256)
        with torch.no_grad():
            logits = self.model(**enc).logits[0]
        probs = torch.softmax(logits, dim=-1)
        return float(sum(probs[i] for i in self.hate_ids))


def _get_occlusion_saliency(model_name: str) -> "OcclusionSaliency":
    """Return a cached OcclusionSaliency, loading once per model_name."""
    if model_name in _oc_cache:
        return _oc_cache[model_name]
    with _oc_lock:
        if model_name not in _oc_cache:
            _oc_cache[model_name] = OcclusionSaliency(model_name)
    return _oc_cache[model_name]


def apply_saliency(segments: List[Segment], config: Config) -> Optional[str]:
    """Mark salient content tokens as protected-in-place. Returns a warning
    string on failure (and leaves segments unchanged), else None."""
    if not config.saliency.enabled:
        return None
    try:
        scorer = _get_occlusion_saliency(config.saliency.model)
    except Exception as exc:  # pragma: no cover - depends on optional HF extra
        return f"saliency disabled: could not load {config.saliency.model!r}: {exc}"

    word_positions = [i for i, s in enumerate(segments) if s.kind == WORD]
    surfaces = [s.text for s in segments]
    base = scorer.hate_prob("".join(surfaces))
    threshold = config.saliency.threshold
    for i in word_positions:
        occluded = surfaces.copy()
        occluded[i] = " "
        drop = base - scorer.hate_prob("".join(occluded))
        if drop >= threshold:
            seg = segments[i]
            seg.kind = PROTECTED
            seg.canonical = seg.text
            canonicalize_protected(seg)
            seg.reason = f"classifier-salient (Δhate={drop:.3f}); protected in place"
    return None
