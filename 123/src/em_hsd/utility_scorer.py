"""Backward-compatible shim for legacy `em_hsd.utility_scorer` imports."""

from __future__ import annotations

from em_hsd.layer4.scorer import (
    HFToxicityScorer,
    _label_index,
    _score_from_logits,
    get_scorer,
    make_scorer,
)

__all__ = [
    "HFToxicityScorer",
    "_label_index",
    "_score_from_logits",
    "get_scorer",
    "make_scorer",
]
