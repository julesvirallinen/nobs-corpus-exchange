"""Backward-compatible shim for legacy `em_hsd.constraints` imports."""

from __future__ import annotations

from em_hsd.layer4.filter import (
    FilterBatch,
    FilterResult,
    extract_protected_terms,
    filter_candidates,
    normalized_edit_ratio,
    protected_skeletons,
    spans_preserved,
)

__all__ = [
    "FilterBatch",
    "FilterResult",
    "extract_protected_terms",
    "filter_candidates",
    "normalized_edit_ratio",
    "protected_skeletons",
    "spans_preserved",
]
