"""Backward-compatible shim for legacy `em_hsd.sensitivity` imports."""

from __future__ import annotations

from em_hsd.core.sensitivity import refined_delta_u, selection_sensitivity, word_count

__all__ = ["refined_delta_u", "selection_sensitivity", "word_count"]
