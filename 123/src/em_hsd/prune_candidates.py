"""Backward-compatible shim for legacy `em_hsd.prune_candidates` imports."""

from __future__ import annotations

from em_hsd.layer4.prune import prune_candidates

__all__ = ["prune_candidates"]
