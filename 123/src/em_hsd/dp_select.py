"""Backward-compatible shim for legacy `em_hsd.dp_select` imports."""

from __future__ import annotations

from em_hsd.core.dp_select import select_rewrite

__all__ = ["select_rewrite"]
