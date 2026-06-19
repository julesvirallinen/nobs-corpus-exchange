"""Backward-compatible shim for legacy `em_hsd.token_sanitize` imports."""

from __future__ import annotations

from em_hsd.core.sanitize import token_sanitize

__all__ = ["token_sanitize"]
