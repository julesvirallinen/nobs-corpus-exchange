"""Backward-compatible shim for legacy `em_hsd.embedding` imports."""

from __future__ import annotations

from em_hsd.core.embedding import (
    SentenceTransformerEncoder,
    SimpleEncoder,
    get_encoder,
    make_encoder,
)

__all__ = ["SimpleEncoder", "SentenceTransformerEncoder", "make_encoder", "get_encoder"]
