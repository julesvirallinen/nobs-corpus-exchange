"""Backward-compatible shim for legacy `em_hsd.generative_proposer` imports."""

from __future__ import annotations

from em_hsd.layer4.proposer import (
    GenerativeProposer,
    MockProposer,
    TransformersQwenProposer,
    UnslothQwenProposer,
    get_proposer,
    make_proposer,
)

__all__ = [
    "GenerativeProposer",
    "MockProposer",
    "TransformersQwenProposer",
    "UnslothQwenProposer",
    "make_proposer",
    "get_proposer",
]
