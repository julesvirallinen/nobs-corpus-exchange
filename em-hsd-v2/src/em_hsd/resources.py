"""Cached resource helpers for EM-HSD pipeline."""

from __future__ import annotations

from typing import List, Tuple

from . import spine_bootstrap  # noqa: F401
from .config import EmHsdConfig
from .constraints import protected_skeletons
from mechanism.spine import get_resources


def protected_canonicals(text: str, config: EmHsdConfig) -> Tuple[List[str], List[str]]:
    """Return (canonical terms, skeletons) found in text."""
    lexicon, _, _ = get_resources(config.spine)
    if not lexicon or not lexicon.loaded:
        return [], []
    spans = lexicon.find_protected_spans(text)
    canonicals = sorted({canon for _s, _e, canon in spans})
    return canonicals, protected_skeletons(canonicals)


def init_spine_resources(config: EmHsdConfig) -> None:
    """Fail early if SPINE backends cannot load."""
    get_resources(config.spine)
