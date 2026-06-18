"""Cached resource helpers for EM-HSD pipeline."""

from __future__ import annotations

from typing import Any, List, Sequence, Tuple

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


def protected_from_token_log(token_log: Sequence[Any]) -> List[str]:
    """Extract Phase 1a protected surfaces (lexicon + saliency) from token log."""
    terms: List[str] = []
    for entry in token_log or []:
        if not isinstance(entry, dict):
            continue
        if entry.get("action") != "protected+canonicalised":
            continue
        surface = entry.get("replacement") or entry.get("normalized") or entry.get("original")
        if surface and str(surface).strip():
            terms.append(str(surface).strip().lower())
    return sorted(set(terms))


def merge_protected_terms(
    lexicon_terms: Sequence[str],
    token_log_terms: Sequence[str],
) -> Tuple[List[str], List[str]]:
    canonicals = sorted({str(t).strip().lower() for t in lexicon_terms if t and str(t).strip()})
    canonicals = sorted(set(canonicals) | {str(t).strip().lower() for t in token_log_terms if t})
    return canonicals, protected_skeletons(canonicals)


def init_spine_resources(config: EmHsdConfig) -> None:
    """Fail early if SPINE backends cannot load."""
    get_resources(config.spine)
