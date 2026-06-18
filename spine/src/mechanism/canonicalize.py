from __future__ import annotations

from .tokenize import Segment


def canonical_form(term: str) -> str:    return (term or "").strip().lower()


def canonicalize_protected(seg: Segment) -> None:
    """Replace a protected segment's surface with its canonical form, in place,
    and record the action for the per-token log."""
    canon = canonical_form(seg.canonical if seg.canonical else seg.original)
    changed = canon != seg.original
    seg.normalized = canon
    seg.text = canon
    seg.replacement = canon
    seg.token_class = "protected"
    seg.action = "protected+canonicalised"
    seg.epsilon = None
    seg.reason = (
        "hate-lexicon match; canonicalised from idiosyncratic surface"
        if changed
        else "hate-lexicon match; already canonical"
    )
