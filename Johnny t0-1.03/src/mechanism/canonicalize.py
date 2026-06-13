"""Canonicalisation of protected (hate-carrying) tokens.

A protected token is kept (the classifier must still see the slur *category*),
but it is mapped to its canonical lexicon surface form so the attacker never
sees *this writer's* idiosyncratic spelling (de-leeted, de-spaced, de-elongated,
lowercased). This is empirical obfuscation, NOT a formal privacy guarantee.
"""

from __future__ import annotations

from .tokenize import Segment


def canonical_form(term: str) -> str:
    """The canonical surface form of a lexicon term."""
    return (term or "").strip().lower()


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
