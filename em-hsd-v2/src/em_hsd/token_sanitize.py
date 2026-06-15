"""Phase 1a: selective ε₁ token sanitization → x^priv."""

from __future__ import annotations

from typing import List, Tuple

from . import spine_bootstrap  # noqa: F401
from .config import EmHsdConfig
from mechanism import dp
from mechanism.canonicalize import canonicalize_protected
from mechanism.normalize import normalize_segments
from mechanism.salient import classify_segment
from mechanism.spine import _build_segments, get_resources
from mechanism.tokenize import HASHTAG, MENTION, PROTECTED, SEP, URL, WORD, reassemble


def _log_entry(seg) -> dict:
    return {
        "original": seg.original,
        "normalized": seg.normalized,
        "action": seg.action,
        "replacement": seg.text,
        "epsilon": seg.epsilon,
        "token_class": seg.token_class,
        "reason": seg.reason,
    }


def token_sanitize(text: str, config: EmHsdConfig, epsilon_1: float) -> Tuple[str, List[dict]]:
    """Run normalize → protect → ε₁ DP on content tokens only."""
    spine = config.spine
    if spine.rng is None:
        raise ValueError("config.spine.rng must be set before token_sanitize")

    lexicon, backend, _ = get_resources(spine)
    segments = _build_segments(text, lexicon)
    normalize_segments(segments, spine)

    for seg in segments:
        if seg.kind == PROTECTED:
            canonicalize_protected(seg)

    surfaces = [seg.text for seg in segments]
    prefix = [""] * (len(segments) + 1)
    for i, s in enumerate(surfaces):
        prefix[i + 1] = prefix[i] + s
    full = prefix[-1]

    clip = spine.mlm.clip
    top_k = spine.mlm.top_k
    include_original = spine.mlm.include_original

    for i, seg in enumerate(segments):
        if seg.kind in (SEP, PROTECTED):
            continue
        if seg.kind in (URL, MENTION, HASHTAG):
            seg.token_class = seg.kind
            seg.action = "kept"
            seg.replacement = seg.text
            seg.epsilon = None
            seg.reason = f"{seg.kind} preserved verbatim"
            continue

        token_class = classify_segment(seg)
        seg.token_class = token_class
        if token_class != "content" or epsilon_1 <= 0:
            seg.replacement = seg.text
            seg.epsilon = None
            if seg.normalized != seg.original:
                seg.action = "normalised"
                seg.reason = seg.reason or "style-normalised; kept"
            else:
                seg.action = "kept"
                seg.reason = f"{token_class}: kept (epsilon_1 skip)"
            continue

        left = prefix[i]
        right = full[len(prefix[i + 1]):]
        cands, raw_scores = backend.score(
            left, right, original=seg.text, top_k=top_k,
            include_original=include_original,
        )
        chosen, _sel = dp.select(cands, raw_scores, epsilon_1, clip, spine.rng)
        seg.text = chosen
        seg.replacement = chosen
        seg.action = "rewritten"
        seg.epsilon = float(epsilon_1)
        seg.reason = (
            f"em-hsd token sanitize; backend={backend.name}; "
            f"candidates={len(cands)}; epsilon_1={epsilon_1}"
        )

    x_priv = reassemble(segments)
    token_log = [_log_entry(s) for s in segments if s.is_token]
    return x_priv, token_log
