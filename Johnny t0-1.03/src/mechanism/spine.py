"""SPINE orchestrator: the full per-row mechanism.

Public entry point::

    privatize(text: str, config) -> (new_text: str, token_log: list[dict])

By construction the mechanism receives ONLY the Text string. It never sees the
other CSV columns — there is simply no parameter for them. The per-row RNG is
read from ``config.rng`` (set by the wrapper; see SEED_POLICY.md).

Pipeline (see README "Module map"):
  1. carve protected (hate) spans out of the raw text  (lexicon, obfuscation-aware)
  2. segment the gaps into word/url/mention/hashtag/sep
  3. deterministic style normalisation                 (normalize)
  4. canonicalise protected tokens                      (canonicalize)
  5. classify the remainder (function / number / content) (salient)
  6. DP rewrite of content tokens via MLM + exponential mechanism (mlm + dp)
  7. reassemble + emit a complete per-token log
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from . import dp
from .canonicalize import canonicalize_protected
from .config import Config
from .lexicon import Lexicon, lexicon_from_config
from .mlm import make_backend
from .normalize import normalize_segments
from .salient import classify_segment
from .tokenize import (HASHTAG, MENTION, PROTECTED, SEP, URL, WORD, Segment,
                       reassemble, segment)


# ── resource caching (build the lexicon + MLM backend once per config) ──────

def get_resources(config: Config):
    """Build and cache (lexicon, backend, warning) on the config object."""
    cached = getattr(config, "_resources", None)
    if cached is not None:
        return cached
    if config.protection_enabled:
        lexicon, warning = lexicon_from_config(config)
    else:
        lexicon, warning = None, None
    backend = make_backend(config)
    resources = (lexicon, backend, warning)
    config._resources = resources
    return resources


def lexicon_warning(config: Config) -> Optional[str]:
    return get_resources(config)[2]


# ── segmentation with protected-span carving ───────────────────────────────

def _build_segments(text: str, lexicon: Optional[Lexicon]) -> List[Segment]:
    if not lexicon or not lexicon.loaded:
        return segment(text)
    spans = lexicon.find_protected_spans(text)
    if not spans:
        return segment(text)
    segments: List[Segment] = []
    idx = 0
    for start, end, canon in spans:
        if start > idx:
            segments.extend(segment(text[idx:start]))
        surface = text[start:end]
        segments.append(
            Segment(kind=PROTECTED, text=surface, original=surface, canonical=canon)
        )
        idx = end
    if idx < len(text):
        segments.extend(segment(text[idx:]))
    return segments


# ── the mechanism ───────────────────────────────────────────────────────────

def _log_entry(seg: Segment) -> dict:
    return {
        "original": seg.original,
        "normalized": seg.normalized,
        "action": seg.action,
        "replacement": seg.text,
        "epsilon": seg.epsilon,
        "token_class": seg.token_class,
        "reason": seg.reason,
    }


def privatize(text: str, config: Config) -> Tuple[str, List[dict]]:
    """Privatise a single Text string. See module docstring for the contract."""
    if config.rng is None:
        raise ValueError(
            "config.rng is not set. The wrapper must assign an independent "
            "per-row RNG before calling privatize (see SEED_POLICY.md)."
        )

    lexicon, backend, _warning = get_resources(config)
    segments = _build_segments(text, lexicon)

    # 3. style normalisation (mutates sep + word surfaces; logs reasons)
    normalize_segments(segments, config)

    # 4. canonicalise protected tokens
    for seg in segments:
        if seg.kind == PROTECTED:
            canonicalize_protected(seg)

    # 4b. optional classifier saliency (off by default; HF-gated, graceful)
    if config.saliency.enabled:
        from .saliency import apply_saliency
        apply_saliency(segments, config)

    # snapshot normalised surfaces for order-independent MLM context
    surfaces = [seg.text for seg in segments]
    prefix = [""] * (len(segments) + 1)
    for i, s in enumerate(surfaces):
        prefix[i + 1] = prefix[i] + s
    full = prefix[-1]

    clip = config.mlm.clip
    top_k = config.mlm.top_k
    include_original = config.mlm.include_original

    # 5/6. classify + DP rewrite content tokens
    for i, seg in enumerate(segments):
        if seg.kind == SEP or seg.kind == PROTECTED:
            continue
        if seg.kind in (URL, MENTION, HASHTAG):
            seg.token_class = seg.kind
            seg.action = "kept"
            seg.replacement = seg.text
            seg.epsilon = None
            seg.reason = f"{seg.kind} preserved verbatim"
            continue

        # WORD
        token_class = classify_segment(seg)
        seg.token_class = token_class
        eps = config.epsilon_for(token_class)

        rewritable = (
            token_class == "content"
            or (config.uniform_budget and token_class == "function_word")
        )
        if rewritable and eps is not None and eps > 0:
            left = prefix[i]
            right = full[len(prefix[i + 1]):]
            cands, raw_scores = backend.score(
                left, right, original=seg.text, top_k=top_k,
                include_original=include_original,
            )
            chosen, sel = dp.select(cands, raw_scores, eps, clip, config.rng)
            seg.text = chosen
            seg.replacement = chosen
            seg.action = "rewritten"
            seg.epsilon = float(eps)
            seg.reason = (
                f"dp-mlm exponential mechanism; backend={backend.name}; "
                f"candidates={len(cands)}; clip={clip}"
            )
        else:
            seg.replacement = seg.text
            seg.epsilon = None
            if seg.normalized != seg.original:
                seg.action = "normalised"
                if not seg.reason:
                    seg.reason = "style-normalised; kept"
            else:
                seg.action = "kept"
                seg.reason = f"{token_class}: kept (epsilon=skip)"

    new_text = reassemble(segments)
    token_log = [_log_entry(s) for s in segments if s.is_token]
    return new_text, token_log


def identity(text: str) -> Tuple[str, List[dict]]:
    """Identity transform: Text passes through unchanged, with a complete
    'kept' log so log-completeness holds in identity mode too."""
    segments = segment(text)
    log: List[dict] = []
    for seg in segments:
        if not seg.is_token:
            continue
        seg.normalized = seg.original
        seg.action = "kept"
        seg.replacement = seg.original
        seg.epsilon = None
        seg.token_class = seg.kind
        seg.reason = "identity mode: passthrough"
        log.append(_log_entry(seg))
    return text, log
