"""Paraphrase prompt templates (Layer 4 §4.3, §15.4 — manual Π₀, no GEPA)."""

from __future__ import annotations

from typing import Literal, Sequence

from .config import EmHsdConfig

PromptProfile = Literal["hate", "neutral"]

_EMPTY_PROTECTED_HINT = (
    "(none listed — but do NOT remove, euphemize, or censor existing insults, "
    "slurs, or hostile language already in the post)"
)

_PROMPT_HATE_VARIANTS = (
    """You are anonymizing a social-media post for a moderation research dataset.

Privacy (p): Remove author-identifying style so the writer cannot be re-identified.
Change habitual openers, distinctive phrasing, word order, punctuation habits, and dialect tics.
Do NOT preserve the author's voice.

Utility (u): Preserve hate/offensive meaning for content moderation.
- Keep hostility, insults, slurs, and targeted offense at the SAME strength.
- Do NOT euphemize, censor, sanitize, or "clean up" offensive language.
- KEEP these terms unchanged in meaning (same or canonical form): {protected_list}
- Preserve who/what is being attacked; do not swap targets.

Constraints:
- Fix obvious grammar breaks from prior sanitization without removing offense.
- Do not invent new facts, names, places, events, or slurs.
- Similar length (±25%).
- Output ONLY the rewritten post, no explanation.

Post: {text}""",
    """Rewrite the post below for privacy while keeping moderation utility.

PRIVACY: Break stylometric fingerprint (openers, tics, rare phrasing, punctuation habits).
UTILITY: Keep insults and slurs at full strength for hate-speech detection.
KEEP unchanged in meaning: {protected_list}
Do not soften, remove, or replace offensive terms with neutral synonyms.
Repair garbled grammar only; do not add commentary or new attacks.
Length within ±25%. Output only the rewritten post.

Post: {text}""",
)

_PROMPT_NEUTRAL_VARIANTS = (
    """You are anonymizing a social-media post for a privacy research dataset.

Privacy (p): Strongly change author-identifying style (openers, phrasing, syntax, punctuation).
Break stylometric patterns so the writer cannot be re-identified.

Utility (u): Preserve the topic, argument, and general stance of the post.
Do not invent new facts, names, or events.
KEEP these terms if present: {protected_list}

Constraints:
- Fix grammar if the post was partially sanitized.
- Similar length (±25%).
- Output ONLY the rewritten post.

Post: {text}""",
    """Rewrite this post with different wording and style to protect author identity.

Change distinctive openers, sentence rhythm, and word choices.
Preserve the main topic and opinion direction.
Terms to keep if present: {protected_list}
Do not add commentary. Output only the rewritten post.

Post: {text}""",
)


def format_protected_list(protected_terms: Sequence[str]) -> str:
    terms = [t.strip() for t in protected_terms if t and str(t).strip()]
    if not terms:
        return _EMPTY_PROTECTED_HINT
    return ", ".join(sorted(set(terms)))


def resolve_prompt_profile(
    config: EmHsdConfig,
    p_hate_original: float | None,
) -> PromptProfile:
    mode = config.generation.prompt_profile.lower()
    if mode == "hate":
        return "hate"
    if mode == "neutral":
        return "neutral"
    threshold = config.generation.hate_p_threshold
    if p_hate_original is not None and p_hate_original >= threshold:
        return "hate"
    return "neutral"


def build_paraphrase_prompt(
    config: EmHsdConfig,
    protected_terms: Sequence[str],
    text: str,
    *,
    profile: PromptProfile,
    variant_idx: int = 0,
) -> str:
    protected_list = format_protected_list(protected_terms)
    variants = _PROMPT_HATE_VARIANTS if profile == "hate" else _PROMPT_NEUTRAL_VARIANTS
    template = variants[variant_idx % len(variants)]
    return template.format(protected_list=protected_list, text=text)


def n_jitter_variants(profile: PromptProfile) -> int:
    if profile == "hate":
        return len(_PROMPT_HATE_VARIANTS)
    return len(_PROMPT_NEUTRAL_VARIANTS)
