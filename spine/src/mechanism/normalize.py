from __future__ import annotations

from typing import List, Tuple

import regex as re

from .config import Config
from .tokenize import SEP, WORD, Segment, reassemble, segment

# Emoji and pictographic symbols (broad, deliberately conservative on text).
# Extended_Pictographic covers most emoji; add regional-indicator flags, the
# emoji variation selector (FE0F) and the zero-width joiner (200D).
_EMOJI_RE = re.compile(
    "[\\p{Extended_Pictographic}\U0001F1E6-\U0001F1FF️‍]",
    re.UNICODE,
)

# Smart punctuation -> ASCII.
_PUNCT_MAP = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"',
    "–": "-", "—": "-", "‒": "-", "―": "-",
    "…": "...",
    " ": " ", " ": " ", " ": " ", " ": " ",
}
_PUNCT_TABLE = {ord(k): v for k, v in _PUNCT_MAP.items()}

_REPEAT_PUNCT_RE = re.compile(r"([^\w\s])\1+", re.UNICODE)
_WS_RE = re.compile(r"\s+", re.UNICODE)
_ELONG_RE = re.compile(r"(.)\1{2,}", re.UNICODE)  # 3+ repeats -> 1

# Small, conservative misspelling / informal-spelling map (keys are the
# lowercased, de-elongated surface). Kept intentionally short.
_MISSPELL = {
    "u": "you", "ur": "your", "r": "are", "n": "and",
    "im": "i'm", "dont": "don't", "cant": "can't", "wont": "won't",
    "didnt": "didn't", "doesnt": "doesn't", "isnt": "isn't", "wasnt": "wasn't",
    "arent": "aren't", "couldnt": "couldn't", "shouldnt": "shouldn't",
    "wouldnt": "wouldn't", "youre": "you're", "theyre": "they're",
    "gonna": "going to", "wanna": "want to", "gotta": "got to",
    "cuz": "because", "becuz": "because", "thru": "through", "tho": "though",
    "pls": "please", "plz": "please", "thx": "thanks", "ppl": "people",
}


def normalize_sep(text: str, cfg: Config) -> str:
    n = text
    if cfg.normalization.strip_emoji:
        n = _EMOJI_RE.sub("", n)
    if cfg.normalization.normalize_punctuation:
        n = n.translate(_PUNCT_TABLE)
        n = _REPEAT_PUNCT_RE.sub(r"\1", n)
    if cfg.normalization.collapse_whitespace:
        n = _WS_RE.sub(" ", n)
    return n


def normalize_word(text: str, cfg: Config) -> Tuple[str, List[str]]:    reasons: List[str] = []
    n = text
    if cfg.normalization.lowercase:
        low = n.lower()
        if low != n:
            reasons.append("lowercased")
        n = low
    if cfg.normalization.repair_elongation:
        de = _ELONG_RE.sub(r"\1", n)
        if de != n:
            reasons.append("de-elongated")
        n = de
    if cfg.normalization.fix_misspellings and n in _MISSPELL:
        fixed = _MISSPELL[n]
        if fixed != n:
            reasons.append("misspelling")
        n = fixed
    return n, reasons


def normalize_segments(segments: List[Segment], cfg: Config) -> None:
    """Mutate segments in place: set ``.text`` to the normalised surface and
    record the normalised snapshot + any change reason. URL/mention/hashtag and
    already-protected segments are left untouched here."""
    for seg in segments:
        if seg.kind == SEP:
            seg.text = normalize_sep(seg.text, cfg)
            seg.normalized = seg.text
        elif seg.kind == WORD:
            new, reasons = normalize_word(seg.text, cfg)
            seg.text = new
            seg.normalized = new
            if reasons:
                seg.reason = "normalised:" + "+".join(reasons)
        else:
            # url / mention / hashtag / protected: snapshot, no change here.
            seg.normalized = seg.text


def normalize_text(text: str, cfg: Config) -> str:
    """Convenience: normalise a raw string (no lexicon protection). Used by the
    normaliser idempotency/determinism tests."""
    segs = segment(text)
    normalize_segments(segs, cfg)
    return reassemble(segs)
