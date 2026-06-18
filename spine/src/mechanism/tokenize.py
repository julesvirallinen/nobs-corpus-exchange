from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import regex as re

# Segment kinds.
SEP = "sep"
WORD = "word"
URL = "url"
MENTION = "mention"
HASHTAG = "hashtag"
PROTECTED = "protected"

TOKEN_KINDS = (WORD, URL, MENTION, HASHTAG, PROTECTED)

# Alternation; leftmost-first wins on ties (URL before WORD so "www.x" is a URL).
# WORD = run of unicode alphanumerics (no underscore) with internal '/-.
_TOKEN_RE = re.compile(
    r"""(?P<url>(?:https?://|www\.)\S+)
      | (?P<mention>@\w+)
      | (?P<hashtag>\#\w+)
      | (?P<word>[^\W_]+(?:['’\-][^\W_]+)*)
    """,
    re.VERBOSE | re.UNICODE,
)


@dataclass
class Segment:
    kind: str
    text: str                       # current surface form (mutated in place)
    original: str                   # original surface (immutable)
    canonical: Optional[str] = None  # protected: the canonical lexicon term
    # filled in by later pipeline stages, used for the per-token log:
    normalized: Optional[str] = None  # surface after the normalisation stage
    action: Optional[str] = None     # normalised | protected | rewritten | kept
    replacement: Optional[str] = None
    epsilon: Optional[float] = None
    reason: Optional[str] = None
    token_class: Optional[str] = None

    @property
    def is_token(self) -> bool:
        return self.kind != SEP


def segment(text: str) -> List[Segment]:    segments: List[Segment] = []
    idx = 0
    for m in _TOKEN_RE.finditer(text):
        start, end = m.start(), m.end()
        if start > idx:
            gap = text[idx:start]
            segments.append(Segment(kind=SEP, text=gap, original=gap))
        kind = m.lastgroup
        tok = m.group()
        segments.append(Segment(kind=kind, text=tok, original=tok))
        idx = end
    if idx < len(text):
        tail = text[idx:]
        segments.append(Segment(kind=SEP, text=tail, original=tail))
    return segments


def reassemble(segments: List[Segment]) -> str:    return "".join(s.text for s in segments)
