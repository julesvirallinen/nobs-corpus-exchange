"""Hate / salient term lexicon with obfuscation-aware matching.

A term such as ``zibber`` is compiled into a regex that also matches leetspeak
(``z1bb3r``), spacing (``z i b b e r``) and character repetition (``ziiibber``).
Matching yields ``(start, end, canonical)`` spans on the *raw* text so the
orchestrator can carve them out atomically and replace each with its canonical
lexicon form (de-leeted, de-spaced, de-elongated, lowercased).

Real lexicons are downloaded by scripts/setup_lexicons.py and are NOT committed
(they may contain slurs, and the repo is public). When ``source: real`` but the
file is absent, the lexicon loads empty and reports it; the pipeline still runs
and protects nothing (logged loudly).

This module never sees CSV columns; it only ever receives a Text string.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import regex as re

# Canonical-letter -> additional leetspeak characters that may stand in for it.
_LEET_VARIANTS = {
    "a": "4@", "b": "8", "c": "(", "e": "3", "g": "69", "i": "1!|",
    "l": "1|", "o": "0", "s": "5$", "t": "7+", "z": "2",
}
# Reverse map used by skeleton(): leet char -> canonical letter (best-effort).
_LEET_TO_LETTER = {
    "4": "a", "@": "a", "8": "b", "(": "c", "3": "e", "6": "g", "9": "g",
    "1": "i", "!": "i", "|": "i", "0": "o", "5": "s", "$": "s", "7": "t",
    "+": "t", "2": "z",
}

_MIN_OBFUSCATION_LEN = 3  # shorter terms are matched as exact words only
_CHUNK = 300              # terms per combined alternation pattern


def skeleton(token: str) -> str:
    """Best-effort de-obfuscated skeleton: lowercase, de-leet, drop non-alnum,
    collapse repeats. A helper for exact comparisons and tests."""
    out = []
    prev = None
    for ch in token.lower():
        ch = _LEET_TO_LETTER.get(ch, ch)
        if not ch.isalnum():
            continue
        if ch != prev:
            out.append(ch)
        prev = ch
    return "".join(out)


def _term_body(term: str, max_gap: int) -> str:
    """Regex body matching an obfuscated spelling of ``term`` (no anchors)."""
    parts = []
    for ch in term:
        low = ch.lower()
        if low.isalpha():
            variants = low + _LEET_VARIANTS.get(low, "")
            cls = "[" + "".join(re.escape(c) for c in variants) + "]"
            parts.append(cls + "+")        # '+' absorbs elongation
        elif ch.isdigit():
            parts.append(re.escape(ch) + "+")
        else:
            parts.append(re.escape(ch))
    gap = r"[\W_]{0,%d}" % max(0, max_gap)
    return gap.join(parts)


@dataclass
class _Chunk:
    pattern: "re.Pattern"
    canonicals: List[str]   # index i -> canonical for group g{i}


class Lexicon:
    """Compiled, obfuscation-aware term matcher."""

    def __init__(self, terms: List[str], max_inter_char_gap: int = 2,
                 source_note: str = ""):
        # Normalise + dedupe canonical terms, preserving order.
        seen = set()
        self.terms: List[str] = []
        for t in terms:
            t = (t or "").strip().lower()
            if t and t not in seen:
                seen.add(t)
                self.terms.append(t)
        self.max_gap = max_inter_char_gap
        self.source_note = source_note
        self._chunks: List[_Chunk] = self._compile(self.terms)

    def _compile(self, terms: List[str]) -> List[_Chunk]:
        chunks: List[_Chunk] = []
        for start in range(0, len(terms), _CHUNK):
            group = terms[start:start + _CHUNK]
            alts = []
            canon = []
            for i, term in enumerate(group):
                if len(term) >= _MIN_OBFUSCATION_LEN:
                    body = _term_body(term, self.max_gap)
                else:
                    body = re.escape(term)
                alts.append(r"(?P<g%d>%s)" % (i, body))
                canon.append(term)
            pat = (
                r"(?<![0-9\p{L}])(?:"
                + "|".join(alts)
                + r")(?![0-9\p{L}])"
            )
            chunks.append(_Chunk(re.compile(pat, re.IGNORECASE | re.UNICODE), canon))
        return chunks

    def __len__(self) -> int:
        return len(self.terms)

    @property
    def loaded(self) -> bool:
        return len(self.terms) > 0

    def find_protected_spans(self, text: str) -> List[Tuple[int, int, str]]:
        """Return non-overlapping (start, end, canonical) spans, sorted by start.
        Longer matches win on overlap."""
        raw: List[Tuple[int, int, str]] = []
        for chunk in self._chunks:
            for m in chunk.pattern.finditer(text):
                gname = m.lastgroup
                if gname is None or not gname.startswith("g"):
                    continue
                idx = int(gname[1:])
                raw.append((m.start(), m.end(), chunk.canonicals[idx]))
        if not raw:
            return []
        # Greedy non-overlapping selection: longest first, then leftmost.
        raw.sort(key=lambda s: (-(s[1] - s[0]), s[0]))
        chosen: List[Tuple[int, int, str]] = []
        occupied: List[Tuple[int, int]] = []
        for start, end, canon in raw:
            if all(end <= os or start >= oe for os, oe in occupied):
                chosen.append((start, end, canon))
                occupied.append((start, end))
        chosen.sort(key=lambda s: s[0])
        return chosen


def load_terms_from_file(path: str) -> List[str]:
    terms: List[str] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            terms.append(line)
    return terms


def lexicon_from_config(cfg) -> Tuple[Lexicon, Optional[str]]:
    """Build a Lexicon from a Config. Returns (lexicon, warning_or_None).

    ``warning`` is a human-readable string when ``source: real`` but the file is
    missing/empty, so callers can surface it loudly instead of silently
    protecting nothing.
    """
    lex_cfg = cfg.lexicon
    warning = None
    if lex_cfg.source == "test":
        terms = list(lex_cfg.test_terms)
        note = "test lexicon (in-config placeholder terms)"
        if not terms:
            warning = "test lexicon configured but test_terms is empty"
    else:
        if os.path.exists(lex_cfg.path):
            terms = load_terms_from_file(lex_cfg.path)
            note = "real lexicon from %s (%d terms)" % (lex_cfg.path, len(terms))
            if not terms:
                warning = "lexicon file %s is empty" % lex_cfg.path
        else:
            terms = []
            note = "real lexicon MISSING at %s" % lex_cfg.path
            warning = (
                "Hate lexicon not found at %r. Run scripts/setup_lexicons.py to "
                "download it. Proceeding with NO protection step." % lex_cfg.path
            )
    return Lexicon(terms, lex_cfg.max_inter_char_gap, source_note=note), warning
