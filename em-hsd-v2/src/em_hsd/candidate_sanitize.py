"""Strip prompt-echo artifacts from LLM paraphrase candidates."""

from __future__ import annotations

import re
from typing import List, Sequence, Tuple

_PROMPT_MARKERS = (
    "privacy (p):",
    "utility (u):",
    "keep unchanged in meaning",
    "stylometric fingerprint",
    "rewrite the post below",
    "rewrite this post",
    "you are anonymizing",
    "output only the rewritten post",
    "do not soften",
    "break stylometric",
    "p_privacy:",
    "ut_utility:",
)

_POST_LABEL = re.compile(r"(?is)(?:^|\n)\s*post:\s*")


def extract_post_body(text: str) -> str:
    """If the model repeated the template, take text after the last ``Post:`` label."""
    if not text:
        return text
    matches = list(_POST_LABEL.finditer(text))
    if not matches:
        return text.strip()
    body = text[matches[-1].end() :].strip()
    return body if body else text.strip()


def is_prompt_echo(candidate: str) -> bool:
    """True when the string looks like leaked instructions, not a post rewrite."""
    if not candidate or not candidate.strip():
        return True
    stripped = candidate.strip()
    lower = stripped.lower()
    if any(marker in lower for marker in _PROMPT_MARKERS):
        return True
    if lower.startswith("privacy:") or lower.startswith("utility:"):
        return True
    if stripped.count("\n") >= 3 and any(m in lower for m in ("privacy", "utility", "constraints:")):
        return True
    return False


def drop_prompt_echoes(candidates: Sequence[str]) -> Tuple[List[str], List[str]]:
    """Return (kept, dropped) after echo stripping and body extraction."""
    kept: List[str] = []
    dropped: List[str] = []
    seen = set()
    for raw in candidates:
        body = extract_post_body(raw or "")
        if is_prompt_echo(body):
            dropped.append(raw[:200] if raw else "")
            continue
        if not body or body in seen:
            if raw:
                dropped.append(raw[:200])
            continue
        seen.add(body)
        kept.append(body)
    return kept, dropped
