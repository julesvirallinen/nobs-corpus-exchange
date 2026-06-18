from __future__ import annotations

import regex as re


def word_count(text: str) -> int:
    return len(re.findall(r"[^\W_]+", text or ""))


def refined_delta_u(text: str) -> float:    length = max(1, word_count(text))
    return min(1.0, 2.0 / length)


def selection_sensitivity(text: str, use_refined: bool) -> float:
    if use_refined:
        return refined_delta_u(text)
    return 1.0
