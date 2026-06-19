"""Candidate filters for EM-HSD Phase 2."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import regex as re

from em_hsd.core.config import EmHsdConfig
from em_hsd.layer4.scorer import _skeleton


@dataclass
class FilterResult:
    candidate: str
    valid: bool
    reject: str = ""
    p_hate: float = 0.0
    sem_cos: float = 0.0


@dataclass
class FilterBatch:
    valid: list[str] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    details: list[FilterResult] = field(default_factory=list)


def protected_skeletons(terms: Sequence[str]) -> list[str]:
    skels = sorted({_skeleton(t) for t in terms if t.strip()})
    return [s for s in skels if len(s) >= 3]


def extract_protected_terms(text: str, canonicals: Sequence[str]) -> list[str]:
    """Return canonical protected terms present in text."""
    found = []
    lower = (text or "").lower()
    for c in canonicals:
        if not c:
            continue
        if c.lower() in lower or _skeleton(c) in {_skeleton(w) for w in re.findall(r"\S+", lower)}:
            found.append(c)
    return sorted(set(found))


def spans_preserved(candidate: str, skeletons: Sequence[str]) -> bool:
    if not skeletons:
        return True
    toks = {_skeleton(t) for t in re.findall(r"\S+", (candidate or "").lower())}
    despaced = _skeleton(re.sub(r"\s+", "", candidate or ""))
    return all(sk in toks or sk in despaced for sk in skeletons)


def normalized_edit_ratio(a: str, b: str) -> float:
    a, b = a or "", b or ""
    if not a and not b:
        return 0.0
    if not a or not b:
        return 1.0
    la, lb = len(a), len(b)
    if la > lb:
        a, b = b, a
        la, lb = lb, la
    prev = list(range(lb + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            ins = cur[j - 1] + 1
            dele = prev[j] + 1
            sub = prev[j - 1] + (ca != cb)
            cur.append(min(ins, dele, sub))
        prev = cur
    dist = prev[lb]
    return dist / max(len(a), len(b), 1)


def filter_candidates(
    candidates: Sequence[str],
    original: str,
    x_priv: str,
    protected_skels: Sequence[str],
    config: EmHsdConfig,
    scorer: object,
    encoder: object,
) -> FilterBatch:
    em = config.em_hsd_v2
    cosine = getattr(encoder, "cosine")
    score = getattr(scorer, "score")
    p_orig = float(score(original))
    batch = FilterBatch()
    for cand in candidates:
        if not cand or not cand.strip():
            batch.details.append(FilterResult(cand, False, "empty"))
            continue

        if not spans_preserved(cand, protected_skels):
            batch.details.append(FilterResult(cand, False, "span"))
            continue

        p_hate = float(score(cand))
        if p_hate < p_orig - em.hate_floor_delta:
            batch.details.append(FilterResult(cand, False, "hate_floor", p_hate))
            continue

        if len(cand.strip().split()) < em.min_length_tokens:
            batch.details.append(FilterResult(cand, False, "length", p_hate))
            continue

        if em.max_length_tokens > 0 and len(cand.strip().split()) > em.max_length_tokens:
            batch.details.append(FilterResult(cand, False, "length", p_hate))
            continue

        sem_cos = float(cosine(original, cand))
        if sem_cos < em.tau_sem_min:
            batch.details.append(FilterResult(cand, False, "semantic", p_hate, sem_cos))
            continue

        edit_ratio = normalized_edit_ratio(original, cand)
        if edit_ratio < em.min_edit_ratio:
            batch.details.append(FilterResult(cand, False, "edit_ratio", p_hate, sem_cos))
            continue

        batch.valid.append(cand)
        # Reward faithful (high sem_cos), low-hate candidates, and — for privacy —
        # those that diverge more in surface form (edit_ratio) from the original.
        batch.scores.append(sem_cos - em.delta_u * p_hate + em.privacy_weight * edit_ratio)
        batch.details.append(FilterResult(cand, True, "", p_hate, sem_cos))

    return batch
