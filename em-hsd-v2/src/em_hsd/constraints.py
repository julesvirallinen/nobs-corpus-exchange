"""Candidate filters for EM-HSD Phase 2."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

import regex as re

from .config import EmHsdConfig
from .utility_scorer import _skeleton, HsdAnalysis


@dataclass
class FilterResult:
    candidate: str
    valid: bool
    reject: str = ""
    p_hate: float = 0.0
    sem_cos: float = 0.0
    sem_cos_x_priv: float = 0.0
    severity: str = "none"
    severity_score: float = 0.0
    hs_labels: List[str] = field(default_factory=list)
    label_probs: dict = field(default_factory=dict)


@dataclass
class FilterBatch:
    valid: List[str] = field(default_factory=list)
    scores: List[float] = field(default_factory=list)
    details: List[FilterResult] = field(default_factory=list)


def protected_skeletons(terms: Sequence[str]) -> List[str]:
    skels = sorted({_skeleton(t) for t in terms if t.strip()})
    return [s for s in skels if len(s) >= 3]


def extract_protected_terms(text: str, canonicals: Sequence[str]) -> List[str]:
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
    for sk in skeletons:
        if sk not in toks and sk not in despaced:
            return False
    return True


def protected_surfaces_present(candidate: str, protected_terms: Sequence[str]) -> bool:
    """True when at least half of listed protected terms appear as substrings."""
    if not protected_terms:
        return True
    lower = (candidate or "").lower()
    terms = [str(t).strip().lower() for t in protected_terms if t and str(t).strip()]
    if not terms:
        return True
    hits = sum(1 for t in terms if t in lower)
    return hits >= max(1, (len(terms) + 1) // 2)


def _hate_floor_threshold(p_orig: float, p_x_priv: float, delta: float) -> float:
    """Adaptive floor: if ε₁ dropped P_hate, do not require original-level toxicity."""
    if p_x_priv <= p_orig - delta:
        return p_x_priv - delta
    return p_orig - delta


def _span_check_passes(
    candidate: str,
    skeletons: Sequence[str],
    protected_terms: Sequence[str],
    p_hate: float,
    p_floor: float,
) -> bool:
    if spans_preserved(candidate, skeletons):
        return True
    if not protected_terms:
        return False
    if protected_surfaces_present(candidate, protected_terms) and p_hate >= p_floor:
        return True
    return False


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


def _analyze(scorer: object, text: str) -> HsdAnalysis:
    if hasattr(scorer, "analyze"):
        return scorer.analyze(text)
    p = float(scorer.score(text))
    return HsdAnalysis(p_hate=p, labels={"toxicity": p}, hs_labels=[], severity="none", severity_score=p)


def _result_from_analysis(
    candidate: str,
    valid: bool,
    reject: str,
    analysis,
    sem_cos: float = 0.0,
    sem_cos_x_priv: float = 0.0,
) -> FilterResult:
    return FilterResult(
        candidate=candidate,
        valid=valid,
        reject=reject,
        p_hate=float(analysis.p_hate),
        sem_cos=sem_cos,
        sem_cos_x_priv=sem_cos_x_priv,
        severity=str(analysis.severity),
        severity_score=float(analysis.severity_score),
        hs_labels=list(analysis.hs_labels),
        label_probs=dict(analysis.labels),
    )


def filter_candidates(
    candidates: Sequence[str],
    original: str,
    x_priv: str,
    protected_skels: Sequence[str],
    config: EmHsdConfig,
    scorer: object,
    encoder: object,
    *,
    protected_terms: Sequence[str] | None = None,
    p_hate_x_priv: float | None = None,
) -> FilterBatch:
    em = config.em_hsd_v2
    orig = _analyze(scorer, original)
    p_orig = float(orig.p_hate)
    if p_hate_x_priv is None:
        p_hate_x_priv = float(_analyze(scorer, x_priv).p_hate)
    p_floor = _hate_floor_threshold(p_orig, float(p_hate_x_priv), em.hate_floor_delta)
    terms = list(protected_terms or [])
    batch = FilterBatch()

    for cand in candidates:
        if not cand or not cand.strip():
            batch.details.append(FilterResult(cand, False, "empty"))
            continue

        cand_hsd = _analyze(scorer, cand)
        p_hate = float(cand_hsd.p_hate)

        if not _span_check_passes(cand, protected_skels, terms, p_hate, p_floor):
            batch.details.append(_result_from_analysis(cand, False, "span", cand_hsd))
            continue

        if p_hate < p_floor:
            batch.details.append(_result_from_analysis(cand, False, "hate_floor", cand_hsd))
            continue

        sem_orig = float(encoder.cosine(original, cand))
        if sem_orig < em.tau_sem_min:
            batch.details.append(
                _result_from_analysis(cand, False, "sem_floor_orig", cand_hsd, sem_orig)
            )
            continue

        sem_priv = float(encoder.cosine(x_priv, cand))
        if sem_priv < em.tau_sem_x_priv_min:
            batch.details.append(
                _result_from_analysis(
                    cand, False, "sem_floor_priv", cand_hsd, sem_orig, sem_priv,
                )
            )
            continue

        len_ratio = len(cand) / max(len(original), 1)
        if len_ratio < 0.4 or len_ratio > 2.5:
            batch.details.append(
                _result_from_analysis(cand, False, "length", cand_hsd, sem_orig, sem_priv)
            )
            continue

        edit = normalized_edit_ratio(cand, original)
        if edit < em.min_edit_ratio:
            batch.details.append(
                _result_from_analysis(cand, False, "min_edit", cand_hsd, sem_orig, sem_priv)
            )
            continue

        batch.valid.append(cand)
        batch.scores.append(p_hate)
        batch.details.append(
            _result_from_analysis(cand, True, "", cand_hsd, sem_orig, sem_priv)
        )

    return batch
