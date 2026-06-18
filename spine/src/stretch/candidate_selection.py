"""Stretch scaffold: a candidate-selection layer for hard rows.

Idea (gated by config, hard rows only): a small LOCAL open-weight generative
model proposes k rewrites of a hard row; a local hate classifier scores each
candidate for utility; the exponential mechanism selects one DP-privately.

What is IMPLEMENTED and unit-tested here:
  * ``select_rewrite`` — exponential-mechanism selection over candidate rewrites
    given their utility scores (reuses mechanism.dp so the DP math is shared).
  * ``is_hard`` — the hard-row gate.

What is INTENTIONALLY NOT implemented (declared, NotImplemented):
  * ``GenerativeProposer.propose`` — proposing candidates with a local generative
    model. No generative model is downloaded or wired up. The interface is fixed
    so the layer can be completed later without touching the rest of the system.

This module is a scaffold: do not enable it for a submission yet.
"""

from __future__ import annotations

from typing import List, Protocol, Sequence, Tuple

import numpy as np

from mechanism import dp


class GenerativeProposer(Protocol):
    """Proposes candidate rewrites for a hard row. NOT IMPLEMENTED."""

    def propose(self, text: str, k: int) -> List[str]:
        ...


class NotImplementedProposer:
    """Placeholder proposer. Raises until a local generative model is wired in.

    To implement: load a SMALL, LOCAL, open-weight instruction model (no hosted
    APIs), prompt it to paraphrase ``text`` k ways while preserving hate
    category, and return the k candidate strings. Keep it CPU-friendly and pin
    the version. Do NOT download any generative model as part of the current
    deliverable.
    """

    def propose(self, text: str, k: int) -> List[str]:
        raise NotImplementedError(
            "GenerativeProposer is a scaffold. Wire in a small local open-weight "
            "generative model here before enabling the stretch layer."
        )


class CandidateScorer(Protocol):
    """Scores a candidate's utility (higher = better hate-classification utility)."""

    def score(self, candidate: str) -> float:
        ...


def select_rewrite(candidates: Sequence[str], scores: Sequence[float],
                   epsilon: float, clip: float,
                   rng: np.random.Generator) -> Tuple[str, "dp.Selection"]:
    """Select one candidate via the exponential mechanism over utility scores.

    Same DP construction as the per-token rewrite: scores are clipped to
    [-clip, clip] (sensitivity 2*clip) and a candidate is sampled with
    probability proportional to exp(epsilon * clipped_score / (2*sensitivity)).
    """
    if len(candidates) != len(scores):
        raise ValueError("candidates and scores must have equal length")
    if not candidates:
        raise ValueError("no candidates to select from")
    return dp.select(list(candidates), list(scores), epsilon, clip, rng)


def is_hard(text: str, hard_row_min_tokens: int) -> bool:
    """Gate: a row is 'hard' if it has at least ``hard_row_min_tokens`` words."""
    import regex as re
    return len(re.findall(r"[^\W_]+", text or "")) >= hard_row_min_tokens


class CandidateSelectionLayer:
    """Wires a proposer + scorer + exponential-mechanism selection together.

    Selection is implemented and tested; the proposer is NotImplemented by
    default so the layer cannot silently run a generative model.
    """

    def __init__(self, proposer: GenerativeProposer, scorer: CandidateScorer,
                 epsilon: float, clip: float, k: int = 4):
        self.proposer = proposer
        self.scorer = scorer
        self.epsilon = epsilon
        self.clip = clip
        self.k = k

    def rewrite(self, text: str, rng: np.random.Generator) -> Tuple[str, "dp.Selection"]:
        candidates = self.proposer.propose(text, self.k)   # raises if NotImplemented
        scores = [self.scorer.score(c) for c in candidates]
        return select_rewrite(candidates, scores, self.epsilon, self.clip, rng)
