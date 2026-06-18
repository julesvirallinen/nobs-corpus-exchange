from __future__ import annotations

from typing import List, Protocol, Sequence, Tuple

import numpy as np

from mechanism import dp


class GenerativeProposer(Protocol):
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


def is_hard(text: str, hard_row_min_tokens: int) -> bool:    import regex as re
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
