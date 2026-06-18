"""Stretch scaffold tests: the exponential-mechanism candidate selection is
implemented and tested; the generative proposer is NotImplemented."""

import numpy as np
import pytest

from stretch.candidate_selection import (CandidateSelectionLayer,
                                         NotImplementedProposer, is_hard,
                                         select_rewrite)


def test_select_rewrite_large_epsilon_picks_best_candidate():
    candidates = ["a", "b", "best", "d"]
    scores = [0.0, 1.0, 5.0, 2.0]   # 'best' has the highest utility
    rng = np.random.default_rng(0)
    picks = [select_rewrite(candidates, scores, 1e6, 5.0, rng)[0]
             for _ in range(200)]
    assert all(p == "best" for p in picks)


def test_select_rewrite_small_epsilon_is_varied():
    candidates = ["a", "b", "c", "d"]
    scores = [4.0, 0.0, -4.0, 1.0]
    rng = np.random.default_rng(1)
    picks = {select_rewrite(candidates, scores, 1e-6, 5.0, rng)[0]
             for _ in range(400)}
    assert len(picks) > 1   # near-uniform -> selects more than one candidate


def test_select_rewrite_validates_inputs():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        select_rewrite(["a", "b"], [1.0], 1.0, 5.0, rng)
    with pytest.raises(ValueError):
        select_rewrite([], [], 1.0, 5.0, rng)


def test_generative_proposer_is_not_implemented():
    with pytest.raises(NotImplementedError):
        NotImplementedProposer().propose("some hard row text", k=4)


def test_layer_rewrite_raises_until_proposer_implemented():
    class DummyScorer:
        def score(self, c):
            return 1.0
    layer = CandidateSelectionLayer(NotImplementedProposer(), DummyScorer(),
                                    epsilon=2.0, clip=5.0)
    with pytest.raises(NotImplementedError):
        layer.rewrite("text", np.random.default_rng(0))


def test_is_hard_gate():
    assert is_hard("word " * 40, 40)
    assert not is_hard("short text", 40)
