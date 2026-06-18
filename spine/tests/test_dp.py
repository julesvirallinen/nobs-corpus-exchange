"""Test 6: DP rewrite — large epsilon -> argmax; small epsilon -> ~uniform over
the clipped candidate set; logits clipped before softmax; epsilon from config."""

import numpy as np
import pytest

from mechanism import dp
from mechanism.rng import make_row_rng


def test_large_epsilon_picks_argmax():
    scores = [0.0, 1.0, 4.0, 2.0, -3.0]  # argmax index = 2
    rng = np.random.default_rng(0)
    picks = [dp.select_index(scores, epsilon=1e6, clip=5.0, rng=rng).index
             for _ in range(300)]
    assert all(p == 2 for p in picks), set(picks)


def test_small_epsilon_approaches_uniform():
    k = 6
    scores = [3.0, -2.0, 1.0, 4.0, -4.0, 0.0]
    # Distribution (deterministic): near-uniform.
    probs = dp.exponential_weights(dp.clip_logits(np.array(scores), 5.0),
                                   epsilon=1e-6, sensitivity=10.0)
    assert np.allclose(probs, 1.0 / k, atol=1e-3)
    # And empirically over many draws.
    rng = np.random.default_rng(1)
    counts = np.zeros(k)
    draws = 6000
    for _ in range(draws):
        counts[dp.select_index(scores, epsilon=1e-6, clip=5.0, rng=rng).index] += 1
    freqs = counts / draws
    assert np.all(np.abs(freqs - 1.0 / k) < 0.05), freqs


def test_logits_clipped_before_softmax():
    sel = dp.select_index([100.0, -100.0, 3.0], epsilon=2.0, clip=5.0,
                          rng=np.random.default_rng(0))
    assert sel.clipped.max() <= 5.0 + 1e-9
    assert sel.clipped.min() >= -5.0 - 1e-9
    # Two raw vectors differing only beyond the clip range must give equal probs.
    p1 = dp.select_index([100.0, 0.0], 1.0, 5.0, np.random.default_rng(0)).probs
    p2 = dp.select_index([7.0, 0.0], 1.0, 5.0, np.random.default_rng(0)).probs
    assert np.allclose(p1, p2)


def test_exponential_mechanism_is_valid_distribution():
    probs = dp.exponential_weights(np.array([1.0, 2.0, 3.0]), epsilon=4.0,
                                   sensitivity=6.0)
    assert abs(probs.sum() - 1.0) < 1e-9
    assert np.all(probs >= 0)


def test_epsilon_read_from_config(cfg):
    from mechanism import privatize
    cfg.rng = make_row_rng(0, run_seed="EPS")
    _, log = privatize("the documentary about creatures was fascinating", cfg)
    rewritten = [e for e in log if e["action"] == "rewritten"]
    assert rewritten, "expected some content tokens to be rewritten"
    for e in rewritten:
        assert e["epsilon"] == cfg.epsilon.content == 6.0
