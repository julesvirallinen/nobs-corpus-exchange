"""EM-HSD 2.0 unit tests (mock proposer, no GPU)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from em_hsd import load_em_hsd_config, privatize_em_hsd_v2
from em_hsd.constraints import filter_candidates, protected_skeletons, spans_preserved
from em_hsd.dp_select import select_rewrite
from em_hsd.sensitivity import refined_delta_u
from em_hsd.utility_scorer import ProxyHateScorer
from em_hsd.embedding import SimpleEncoder
from mechanism.rng import make_row_rng

ROOT = Path(__file__).resolve().parents[1]
TEST_CONFIG = str(ROOT / "configs" / "em-hsd-v2-test.yaml")


def test_full_pipeline_returns_non_empty(cfg):
    cfg.spine.rng = make_row_rng(0, run_seed="test")
    out, audit = privatize_em_hsd_v2(
        "Stop being such a dummy and read the instructions before you ask.", cfg,
    )
    assert isinstance(out, str) and out.strip()
    assert audit["mode"] == "em-hsd-v2"
    assert audit["epsilon_1"] == audit["epsilon_2"]


def test_spans_preserved_rejects_missing_protected():
    skels = protected_skeletons(["dummy"])
    assert spans_preserved("you are a dummy", skels)
    assert not spans_preserved("you are a fool", skels)


def test_filter_rejects_missing_span(cfg):
    cfg.spine.rng = make_row_rng(1, run_seed="test")
    scorer = ProxyHateScorer(cfg.spine.lexicon.test_terms)
    encoder = SimpleEncoder()
    batch = filter_candidates(
        ["you are a fool and should stop"],
        "Stop being such a dummy please",
        "stop being such a dummy please",
        protected_skeletons(["dummy"]),
        cfg,
        scorer,
        encoder,
    )
    assert batch.valid == []
    assert any(d.reject == "span" for d in batch.details)


def test_fallback_to_x_priv_when_all_invalid(cfg):
    cfg.spine.rng = make_row_rng(2, run_seed="test")
    cfg.em_hsd_v2.tau_sem_min = 0.99
    cfg.em_hsd_v2.min_edit_ratio = 0.99
    out, audit = privatize_em_hsd_v2("What an absolute doofus move.", cfg)
    assert audit["fallback"] is True
    assert out == audit["x_priv"]


def test_refined_delta_u_shorter_text_larger_bound():
    short = refined_delta_u("hello world")
    long = refined_delta_u("word " * 50)
    assert short > long


def test_select_rewrite_high_epsilon_picks_best():
    candidates = ["a", "b", "best"]
    scores = [0.1, 0.2, 0.95]
    rng = np.random.default_rng(0)
    picks = [
        select_rewrite(candidates, scores, 1e6, 1.0, rng, sensitivity=0.5)[0]
        for _ in range(100)
    ]
    assert all(p == "best" for p in picks)


def test_em_hsd_does_not_import_harness():
    import re
    em_dir = ROOT / "src" / "em_hsd"
    pat = re.compile(r"^\s*(?:from harness|import harness)\b", re.M)
    for py in em_dir.rglob("*.py"):
        text = py.read_text(encoding="utf-8")
        assert not pat.search(text), py.name


def test_load_config(cfg):
    assert cfg.generation.backend == "mock"
    assert cfg.em_hsd_v2.k_generate == 4
