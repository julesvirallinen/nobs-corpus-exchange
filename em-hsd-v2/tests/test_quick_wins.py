"""Tests for quick-win filter and candidate sanitization improvements."""

from __future__ import annotations

from pathlib import Path

from em_hsd import load_em_hsd_config
from em_hsd.candidate_sanitize import drop_prompt_echoes, extract_post_body, is_prompt_echo
from em_hsd.constraints import (
    _hate_floor_threshold,
    filter_candidates,
    protected_skeletons,
    spans_preserved,
)
from em_hsd.embedding import SimpleEncoder
from em_hsd.utility_scorer import ProxyHateScorer

ROOT = Path(__file__).resolve().parents[1]


def test_is_prompt_echo_detects_instruction_leak():
    assert is_prompt_echo("PRIVACY: Break stylometric fingerprint. UTILITY: Keep insults.")
    assert not is_prompt_echo("you are a dummy and should stop posting nonsense")


def test_extract_post_body_after_template():
    raw = "Rewrite this.\n\nPost: you are a dummy fool\n"
    assert extract_post_body(raw) == "you are a dummy fool"


def test_drop_prompt_echoes_keeps_valid():
    kept, dropped = drop_prompt_echoes([
        "Privacy (p): change style\nPost: real rewrite here",
        "PRIVACY: leaked template only",
        "real rewrite here",
    ])
    assert kept == ["real rewrite here"]
    assert len(dropped) == 2


def test_hate_floor_adaptive_when_x_priv_dropped():
    assert _hate_floor_threshold(0.90, 0.30, 0.15) == 0.15
    assert _hate_floor_threshold(0.90, 0.88, 0.15) == 0.75


def test_soft_span_accepts_surface_term_with_hate(cfg):
    from em_hsd.constraints import _span_check_passes, protected_surfaces_present

    assert protected_surfaces_present("honestly you are still a dummy person", ["dummy"])
    assert protected_surfaces_present("some dummy here", ["dummy", "fool"])
    assert _span_check_passes(
        "honestly you are still a dummy person",
        ["xxxx"],
        ["dummy"],
        p_hate=0.8,
        p_floor=0.5,
    )


def test_dual_sem_floor_rejects_low_x_priv_similarity(cfg):
    scorer = ProxyHateScorer(cfg.spine.lexicon.test_terms)
    encoder = SimpleEncoder()
    cfg.em_hsd_v2.tau_sem_min = 0.0
    cfg.em_hsd_v2.tau_sem_x_priv_min = 0.99
    batch = filter_candidates(
        ["Stop being such a dummy please ok"],
        "Stop being such a dummy please",
        "totally unrelated scrambled x priv text",
        protected_skeletons(["dummy"]),
        cfg,
        scorer,
        encoder,
        protected_terms=["dummy"],
        p_hate_x_priv=0.9,
    )
    assert batch.valid == []
    assert any(d.reject == "sem_floor_priv" for d in batch.details)


def test_production_config_quick_win_defaults():
    prod = load_em_hsd_config(str(ROOT / "configs" / "em-hsd-v2.yaml"))
    assert prod.em_hsd_v2.k_generate == 10
    assert prod.em_hsd_v2.k_max_after_prune == 6
    assert prod.em_hsd_v2.tau_sem_x_priv_min == 0.35
    assert prod.em_hsd_v2.tau_sem_min == 0.28
    assert prod.em_hsd_v2.generation_temperature == 0.7
    assert prod.generation.top_p == 0.8
    assert prod.generation.presence_penalty == 1.5
