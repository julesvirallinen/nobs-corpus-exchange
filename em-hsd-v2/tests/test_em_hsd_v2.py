"""EM-HSD 2.0 unit tests (mock proposer, no GPU)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from em_hsd import load_em_hsd_config, privatize_em_hsd_v2
from em_hsd.constraints import filter_candidates, protected_skeletons, spans_preserved
from em_hsd.dp_select import select_rewrite
from em_hsd.sensitivity import refined_delta_u
from em_hsd.utility_scorer import ProxyHateScorer, analyze_from_logits, _derive_severity
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
    assert audit["deployment_mode"] == "standalone"
    assert audit["epsilon_1"] == audit["epsilon_2"]
    assert audit["epsilon_1_spent_here"] == audit["epsilon_1"]
    assert audit["utility_backend"] == "proxy"
    assert "P_hate_original" in audit
    assert "P_hate_x_priv" in audit
    assert "hsd_original" in audit
    assert "hsd_output" in audit
    assert audit["hsd_original"]["severity"] in ("none", "mild", "moderate", "severe")
    assert isinstance(audit["hsd_original"]["hs_labels"], list)
    assert isinstance(audit["hsd_original"]["labels"], dict)


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
    assert cfg.em_hsd_v2.deployment_mode == "standalone"
    assert cfg.em_hsd_v2.k_generate == 4
    assert cfg.utility.backend == "proxy"
    assert cfg.utility.model == "unitary/unbiased-toxic-roberta"
    assert cfg.utility.score_label == "toxicity"
    assert cfg.utility.label_threshold == 0.5


def test_production_config_llama_cpp():
    prod = load_em_hsd_config(str(ROOT / "configs" / "em-hsd-v2.yaml"))
    assert prod.utility.backend == "hf"
    assert prod.generation.backend == "llama_cpp"
    assert prod.generation.model == "unsloth/Qwen3.5-0.8B-GGUF"
    assert prod.generation.quant == "UD-Q4_K_XL"
    assert prod.generation.enable_thinking is False


def test_multilabel_analyze_from_logits():
    import torch
    from types import SimpleNamespace

    config = SimpleNamespace(
        problem_type="multi_label_classification",
        id2label={
            0: "toxicity",
            1: "severe_toxicity",
            2: "obscene",
            3: "insult",
        },
    )
    logits = torch.tensor([2.0, 1.5, 0.0, 0.8])
    hsd = analyze_from_logits(logits, config, "toxicity", 0.5, torch)
    expected_tox = float(torch.sigmoid(torch.tensor(2.0)))
    assert abs(hsd.p_hate - expected_tox) < 1e-6
    assert "toxicity" in hsd.labels
    assert "severe_toxicity" in hsd.labels
    assert "toxicity" in hsd.hs_labels
    assert hsd.severity in ("moderate", "severe", "mild")


def test_proxy_analyze_hs_labels():
    scorer = ProxyHateScorer(["dummy"])
    hsd = scorer.analyze("you are a dummy")
    assert hsd.p_hate > 0.5
    assert "toxicity" in hsd.hs_labels
    assert hsd.severity != "none"


def test_derive_severity_tiers():
    severity, score = _derive_severity(
        {"toxicity": 0.9, "severe_toxicity": 0.75, "threat": 0.1},
        0.9,
    )
    assert severity == "severe"
    assert score >= 0.75

    severity_mild, _ = _derive_severity({"toxicity": 0.25}, 0.25)
    assert severity_mild == "mild"


def test_qwen_gguf_model_resolution():
    from em_hsd.qwen_models import (
        is_gguf_repo,
        llama_server_model_id,
        resolve_unsloth_safetensors_model,
    )

    assert is_gguf_repo("unsloth/Qwen3.5-0.8B-GGUF")
    assert not is_gguf_repo("Qwen/Qwen3.5-0.8B")
    assert resolve_unsloth_safetensors_model("unsloth/Qwen3.5-0.8B-GGUF") == "Qwen/Qwen3.5-0.8B"
    assert llama_server_model_id("unsloth/Qwen3.5-0.8B-GGUF") == "unsloth/Qwen3.5-0.8B-GGUF"


def test_composed_mode_skips_phase_1a():
    composed = load_em_hsd_config(str(ROOT / "configs" / "em-hsd-v2-composed.yaml"))
    composed.spine.rng = make_row_rng(10, run_seed="test")
    raw = "Stop being such a dummy and read the instructions before you ask."
    t_prime = "stop being such a dummy and read the instructions before you ask."
    out, audit = privatize_em_hsd_v2(
        t_prime,
        composed,
        original_text=raw,
        protected_spans=["dummy"],
        upstream_token_log=[{"token": "stop", "action": "sanitized"}],
    )
    assert isinstance(out, str) and out.strip()
    assert audit["deployment_mode"] == "composed"
    assert audit["epsilon_1"] == 0.0
    assert audit["epsilon_1_spent_here"] == 0.0
    assert audit["epsilon_2"] == composed.em_hsd_v2.epsilon_total
    assert audit["x_priv"] == t_prime
    assert audit["protected_terms"] == ["dummy"]
    assert audit["reference_text"] == raw[:500]
    assert len(audit["token_log"]) >= 1


def test_composed_mode_external_protected_spans():
    composed = load_em_hsd_config(str(ROOT / "configs" / "em-hsd-v2-composed.yaml"))
    composed.spine.rng = make_row_rng(11, run_seed="test")
    _, audit = privatize_em_hsd_v2(
        "you are a dummy fool",
        composed,
        original_text="You are a dummy fool",
        protected_spans=["dummy", "fool"],
    )
    assert set(audit["protected_terms"]) == {"dummy", "fool"}
