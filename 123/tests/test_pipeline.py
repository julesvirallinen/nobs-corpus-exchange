"""Tests for the triage_dp pipeline package (Layers 1–4 composition)."""

from __future__ import annotations

from mechanism.rng import make_row_rng

from em_hsd import load_em_hsd_config
from em_hsd.interfaces.triage import StylometricPrior, TOOptimizer, TriageRouter
from em_hsd.layer4.orchestrator import Layer4Orchestrator
from triage_dp import TriageDpPipeline
from triage_dp.layer1_triage import DefaultTriageRouter
from triage_dp.layer2_stylometric import DefaultStylometricPrior
from triage_dp.layer3_calibration import DefaultTOOptimizer
from triage_dp.layer4_rewrite import DefaultRewriteLayer, RewriteLayer

TRIAGE_DP_CONFIG = "configs/em-hsd-v2-triage-real.yaml"


def test_layer_defaults_satisfy_protocols():
    assert isinstance(DefaultTriageRouter(), TriageRouter)
    assert isinstance(DefaultStylometricPrior(), StylometricPrior)
    assert isinstance(DefaultTOOptimizer(), TOOptimizer)
    assert isinstance(DefaultRewriteLayer(), RewriteLayer)


def test_pipeline_runs_standalone(config_path):
    pipe = TriageDpPipeline.from_config(config_path)
    pipe.config.spine.rng = make_row_rng(20, run_seed="test")
    out, audit = pipe.sanitize("you are a dummy fool")
    assert isinstance(out, str) and out.strip()
    assert audit["mode"] == "em-hsd-v2"


def test_pipeline_matches_direct_orchestrator(config_path):
    cfg = load_em_hsd_config(config_path)
    cfg.spine.rng = make_row_rng(21, run_seed="test")
    direct_out, direct_audit = Layer4Orchestrator().privatize("you are a dummy fool", cfg)

    pipe = TriageDpPipeline.from_config(config_path)
    pipe.config.spine.rng = make_row_rng(21, run_seed="test")
    pipe_out, pipe_audit = pipe.sanitize(
        "you are a dummy fool", original_text="you are a dummy fool"
    )
    assert pipe_out == direct_out
    assert pipe_audit["x_priv"] == direct_audit["x_priv"]


def test_pipeline_triage_dp_mode_runs():
    pipe = TriageDpPipeline.from_config(TRIAGE_DP_CONFIG)
    pipe.config.spine.rng = make_row_rng(22, run_seed="test")
    out, audit = pipe.sanitize(
        "stop being such a dummy", original_text="Stop being such a dummy"
    )
    assert isinstance(out, str) and out.strip()
    assert audit["mode"] == "em-hsd-v2"
