from __future__ import annotations

from mechanism.rng import make_row_rng

from em_hsd import load_em_hsd_config
from em_hsd.interfaces.mock import (
    NoOpStylometricPrior,
    NoOpTOOptimizer,
    NoOpTriageRouter,
)
from em_hsd.interfaces.triage import StylometricPrior, TOOptimizer, TriageRouter
from em_hsd.layer4.orchestrator import Layer4Orchestrator
from triage_dp.pipeline import TriageDpPipeline


def test_no_op_classes_satisfy_protocols():
    assert isinstance(NoOpTriageRouter(), TriageRouter)
    assert isinstance(NoOpStylometricPrior(), StylometricPrior)
    assert isinstance(NoOpTOOptimizer(), TOOptimizer)


def test_pipeline_runs_in_standalone_mode():
    cfg = load_em_hsd_config("configs/em-hsd-v2-test.yaml")
    cfg.spine.rng = make_row_rng(20, run_seed="test")
    pipe = TriageDpPipeline(cfg)
    out, audit = pipe.sanitize("you are a dummy fool")
    assert isinstance(out, str) and out.strip()
    assert audit["mode"] == "em-hsd-v2"


def test_pipeline_matches_standalone_output():
    cfg = load_em_hsd_config("configs/em-hsd-v2-test.yaml")
    cfg.spine.rng = make_row_rng(21, run_seed="test")
    standalone_out, standalone_audit = Layer4Orchestrator().privatize(
        "you are a dummy fool", cfg
    )

    cfg.spine.rng = make_row_rng(21, run_seed="test")
    pipe_out, pipe_audit = TriageDpPipeline(cfg).sanitize(
        "you are a dummy fool",
        original_text="you are a dummy fool",
    )
    assert pipe_out == standalone_out
    assert pipe_audit["x_priv"] == standalone_audit["x_priv"]


def test_pipeline_with_triage_dp_enabled_mode():
    cfg = load_em_hsd_config("configs/em-hsd-v2-triage-real.yaml")
    cfg.spine.rng = make_row_rng(22, run_seed="test")
    pipe = TriageDpPipeline(cfg)
    out, audit = pipe.sanitize("stop being such a dummy", original_text="Stop being such a dummy")
    assert isinstance(out, str) and out.strip()
    assert audit["mode"] == "em-hsd-v2"
