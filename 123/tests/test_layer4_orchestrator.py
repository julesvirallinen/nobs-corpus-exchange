from __future__ import annotations

import re
from pathlib import Path

import pytest
from mechanism.rng import make_row_rng

from em_hsd import load_em_hsd_config
from em_hsd.core.policy import DownloadPolicy
from em_hsd.core.resources import ResourceManager
from em_hsd.layer4.orchestrator import Layer4Orchestrator

ROOT = Path(__file__).resolve().parents[1]


def test_policy_defaults_to_false():
    assert not DownloadPolicy.is_allowed()


def test_orchestrator_fallback_when_no_valid_candidates(cfg):
    cfg.spine.rng = make_row_rng(30, run_seed="test")
    cfg.em_hsd_v2.tau_sem_min = 0.99
    cfg.em_hsd_v2.min_edit_ratio = 0.99
    out, audit = Layer4Orchestrator().privatize("What an absolute doofus move.", cfg)
    assert audit["fallback"] is True
    assert out == audit["x_priv"]


def test_orchestrator_with_layer1_routes_preserves_protected(cfg):
    from em_hsd.interfaces.triage import TokenRoute

    cfg.spine.rng = make_row_rng(31, run_seed="test")
    routes = [TokenRoute(token="dummy", start=0, end=5, quadrant="Q1", action="protect")]
    out, audit = Layer4Orchestrator().privatize("you are a dummy fool", cfg, layer1_routes=routes)
    assert isinstance(out, str) and out.strip()
    assert "dummy" in audit["protected_terms"]


def test_orchestrator_with_layer3_overrides(cfg):
    cfg.spine.rng = make_row_rng(32, run_seed="test")
    original_total = cfg.em_hsd_v2.epsilon_total
    Layer4Orchestrator().privatize("you are a dummy", cfg, layer3_overrides={"epsilon_total": 9.0})
    assert cfg.em_hsd_v2.epsilon_total == 9.0
    cfg.em_hsd_v2.epsilon_total = original_total


def test_resource_manager_blocks_hf_download():
    prod = load_em_hsd_config(str(ROOT / "configs" / "em-hsd-v2.yaml"))
    rm = ResourceManager(prod, allow_downloads=False)
    with pytest.raises(RuntimeError):
        rm.scorer()


def test_resource_manager_blocks_unsloth_proposer(cfg):
    cfg.generation.backend = "unsloth"
    rm = ResourceManager(cfg, allow_downloads=False)
    with pytest.raises(RuntimeError):
        rm.proposer()


def test_em_hsd_core_does_not_import_harness():
    em_dir = ROOT / "src" / "em_hsd"
    pat = re.compile(r"^\s*(?:from harness|import harness)\b", re.M)
    for py in em_dir.rglob("*.py"):
        text = py.read_text(encoding="utf-8")
        assert not pat.search(text), py.name
