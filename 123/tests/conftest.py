"""Shared test fixtures."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPINE_SRC = ROOT.parent / "spine" / "src"
if str(SPINE_SRC) not in sys.path:
    sys.path.insert(0, str(SPINE_SRC))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

TEST_CONFIG = str(ROOT / "configs" / "em-hsd-v2-test.yaml")
SYNTH_CSV = str(ROOT.parent / "spine" / "data" / "synthetic_dev.csv")


@pytest.fixture(autouse=True)
def _allow_model_downloads(monkeypatch):
    """All backends are real now, so tests load models from the HF cache.

    Permit it by default; download-policy tests opt out with monkeypatch.delenv.
    """
    monkeypatch.setenv("EM_HSD_ALLOW_DOWNLOADS", "1")


@pytest.fixture
def config_path():
    return TEST_CONFIG


@pytest.fixture
def cfg():
    from em_hsd import load_em_hsd_config
    return load_em_hsd_config(TEST_CONFIG)


@pytest.fixture
def synth_path():
    assert os.path.exists(SYNTH_CSV), "synthetic_dev.csv missing in spine"
    return SYNTH_CSV
