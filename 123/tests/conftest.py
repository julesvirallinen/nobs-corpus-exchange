"""Shared test fixtures."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPINE_SRC = ROOT.parent / "Johnny t0-1.03" / "src"
if str(SPINE_SRC) not in sys.path:
    sys.path.insert(0, str(SPINE_SRC))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

TEST_CONFIG = str(ROOT / "configs" / "em-hsd-v2-test.yaml")
SYNTH_CSV = str(ROOT.parent / "Johnny t0-1.03" / "data" / "synthetic_dev.csv")


@pytest.fixture
def config_path():
    return TEST_CONFIG


@pytest.fixture
def cfg():
    from em_hsd import load_em_hsd_config
    return load_em_hsd_config(TEST_CONFIG)


@pytest.fixture
def synth_path():
    assert os.path.exists(SYNTH_CSV), "synthetic_dev.csv missing in Johnny t0-1.03"
    return SYNTH_CSV
