"""Shared test fixtures. All tests use configs/test.yaml: the deterministic
`hash` MLM backend and an in-config test lexicon of MILD placeholder terms.
No network, no model downloads, no real slurs.
"""

import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TEST_CONFIG = str(ROOT / "configs" / "test.yaml")
SYNTH_CSV = str(ROOT / "data" / "synthetic_dev.csv")


@pytest.fixture
def config_path():
    return TEST_CONFIG


@pytest.fixture
def cfg():
    from mechanism import load_config
    return load_config(TEST_CONFIG)


@pytest.fixture
def synth_path():
    assert os.path.exists(SYNTH_CSV), (
        "data/synthetic_dev.csv missing — run scripts/make_synthetic.py"
    )
    return SYNTH_CSV
