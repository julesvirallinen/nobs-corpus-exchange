"""Test 3: author isolation + module boundaries.

* The mechanism's public API receives only (text, config) — no column access.
* The string 'Author' appears nowhere in src/mechanism/.
* The mechanism imports nothing from the harness; the harness imports nothing
  from the mechanism. These boundaries are enforced architecturally.
"""

import inspect
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MECH = ROOT / "src" / "mechanism"
HARNESS = ROOT / "src" / "harness"


def test_privatize_signature_is_text_and_config_only():
    from mechanism import privatize
    params = list(inspect.signature(privatize).parameters)
    assert params == ["text", "config"], params


def test_identity_signature_is_text_only():
    from mechanism import identity
    params = list(inspect.signature(identity).parameters)
    assert params == ["text"], params


def test_word_author_absent_from_mechanism_sources():
    offenders = []
    for py in MECH.rglob("*.py"):
        text = py.read_text(encoding="utf-8")
        if "Author" in text or "author" in text:
            offenders.append(py.name)
    assert not offenders, (
        f"'author' must not appear in src/mechanism/: {offenders}"
    )


def test_mechanism_does_not_import_harness_or_wrapper():
    for py in MECH.rglob("*.py"):
        text = py.read_text(encoding="utf-8")
        assert "import harness" not in text and "from harness" not in text, py.name
        assert "import wrapper" not in text and "from wrapper" not in text, py.name


def test_harness_does_not_import_mechanism():
    for py in HARNESS.rglob("*.py"):
        text = py.read_text(encoding="utf-8")
        assert "import mechanism" not in text and "from mechanism" not in text, py.name


def test_privatize_only_sees_text(cfg):
    """Passing a full row dict is impossible by signature; the mechanism only
    ever receives the Text string."""
    from mechanism import privatize
    from mechanism.rng import make_row_rng
    cfg.rng = make_row_rng(0, run_seed="ISO")
    out, log = privatize("you are a dummy and a z1bb3r", cfg)
    assert isinstance(out, str)
    assert isinstance(log, list)
