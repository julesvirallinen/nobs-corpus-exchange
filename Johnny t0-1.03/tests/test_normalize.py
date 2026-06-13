"""Test 4: normaliser — idempotent, deterministic, logged."""

import pytest

from mechanism.normalize import normalize_text
from mechanism.rng import make_row_rng

SAMPLES = [
    "Sooo GOOOD and reallyyy niiice!!!",
    "He said “hello” and left…",
    "MiXeD CaSe with   weird    spacing",
    "emoji \U0001F600\U0001F525 here",
    "café NAÏVE Über",
    "u r gonna luv this tho",
    "",
    "no-change text already normal",
]


@pytest.mark.parametrize("text", SAMPLES)
def test_idempotent(text, cfg):
    once = normalize_text(text, cfg)
    twice = normalize_text(once, cfg)
    assert twice == once, f"not idempotent: {once!r} -> {twice!r}"


@pytest.mark.parametrize("text", SAMPLES)
def test_deterministic(text, cfg):
    assert normalize_text(text, cfg) == normalize_text(text, cfg)


def test_de_elongation(cfg):
    assert normalize_text("sooo", cfg) == "so"
    assert normalize_text("yesss", cfg) == "yes"
    assert "hello" == normalize_text("hellooo", cfg)


def test_lowercasing(cfg):
    assert normalize_text("HELLO World", cfg) == "hello world"


def test_normalisation_is_logged(cfg):
    from mechanism import privatize
    cfg.rng = make_row_rng(0, run_seed="N")
    _, log = privatize("Sooo GOOOD", cfg)
    actions = {e["original"]: e for e in log}
    # 'Sooo' is a function word after normalisation? it's content; either way it
    # was normalised (lowercased + de-elongated) and that is recorded.
    entry = actions["Sooo"]
    assert "normalised" in (entry["reason"] or "") or entry["action"] == "rewritten"
    assert entry["normalized"] == "so"
