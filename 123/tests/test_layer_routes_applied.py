"""Layers 1 & 2 routes must actually change token_sanitize behaviour.

These guard the wiring that was previously dead: Layer 1 protect routes keep a
content token verbatim, and Layer 2 force-sanitise routes rewrite a function
word that would otherwise be kept.
"""

from __future__ import annotations

from mechanism.rng import make_row_rng

from em_hsd.core.sanitize import token_sanitize


def _entry(log, word):
    for e in log:
        if (e.get("original") or "").lower().strip(".,!?") == word:
            return e
    return None


def test_layer1_protected_content_token_kept(cfg):
    cfg.spine.rng = make_row_rng(0, run_seed="t")
    eps = cfg.em_hsd_v2.epsilon_1
    _, log = token_sanitize(
        "those people gather downtown", cfg, eps, protected_tokens={"people"}
    )
    e = _entry(log, "people")
    assert e is not None
    assert e["action"] == "kept"
    assert "Layer 1" in e["reason"]
    assert e["replacement"] == "people"


def test_layer2_force_sanitizes_function_word(cfg):
    eps = cfg.em_hsd_v2.epsilon_1
    text = "you should think about this"

    cfg.spine.rng = make_row_rng(0, run_seed="t")
    _, plain = token_sanitize(text, cfg, eps)
    cfg.spine.rng = make_row_rng(0, run_seed="t")
    _, forced = token_sanitize(text, cfg, eps, force_sanitize={"you"})

    assert _entry(plain, "you")["action"] == "kept"        # default: function word kept
    assert _entry(forced, "you")["action"] == "rewritten"  # Layer 2: forced to ε₁ rewrite


def test_no_routes_is_unchanged_baseline(cfg):
    cfg.spine.rng = make_row_rng(0, run_seed="t")
    eps = cfg.em_hsd_v2.epsilon_1
    _, log = token_sanitize("you should think about this", cfg, eps)
    assert _entry(log, "you")["action"] == "kept"
