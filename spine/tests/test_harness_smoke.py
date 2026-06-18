"""Test 9: harness smoke test, end to end on synthetic data.

* identity mode -> TO ~ 0 (utility ratio ~ 1, privacy ratio ~ 1).
* spine mode    -> lower re-identification than identity on the synthetic set.
"""

import pytest

from harness.evaluate import evaluate
from wrapper.run import run


@pytest.fixture
def outputs(tmp_path, synth_path, config_path):
    ident = tmp_path / "identity.csv"
    spine = tmp_path / "spine.csv"
    assert run(synth_path, str(ident), "identity", config_path,
               log_path=str(tmp_path / "li")) == 0
    assert run(synth_path, str(spine), "spine", config_path, debug_seed="SMOKE",
               log_path=str(tmp_path / "ls")) == 0
    return str(ident), str(spine)


def test_identity_trade_off_is_about_zero(outputs, synth_path, config_path):
    ident, _ = outputs
    rep = evaluate(synth_path, ident, "proxy", config_path, False, "distilroberta-base")
    to = rep["trade_off"]
    assert abs(to["utility_ratio"] - 1.0) < 1e-6
    assert abs(to["privacy_ratio"] - 1.0) < 1e-6
    assert abs(to["trade_off_estimate"]) < 1e-6


def test_spine_reduces_reidentification(outputs, synth_path, config_path):
    ident, spine = outputs
    rep_id = evaluate(synth_path, ident, "proxy", config_path, False, "distilroberta-base")
    rep_sp = evaluate(synth_path, spine, "proxy", config_path, False, "distilroberta-base")

    reid_id = rep_id["reidentification"]["privacy_privatized"]
    reid_sp = rep_sp["reidentification"]["privacy_privatized"]
    assert reid_sp < reid_id, (
        f"spine reident ({reid_sp}) not lower than identity ({reid_id})"
    )
    # and spine's own privatised reident is below its original-text upper bound
    assert (rep_sp["reidentification"]["privacy_privatized"]
            < rep_sp["reidentification"]["privacy_original"])


def test_harness_reads_author_only_from_original(outputs, synth_path, config_path):
    # The privatized file has no Author column at all in the contract sense; the
    # harness must still work, taking Author solely from the original file.
    ident, spine = outputs
    rep = evaluate(synth_path, spine, "proxy", config_path, False, "distilroberta-base")
    assert rep["n_rows"] > 0
    assert "utility" in rep and "reidentification" in rep
