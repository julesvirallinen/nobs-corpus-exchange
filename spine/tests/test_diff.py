"""Test 1: diff test for every mode — ID/Author/HS byte-identical, rows 1:1."""

import pytest

from wrapper.csvio import read_csv
from wrapper.run import run

MODES = ("identity", "dpmlm", "spine")


@pytest.mark.parametrize("mode", MODES)
def test_diff_preserves_other_fields(mode, tmp_path, synth_path, config_path):
    out = tmp_path / f"{mode}.csv"
    log = tmp_path / f"{mode}.jsonl"
    rc = run(synth_path, str(out), mode, config_path,
             debug_seed="DIFF", log_path=str(log))
    assert rc == 0, f"{mode} run failed (rc={rc})"

    _, original = read_csv(synth_path)
    _, output = read_csv(str(out))
    assert len(original) == len(output)
    for i, (o, n) in enumerate(zip(original, output)):
        assert o["ID"] == n["ID"], f"row {i} ID changed"
        assert o["Author"] == n["Author"], f"row {i} Author changed"
        assert o["HS"] == n["HS"], f"row {i} HS changed"


def test_identity_leaves_text_unchanged(tmp_path, synth_path, config_path):
    out = tmp_path / "id.csv"
    run(synth_path, str(out), "identity", config_path, log_path=str(tmp_path / "l"))
    _, original = read_csv(synth_path)
    _, output = read_csv(str(out))
    for o, n in zip(original, output):
        assert o["Text"] == n["Text"]


def test_spine_changes_some_text(tmp_path, synth_path, config_path):
    out = tmp_path / "sp.csv"
    run(synth_path, str(out), "spine", config_path, debug_seed="X",
        log_path=str(tmp_path / "l"))
    _, original = read_csv(synth_path)
    _, output = read_csv(str(out))
    changed = sum(1 for o, n in zip(original, output) if o["Text"] != n["Text"])
    assert changed > 0, "spine mode changed no Text at all"
