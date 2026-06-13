"""Test 10: CLI robustness — bad paths, malformed/contract-violating CSV, and
missing columns fail loudly with clear messages and correct exit codes."""

import pytest

from wrapper.diffcheck import DiffCheckError, assert_preserved, diff_field_values
from wrapper.run import main, run

GOOD = "ID,Author,Text,HS\n1,alice,hello world,0\n2,bob,another post,1\n"


def _write(tmp_path, name, content):
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return str(p)


def test_missing_input_file(tmp_path, config_path, capsys):
    out = tmp_path / "o.csv"
    rc = run(str(tmp_path / "nope.csv"), str(out), "spine", config_path,
             log_path=str(tmp_path / "l"))
    assert rc == 2
    assert "not found" in capsys.readouterr().err.lower()


def test_missing_required_column(tmp_path, config_path, capsys):
    bad = _write(tmp_path, "bad.csv", "ID,Author,Text\n1,alice,hi\n")  # no HS
    rc = run(bad, str(tmp_path / "o.csv"), "spine", config_path,
             log_path=str(tmp_path / "l"))
    assert rc == 2
    err = capsys.readouterr().err.lower()
    assert "missing" in err and "hs" in err


def test_ragged_row_too_few_fields(tmp_path, config_path, capsys):
    bad = _write(tmp_path, "ragged.csv", "ID,Author,Text,HS\n1,alice,hi\n")  # 3 vals
    rc = run(bad, str(tmp_path / "o.csv"), "identity", config_path,
             log_path=str(tmp_path / "l"))
    assert rc == 2
    assert "ragged" in capsys.readouterr().err.lower()


def test_empty_file(tmp_path, config_path, capsys):
    bad = _write(tmp_path, "empty.csv", "")
    rc = run(bad, str(tmp_path / "o.csv"), "identity", config_path,
             log_path=str(tmp_path / "l"))
    assert rc == 2
    assert "empty" in capsys.readouterr().err.lower()


def test_good_run_via_main(tmp_path, config_path):
    good = _write(tmp_path, "good.csv", GOOD)
    out = tmp_path / "out.csv"
    rc = main(["--in", good, "--out", str(out), "--mode", "identity",
               "--config", config_path, "--logs", str(tmp_path / "l")])
    assert rc == 0
    assert out.exists()


def test_unwritable_output_path(tmp_path, config_path, capsys):
    good = _write(tmp_path, "good.csv", GOOD)
    # directory that does not exist -> OSError on open
    bad_out = tmp_path / "no_such_dir" / "out.csv"
    rc = run(good, str(bad_out), "identity", config_path,
             log_path=str(tmp_path / "l"))
    assert rc == 2


def test_diff_check_detects_tampering(tmp_path, config_path):
    good = _write(tmp_path, "good.csv", GOOD)
    tampered = _write(tmp_path, "tampered.csv",
                      "ID,Author,Text,HS\n1,EVE,hello world,0\n2,bob,another post,1\n")
    problems = diff_field_values(
        [{"ID": "1", "Author": "alice", "HS": "0"}],
        [{"ID": "1", "Author": "EVE", "HS": "0"}],
    )
    assert problems and "Author" in problems[0]
    with pytest.raises(DiffCheckError):
        assert_preserved(good, tampered)


def test_diff_check_detects_row_count_change():
    problems = diff_field_values(
        [{"ID": "1", "Author": "a", "HS": "0"}],
        [{"ID": "1", "Author": "a", "HS": "0"}, {"ID": "2", "Author": "b", "HS": "1"}],
    )
    assert any("row count" in p for p in problems)
