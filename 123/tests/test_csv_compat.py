"""CSV alias + checkpoint contract tests."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from em_hsd.cli.run import _checkpoint_path, _save_checkpoint, run
from em_hsd.csv_compat import (
    assert_preserved_compat,
    read_csv_compat,
    write_canonical_csv,
    write_canonical_privatized_csv,
)

REDDIT_HEADER = "ID,author,text,hs\n"
REDDIT_ROW = 'REDDIT_0,13,"hello world",1\n'


@pytest.fixture
def reddit_csv(tmp_path: Path) -> Path:
    path = tmp_path / "reddit.csv"
    path.write_text(REDDIT_HEADER + REDDIT_ROW, encoding="utf-8")
    return path


def test_read_lowercase_aliases(reddit_csv: Path):
    fieldnames, rows, column_map = read_csv_compat(str(reddit_csv))
    assert fieldnames == ["ID", "author", "text", "hs"]
    assert column_map["Text"] == "text"
    assert rows[0]["Text"] == "hello world"
    assert rows[0]["Author"] == "13"


def test_canonical_views_roundtrip(reddit_csv: Path, tmp_path: Path):
    _, rows, _ = read_csv_compat(str(reddit_csv))
    orig = tmp_path / "orig.csv"
    priv = tmp_path / "priv.csv"
    write_canonical_csv(str(orig), rows)
    mutated = [dict(rows[0], Text="goodbye world")]
    write_canonical_privatized_csv(str(priv), mutated)
    with open(orig, encoding="utf-8") as fh:
        assert csv.DictReader(fh).fieldnames == ["ID", "Author", "Text", "HS"]
    with open(priv, encoding="utf-8") as fh:
        assert list(csv.DictReader(fh))[0]["Text"] == "goodbye world"


def test_diff_check_accepts_alias_headers(reddit_csv: Path, tmp_path: Path):
    out = tmp_path / "out.csv"
    out.write_text(
        'ID,author,text,hs\nREDDIT_0,13,"changed text",1\n',
        encoding="utf-8",
    )
    assert_preserved_compat(str(reddit_csv), str(out))


def test_run_mock_checkpoint_resume(reddit_csv: Path, tmp_path: Path):
    config = Path(__file__).resolve().parents[1] / "configs" / "em-hsd-v2-test.yaml"
    out = tmp_path / "out.csv"
    rc = run(
        str(reddit_csv), str(out), str(config),
        debug_seed="CKPT", resume=False,
    )
    assert rc == 0
    assert out.exists()
    ckpt = _checkpoint_path(str(out))
    assert not Path(ckpt).exists()

    # Simulate interrupted second row on a 1-row file: checkpoint cleaned on success
    _save_checkpoint(ckpt, 1, 1)
    rc2 = run(str(reddit_csv), str(out), str(config), debug_seed="CKPT", resume=True)
    assert rc2 == 0
