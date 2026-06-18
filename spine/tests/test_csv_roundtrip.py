"""Test 2: adversarial CSV round-trip — every edge case survives with other
fields intact (commas, double quotes, newlines, emoji, non-ASCII in any field)."""

import pytest

from wrapper.csvio import read_csv, write_csv
from wrapper.run import run

ADVERSARIAL_ROWS = [
    {"ID": "id,001", "Author": 'O\'Brien, "Jr."', "Text": "plain text", "HS": "0"},
    {"ID": "id\t002", "Author": "alice", "Text": "comma, inside, text", "HS": "1"},
    {"ID": "id003", "Author": 'quote"author', "Text": 'he said "hi" loudly', "HS": "0"},
    {"ID": "id004", "Author": "bob", "Text": "line1\nline2\nline3", "HS": "1"},
    {"ID": "id005", "Author": "café", "Text": "emoji \U0001F600 and \U0001F525", "HS": "0"},
    {"ID": "id006", "Author": "däve", "Text": "non-ascii: über naïve Москва", "HS": "1"},
    {"ID": "id007", "Author": "erin", "Text": "", "HS": "0"},
    {"ID": "id008", "Author": "frank", "Text": "z1bb3r and d u m m y obfuscated", "HS": "1"},
    {"ID": "id009", "Author": "grace", "Text": "url http://x.co/a,b and @mention #tag", "HS": "0"},
]
FIELDS = ["ID", "Author", "Text", "HS"]


@pytest.fixture
def adversarial_csv(tmp_path):
    path = tmp_path / "adversarial.csv"
    write_csv(str(path), FIELDS, ADVERSARIAL_ROWS)
    return str(path)


def test_input_roundtrips_unchanged(adversarial_csv):
    _, rows = read_csv(adversarial_csv)
    assert len(rows) == len(ADVERSARIAL_ROWS)
    for got, want in zip(rows, ADVERSARIAL_ROWS):
        assert got == {k: want[k] for k in FIELDS}


@pytest.mark.parametrize("mode", ["identity", "dpmlm", "spine"])
def test_edge_cases_survive_each_mode(mode, adversarial_csv, tmp_path, config_path):
    out = tmp_path / f"out_{mode}.csv"
    rc = run(adversarial_csv, str(out), mode, config_path,
             debug_seed="RT", log_path=str(tmp_path / f"{mode}.jsonl"))
    assert rc == 0
    _, original = read_csv(adversarial_csv)
    _, output = read_csv(str(out))
    assert len(original) == len(output)
    for o, n in zip(original, output):
        assert o["ID"] == n["ID"]
        assert o["Author"] == n["Author"]
        assert o["HS"] == n["HS"]
