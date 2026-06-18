"""Test 8: per-token log accounts for every input token; JSONL parses; the log
is keyed by row index and contains no Author/ID fields."""

import json

import pytest

from mechanism import privatize
from mechanism.rng import make_row_rng
from mechanism.spine import _build_segments, get_resources
from wrapper.csvio import read_csv
from wrapper.run import run

ALLOWED_ACTIONS = {"normalised", "protected+canonicalised", "rewritten", "kept"}
TOKEN_KEYS = {"original", "normalized", "action", "replacement", "epsilon",
              "token_class", "reason"}


def test_segmentation_is_lossless(synth_path, cfg):
    lexicon = get_resources(cfg)[0]
    _, rows = read_csv(synth_path)
    for r in rows:
        segs = _build_segments(r["Text"], lexicon)
        assert "".join(s.original for s in segs) == r["Text"]


def test_log_accounts_for_every_token(synth_path, cfg):
    lexicon = get_resources(cfg)[0]
    _, rows = read_csv(synth_path)
    for i, r in enumerate(rows):
        cfg.rng = make_row_rng(i, run_seed="LOG")
        _, log = privatize(r["Text"], cfg)
        segs = _build_segments(r["Text"], lexicon)
        n_tokens = sum(1 for s in segs if s.is_token)
        assert len(log) == n_tokens, f"row {i}: {len(log)} log != {n_tokens} tokens"
        for e in log:
            assert set(e.keys()) == TOKEN_KEYS
            assert e["action"] in ALLOWED_ACTIONS
            assert e["original"] is not None


def test_jsonl_parses_and_is_author_free(tmp_path, synth_path, config_path):
    out = tmp_path / "spine.csv"
    log_path = tmp_path / "spine.jsonl"
    assert run(synth_path, str(out), "spine", config_path, debug_seed="LOG",
               log_path=str(log_path)) == 0

    lines = log_path.read_text(encoding="utf-8").splitlines()
    _, rows = read_csv(synth_path)
    assert len(lines) == len(rows)
    for idx, line in enumerate(lines):
        obj = json.loads(line)                       # parses
        assert set(obj.keys()) == {"row", "n_tokens", "tokens"}
        assert obj["row"] == idx                     # keyed by row index
        # no Author / ID field names anywhere in the record
        blob = json.dumps(obj).lower()
        assert '"author"' not in blob
        assert '"id"' not in blob
        for t in obj["tokens"]:
            assert set(t.keys()) == TOKEN_KEYS
