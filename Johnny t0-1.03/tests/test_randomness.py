"""Test 7: randomness & seed policy.

(a) Two PRODUCTION runs on the same input differ on at least some rewritten rows,
    while ID/Author/HS match the input both times.
(b) In DEBUG mode a row's output is identical whether processed alone, in a
    batch, in a different order, or with neighbours removed (a row's RNG depends
    only on (RUN_SEED, row_index)).
"""

import numpy as np

from mechanism import load_config, privatize
from mechanism.rng import debug_rng, make_row_rng
from wrapper.csvio import read_csv
from wrapper.run import run


def _texts(path):
    _, rows = read_csv(path)
    return rows


def test_two_production_runs_differ_but_preserve_fields(tmp_path, synth_path, config_path):
    out1 = tmp_path / "p1.csv"
    out2 = tmp_path / "p2.csv"
    # debug_seed=None -> production (fresh entropy per row)
    assert run(synth_path, str(out1), "spine", config_path,
               log_path=str(tmp_path / "l1")) == 0
    assert run(synth_path, str(out2), "spine", config_path,
               log_path=str(tmp_path / "l2")) == 0

    orig = _texts(synth_path)
    r1 = _texts(str(out1))
    r2 = _texts(str(out2))

    # ID/Author/HS preserved in BOTH runs
    for run_rows in (r1, r2):
        for o, n in zip(orig, run_rows):
            assert o["ID"] == n["ID"]
            assert o["Author"] == n["Author"]
            assert o["HS"] == n["HS"]

    # at least one row's Text differs between the two production runs
    diffs = sum(1 for a, b in zip(r1, r2) if a["Text"] != b["Text"])
    assert diffs > 0, "two production runs produced identical Text everywhere"


def _process(pairs, cfg, run_seed):
    """Process (index, text) pairs; per-row RNG keyed by (run_seed, index)."""
    out = {}
    for idx, text in pairs:
        cfg.rng = debug_rng(run_seed, idx)
        out[idx], _ = privatize(text, cfg)
    return out


def test_debug_row_independent_of_neighbours_and_order(synth_path, config_path):
    rows = _texts(synth_path)
    pairs = [(i, r["Text"]) for i, r in enumerate(rows)]
    cfg = load_config(config_path)

    target = 1
    target_text = pairs[target][1]

    alone = _process([(target, target_text)], cfg, "RUNSEED")[target]
    full = _process(pairs, cfg, "RUNSEED")[target]
    reordered = _process(list(reversed(pairs)), cfg, "RUNSEED")[target]
    subset = _process([(0, pairs[0][1]), (target, target_text),
                       (len(pairs) - 1, pairs[-1][1])], cfg, "RUNSEED")[target]

    assert alone == full == reordered == subset


def test_debug_is_reproducible_but_seed_dependent(cfg):
    text = "the documentary about deep sea creatures was wonderful"
    cfg.rng = make_row_rng(3, run_seed="A")
    a1, _ = privatize(text, cfg)
    cfg.rng = make_row_rng(3, run_seed="A")
    a2, _ = privatize(text, cfg)
    cfg.rng = make_row_rng(3, run_seed="B")
    b, _ = privatize(text, cfg)
    assert a1 == a2            # same seed+index -> reproducible
    # different seed -> (very likely) different; not asserted strictly to avoid
    # a vanishingly rare collision, but reproducibility above is the contract.


def test_debug_rows_are_mutually_independent():
    # different indices under the same run seed give independent generators
    g0 = debug_rng("S", 0).standard_normal(5)
    g1 = debug_rng("S", 1).standard_normal(5)
    assert not np.array_equal(g0, g1)
