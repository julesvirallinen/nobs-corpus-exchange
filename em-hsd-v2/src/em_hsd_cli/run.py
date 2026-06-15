"""CSV CLI for EM-HSD 2.0 (uses Johnny t0-1.03 wrapper CSV utilities)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import List, Optional

from em_hsd import load_em_hsd_config, privatize_em_hsd_v2
from em_hsd import spine_bootstrap as _spine_bootstrap  # noqa: F401 — path setup
from em_hsd.config import resolve_config_path
from em_hsd.resources import init_spine_resources

from mechanism.rng import make_row_rng
from wrapper.csvio import CsvContractError, read_csv, write_csv
from wrapper.diffcheck import DiffCheckError, assert_preserved


def _default_log_path(out_path: str) -> str:
    return out_path + ".log.jsonl"


def run(
    in_path: str,
    out_path: str,
    config_path: str,
    debug_seed: Optional[str] = None,
    log_path: Optional[str] = None,
) -> int:
    try:
        fieldnames, rows = read_csv(in_path)
    except FileNotFoundError:
        print(f"ERROR: input file not found: {in_path!r}", file=sys.stderr)
        return 2
    except CsvContractError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    cfg_path = resolve_config_path(config_path)
    config = load_em_hsd_config(cfg_path)

    if debug_seed is not None:
        print(
            "WARNING: --debug-seed is for development ONLY.",
            file=sys.stderr,
        )

    try:
        init_spine_resources(config)
    except ImportError as exc:
        print(f"ERROR: missing optional deps: {exc}", file=sys.stderr)
        return 4
    except Exception as exc:
        print(f"ERROR: could not initialise resources: {exc}", file=sys.stderr)
        return 4

    log_path = log_path or _default_log_path(out_path)
    out_rows: List[dict] = []
    n_changed = 0
    start = time.time()

    try:
        with open(log_path, "w", encoding="utf-8", newline="") as logf:
            for idx, row in enumerate(rows):
                text = row["Text"]
                config.spine.rng = make_row_rng(idx, run_seed=debug_seed)
                new_text, audit = privatize_em_hsd_v2(text, config)
                if new_text != text:
                    n_changed += 1
                new_row = dict(row)
                new_row["Text"] = new_text
                out_rows.append(new_row)
                logf.write(json.dumps(
                    {"row": idx, "mode": "em-hsd-v2", "audit": audit},
                    ensure_ascii=False,
                ) + "\n")
    except KeyError:
        print("ERROR: a row is missing the 'Text' column value.", file=sys.stderr)
        return 2

    try:
        write_csv(out_path, fieldnames, out_rows)
    except OSError as exc:
        print(f"ERROR: could not write output {out_path!r}: {exc}", file=sys.stderr)
        return 2

    try:
        assert_preserved(in_path, out_path)
    except DiffCheckError as exc:
        print(str(exc), file=sys.stderr)
        return 3

    elapsed = time.time() - start
    rate = (len(rows) / elapsed) if elapsed > 0 else float("inf")
    print(
        f"OK [em-hsd-v2] {len(rows)} rows -> {out_path}  "
        f"(changed in {n_changed} rows)  log: {log_path}",
        file=sys.stderr,
    )
    print(f"diff check PASSED. throughput ~{rate:.1f} rows/s.", file=sys.stderr)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="em-hsd-run",
        description="Privatise Text column via EM-HSD 2.0 (layer-4 only).",
    )
    p.add_argument("--in", dest="in_path", required=True)
    p.add_argument("--out", dest="out_path", required=True)
    p.add_argument("--config", dest="config_path", default="em-hsd-v2-test.yaml")
    p.add_argument("--debug-seed", dest="debug_seed", default=None)
    p.add_argument("--logs", dest="log_path", default=None)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(
            args.in_path, args.out_path, args.config_path,
            debug_seed=args.debug_seed, log_path=args.log_path,
        )
    except KeyboardInterrupt:
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
