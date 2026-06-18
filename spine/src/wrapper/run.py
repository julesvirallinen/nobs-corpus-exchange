from __future__ import annotations

import argparse
import json
import sys
import time
from typing import List, Optional

# Make stdout/stderr UTF-8 so emoji/non-ASCII never crash on Windows consoles.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass

from mechanism import identity, load_config, privatize
from mechanism.rng import make_row_rng
from mechanism.spine import get_resources, lexicon_warning

from .csvio import CsvContractError, read_csv, write_csv
from .diffcheck import DiffCheckError, assert_preserved

MODES = ("identity", "dpmlm", "spine")


def _apply_mode(config, mode: str) -> None:
    if mode == "dpmlm":
        config.protection_enabled = False
        config.uniform_budget = True
    elif mode == "spine":
        config.protection_enabled = True
        config.uniform_budget = False
    # identity: mechanism path bypasses config entirely.


def _default_log_path(out_path: str) -> str:
    return out_path + ".log.jsonl"


def run(in_path: str, out_path: str, mode: str, config_path: str,
        debug_seed: Optional[str] = None, log_path: Optional[str] = None,
        backend: Optional[str] = None) -> int:
    # 1. read + validate input
    try:
        fieldnames, rows = read_csv(in_path)
    except FileNotFoundError:
        print(f"ERROR: input file not found: {in_path!r}", file=sys.stderr)
        return 2
    except CsvContractError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    config = load_config(config_path)
    if backend:
        config.mlm.backend = backend
    _apply_mode(config, mode)

    if debug_seed is not None:
        print(
            "WARNING: --debug-seed makes the run reproducible. Debug mode is for "
            "development ONLY and must NOT be used to produce a submission.",
            file=sys.stderr,
        )

    # 2. build resources once (fail early with a clear message) + surface warnings
    if mode != "identity":
        try:
            get_resources(config)
        except ImportError as exc:
            print(
                f"ERROR: backend {config.mlm.backend!r} needs optional deps "
                f"({exc}). Install requirements-hf.txt, or use a config with "
                "mlm.backend: hash (e.g. configs/fast.yaml).",
                file=sys.stderr,
            )
            return 4
        except Exception as exc:
            print(f"ERROR: could not initialise mechanism: {exc}", file=sys.stderr)
            return 4
        warn = lexicon_warning(config)
        if warn:
            print(f"WARNING: {warn}", file=sys.stderr)

    # 3. transform rows (Text only)
    log_path = log_path or _default_log_path(out_path)
    out_rows: List[dict] = []
    n_rewritten = 0
    start = time.time()
    try:
        with open(log_path, "w", encoding="utf-8", newline="") as logf:
            for idx, row in enumerate(rows):
                text = row["Text"]
                if mode == "identity":
                    new_text, token_log = identity(text)
                else:
                    config.rng = make_row_rng(idx, run_seed=debug_seed)
                    new_text, token_log = privatize(text, config)
                if any(t["action"] == "rewritten" for t in token_log):
                    n_rewritten += 1
                new_row = dict(row)
                new_row["Text"] = new_text
                out_rows.append(new_row)
                # JSONL keyed by row index ONLY — never Author.
                logf.write(json.dumps(
                    {"row": idx, "n_tokens": len(token_log), "tokens": token_log},
                    ensure_ascii=False,
                ) + "\n")
    except KeyError:
        print("ERROR: a row is missing the 'Text' column value.", file=sys.stderr)
        return 2

    # 4. write output preserving all other fields
    try:
        write_csv(out_path, fieldnames, out_rows)
    except OSError as exc:
        print(f"ERROR: could not write output {out_path!r}: {exc}", file=sys.stderr)
        return 2

    # 5. built-in diff check (loud failure)
    try:
        assert_preserved(in_path, out_path)
    except DiffCheckError as exc:
        print(str(exc), file=sys.stderr)
        return 3

    elapsed = time.time() - start
    rate = (len(rows) / elapsed) if elapsed > 0 else float("inf")
    print(
        f"OK [{mode}] {len(rows)} rows -> {out_path}  "
        f"(rewritten in {n_rewritten} rows)  log: {log_path}",
        file=sys.stderr,
    )
    print(f"diff check PASSED. throughput ~{rate:.1f} rows/s (runtime only).",
          file=sys.stderr)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m wrapper.run",
        description="Privatise the Text column of a CSV (ID,Author,Text,HS).",
    )
    p.add_argument("--in", dest="in_path", required=True, help="input CSV path")
    p.add_argument("--out", dest="out_path", required=True, help="output CSV path")
    p.add_argument("--mode", choices=MODES, required=True)
    p.add_argument("--config", dest="config_path", default="configs/default.yaml")
    p.add_argument("--debug-seed", dest="debug_seed", default=None,
                   help="DEV ONLY: reproducible per-row RNG = hash(SEED, row). "
                        "Never use for a submission.")
    p.add_argument("--logs", dest="log_path", default=None,
                   help="JSONL token-log path (default: <out>.log.jsonl)")
    p.add_argument("--mlm-backend", dest="backend", default=None,
                   choices=("hf", "hash"),
                   help="override mlm.backend from the config")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args.in_path, args.out_path, args.mode, args.config_path,
                   debug_seed=args.debug_seed, log_path=args.log_path,
                   backend=args.backend)
    except KeyboardInterrupt:
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
