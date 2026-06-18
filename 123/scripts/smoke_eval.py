"""End-to-end smoke evaluation on a small synthetic CSV (no model downloads)."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from em_hsd.cli.run import run
from em_hsd.csv_compat import read_csv_compat
from em_hsd.harness_integration import evaluate_with_harness


def _diagnostics(in_path: str, out_path: str, log_path: str | None) -> dict[str, Any]:
    _, in_rows, _ = read_csv_compat(in_path)
    _, out_rows, _ = read_csv_compat(out_path)
    n = len(in_rows)
    changed = sum(1 for i, r in enumerate(out_rows) if r["Text"] != in_rows[i]["Text"])
    fallback = 0
    avg_k_valid: list[int] = []
    if log_path and Path(log_path).exists():
        for line in Path(log_path).read_text(encoding="utf-8").strip().splitlines():
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            audit = obj.get("audit", {})
            if audit.get("fallback"):
                fallback += 1
            avg_k_valid.append(audit.get("k_valid", 0))
    return {
        "rows": n,
        "changed": changed,
        "changed_rate": changed / n if n else 0.0,
        "fallback": fallback,
        "fallback_rate": fallback / n if n else 0.0,
        "avg_k_valid": sum(avg_k_valid) / len(avg_k_valid) if avg_k_valid else 0.0,
    }


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Smoke-evaluate EM-HSD v2 on a CSV.")
    p.add_argument("--in", dest="in_path", required=True)
    p.add_argument("--out", dest="out_path", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--mode", choices=("em-hsd-v2", "triage-dp"), default="em-hsd-v2")
    p.add_argument("--evaluate", action="store_true", help="run harness integration stub if available")
    p.add_argument("--debug-seed", default="SMOKE")
    args = p.parse_args(argv)

    log_path = args.out_path + ".log.jsonl"
    rc = run(
        args.in_path,
        args.out_path,
        args.config,
        mode=args.mode,
        debug_seed=args.debug_seed,
        log_path=log_path,
        resume=False,
    )
    if rc != 0:
        return rc

    diag = _diagnostics(args.in_path, args.out_path, log_path)
    print(json.dumps(diag, indent=2), file=sys.stderr)

    if args.evaluate:
        harness_report = evaluate_with_harness(args.in_path, args.out_path, args.config)
        print(json.dumps(harness_report, indent=2), file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
