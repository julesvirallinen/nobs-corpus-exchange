#!/usr/bin/env python3
"""Run Reddit slice eval and print TO + pipeline diagnostics."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _parse_to(log_text: str) -> dict | None:
    out = {}
    for line in log_text.splitlines():
        if "utility_ratio" in line:
            out["utility_ratio"] = float(line.split("=")[-1].strip())
        if "privacy_ratio" in line:
            out["privacy_ratio"] = float(line.split("=")[-1].strip())
        if line.strip().startswith("TO ="):
            out["to"] = float(line.split("=")[-1].strip())
        if "ensemble utility:" in line:
            parts = line.split("original=")[-1]
            o, p = parts.split("privatized=")
            out["f1_orig"] = float(o.strip())
            out["f1_priv"] = float(p.strip())
    return out if "to" in out else None


def _audit_stats(log_path: Path) -> dict:
    fb = kv = echo = 0
    rejects: Counter = Counter()
    n = 0
    with log_path.open(encoding="utf-8") as fh:
        for line in fh:
            a = json.loads(line)["audit"]
            n += 1
            if a.get("fallback"):
                fb += 1
            if (a.get("k_valid") or 0) >= 1:
                kv += 1
            echo += len(a.get("echo_dropped") or [])
            for d in a.get("filter_details") or []:
                if d.get("reject"):
                    rejects[d["reject"]] += 1
    return {
        "rows": n,
        "fallback_rate": fb / n if n else 0,
        "k_valid_rate": kv / n if n else 0,
        "echo_dropped": echo,
        "rejects": dict(rejects),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_path", default="/tmp/reddit_25_slice.csv")
    p.add_argument("--out", dest="out_path", required=True)
    p.add_argument("--config", default="configs/em-hsd-v2-qwen-reddit-tune.yaml")
    args = p.parse_args()

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_reddit_eval.py"),
        "--in", args.in_path,
        "--out", args.out_path,
        "--config", args.config,
        "--no-resume",
        "--utility-backend", "hf",
    ]
    print("RUN:", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    print(proc.stdout)
    if proc.stderr:
        print(proc.stderr, file=sys.stderr)
    if proc.returncode != 0:
        return proc.returncode

    metrics = _parse_to(proc.stdout)
    log_path = Path(args.out_path + ".log.jsonl")
    stats = _audit_stats(log_path) if log_path.is_file() else {}
    report = {"metrics": metrics, "audit": stats, "out": args.out_path}
    print("REPORT:", json.dumps(report, indent=2))
    return 0 if metrics and metrics.get("to", -1) > 0.5 else 1


if __name__ == "__main__":
    raise SystemExit(main())
