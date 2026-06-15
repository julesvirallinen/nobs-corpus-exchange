#!/usr/bin/env python3
"""Run EM-HSD v2 with Qwen on hate rows only (faster CPU smoke)."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPINE = ROOT.parent / "Johnny t0-1.03"
SYNTH = SPINE / "data" / "synthetic_dev.csv"


def main() -> int:
    config = sys.argv[1] if len(sys.argv) > 1 else "configs/em-hsd-v2-qwen.yaml"
    rows = []
    with open(SYNTH, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames
        for row in reader:
            if row.get("HS") == "1":
                rows.append(row)

    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8") as tmp:
        w = csv.DictWriter(tmp, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
        hate_csv = tmp.name

    out_csv = "/tmp/em_hsd_qwen_hate.csv"
    env = {**dict(__import__("os").environ), "CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": str(ROOT / "src")}
    cmd = [
        sys.executable, "-m", "em_hsd_cli.run",
        "--in", hate_csv,
        "--out", out_csv,
        "--config", config,
        "--debug-seed", "QWEN",
    ]
    print("Running:", " ".join(cmd), flush=True)
    rc = subprocess.call(cmd, cwd=str(ROOT), env=env)
    if rc != 0:
        return rc

    harness = [
        sys.executable, "-m", "harness.evaluate",
        "--original", hate_csv,
        "--privatized", out_csv,
        "--config", str(SPINE / "configs" / "test.yaml"),
        "--utility-backend", "proxy",
    ]
    env["PYTHONPATH"] = str(SPINE / "src")
    print("\nHarness:", " ".join(harness), flush=True)
    return subprocess.call(harness, cwd=str(SPINE), env=env)


if __name__ == "__main__":
    raise SystemExit(main())
