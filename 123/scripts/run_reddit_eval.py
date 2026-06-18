#!/usr/bin/env python3
"""Privatize Reddit CSV and run SPINE harness.evaluate for TO / utility / privacy."""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPINE = ROOT.parent / "spine"
DEFAULT_IN = ROOT.parent / "data" / "reddit_25_TKDeVb4.csv"
DEFAULT_OUT = ROOT.parent / "data" / "reddit_25_emhsd_v2.csv"
DEFAULT_CONFIG = ROOT / "configs" / "em-hsd-v2-qwen-reddit.yaml"
HARNESS_CONFIG = SPINE / "configs" / "test.yaml"


def main() -> int:
    p = argparse.ArgumentParser(description="Run EM-HSD v2 on Reddit + harness eval")
    p.add_argument("--in", dest="in_path", default=str(DEFAULT_IN))
    p.add_argument("--out", dest="out_path", default=str(DEFAULT_OUT))
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--utility-backend", default="hf", choices=["proxy", "hf"])
    p.add_argument("--skip-privatize", action="store_true")
    p.add_argument("--skip-eval", action="store_true")
    p.add_argument("--no-resume", action="store_true")
    args = p.parse_args()

    env = dict(__import__("os").environ)
    env["PYTHONPATH"] = str(ROOT / "src")

    if not args.skip_privatize:
        cmd = [
            sys.executable, "-m", "em_hsd_cli.run",
            "--in", args.in_path,
            "--out", args.out_path,
            "--config", args.config,
            "--utility-backend", args.utility_backend,
        ]
        if args.no_resume:
            cmd.append("--no-resume")
        print("Privatizing:", " ".join(cmd), flush=True)
        rc = subprocess.call(cmd, cwd=str(ROOT), env=env)
        if rc != 0:
            return rc

    if args.skip_eval:
        return 0

    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(SPINE / "src"))
    from em_hsd.csv_compat import (
        read_csv_compat,
        write_canonical_csv,
        write_canonical_privatized_csv,
    )

    _, orig_rows, _ = read_csv_compat(args.in_path)
    _, priv_rows, _ = read_csv_compat(args.out_path)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        canon_orig = tmp / "original.csv"
        canon_priv = tmp / "privatized.csv"
        write_canonical_csv(str(canon_orig), orig_rows)
        write_canonical_privatized_csv(str(canon_priv), priv_rows)

        harness = [
            sys.executable, "-m", "harness.evaluate",
            "--original", str(canon_orig),
            "--privatized", str(canon_priv),
            "--config", str(HARNESS_CONFIG),
            "--utility-backend", args.utility_backend,
        ]
        env["PYTHONPATH"] = str(SPINE / "src")
        print("\nHarness:", " ".join(harness), flush=True)
        return subprocess.call(harness, cwd=str(SPINE), env=env)


if __name__ == "__main__":
    raise SystemExit(main())
