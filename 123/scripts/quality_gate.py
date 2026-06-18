#!/usr/bin/env python3
"""Pre-commit quality gate: ruff + mypy + pytest.

Exits non-zero if any step fails. Capture stdout for CI evidence.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run(cmd: list[str], label: str) -> int:
    print(f"\n{'=' * 60}")
    print(f"STEP: {label}")
    print(f"COMMAND: {' '.join(cmd)}")
    print("=" * 60)
    result = subprocess.run(cmd, cwd=ROOT)
    status = "PASS" if result.returncode == 0 else "FAIL"
    print(f"[{status}] {label} (exit {result.returncode})")
    return result.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="em-hsd-v2 quality gate")
    parser.add_argument("--skip-ruff", action="store_true")
    parser.add_argument("--skip-mypy", action="store_true")
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args()

    overall = 0

    if not args.skip_ruff:
        overall |= run([sys.executable, "-m", "ruff", "check", "src", "scripts", "tests"], "ruff check")

    if not args.skip_mypy:
        overall |= run([sys.executable, "-m", "mypy", "src/em_hsd"], "mypy type check")

    if not args.skip_tests:
        overall |= run([sys.executable, "-m", "pytest", "tests/", "-q"], "pytest suite")

    print(f"\n{'=' * 60}")
    if overall == 0:
        print("QUALITY GATE: PASS")
    else:
        print("QUALITY GATE: FAIL")
    print("=" * 60)
    return overall


if __name__ == "__main__":
    sys.exit(main())

