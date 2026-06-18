"""Demo of the TRIAGE-DP pipeline entry point connecting Layers 1–4.

Usage:
    PYTHONPATH=src python scripts/pipeline_demo.py \
        --text "you are a complete dummy" \
        --config configs/em-hsd-v2-test.yaml --json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path


def _bootstrap_spine() -> None:
    """Put the SPINE src dir on sys.path before importing mechanism/triage_dp."""
    candidates = []
    env = os.environ.get("EM_HSD_SPINE_PATH", "").strip()
    if env:
        candidates.append(Path(env).expanduser())
    repo_root = Path(__file__).resolve().parents[1]  # 123/
    candidates.append(repo_root.parent / "Johnny t0-1.03" / "src")
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.exists() and str(resolved) not in sys.path:
            sys.path.insert(0, str(resolved))
            return


def run_demo(text: str, config_path: str, *, json_output: bool) -> int:
    from mechanism.rng import make_row_rng

    from triage_dp import TriageDpPipeline

    pipe = TriageDpPipeline.from_config(os.path.abspath(config_path))
    pipe.config.spine.rng = make_row_rng(0, run_seed="demo")
    out, audit = pipe.sanitize(text)

    if json_output:
        print(json.dumps(audit, indent=2, ensure_ascii=False, default=str))
        return 0

    print(f"original : {text}")
    print(f"x_priv   : {audit['x_priv']}")
    print(f"selected : {out}")
    print(f"fallback : {audit['fallback']} ({audit.get('fallback_reason', '')})")
    print(f"k_valid  : {audit['k_valid']}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Demo the TRIAGE-DP pipeline (Layers 1–4)")
    p.add_argument("--text", required=True)
    p.add_argument("--config", default="configs/em-hsd-v2-test.yaml")
    p.add_argument("--json", action="store_true", dest="json_output")
    args = p.parse_args(argv)
    _bootstrap_spine()
    return run_demo(args.text, args.config, json_output=args.json_output)


if __name__ == "__main__":
    raise SystemExit(main())
