#!/usr/bin/env python3
"""Smoke-test Qwen proposer: load model + one paraphrase + optional CSV subset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from em_hsd import load_em_hsd_config, privatize_em_hsd_v2
from em_hsd.config import resolve_config_path
from em_hsd.generative_proposer import get_proposer
from em_hsd.resources import init_spine_resources, protected_canonicals
from em_hsd.token_sanitize import token_sanitize
from mechanism.rng import make_row_rng


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/em-hsd-v2-qwen.yaml")
    p.add_argument(
        "--text",
        default="Stop being such a dummy and read the instructions before you ask.",
    )
    p.add_argument("--k", type=int, default=3)
    args = p.parse_args()

    cfg = load_em_hsd_config(resolve_config_path(args.config))
    cfg.spine.rng = make_row_rng(0, run_seed="qwen-smoke")
    init_spine_resources(cfg)

    if cfg.generation.backend == "llama_cpp":
        print(
            "Note: start llama-server first: bash scripts/start_llama_server.sh",
            flush=True,
        )

    print(f"Loading proposer ({cfg.generation.backend}: {cfg.generation.model})...")
    proposer = get_proposer(cfg)
    x_priv, _ = token_sanitize(args.text, cfg, cfg.em_hsd_v2.epsilon_1)
    canonicals, _ = protected_canonicals(args.text, cfg)
    proposer.bind(cfg.spine.rng, canonicals)

    print(f"x_priv: {x_priv[:120]}...")
    print(f"protected: {canonicals}")
    candidates = proposer.propose(x_priv, args.k)
    print(f"\nGenerated {len(candidates)} candidates:")
    for i, c in enumerate(candidates):
        print(f"  [{i}] {c[:200]}")

    out, audit = privatize_em_hsd_v2(args.text, cfg)
    print(f"\nPipeline output: {out[:200]}")
    print(f"fallback={audit['fallback']} k_valid={audit['k_valid']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
