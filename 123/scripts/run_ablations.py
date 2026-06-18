"""Run ablations A1-A9 on the EM-HSD v2 mock pipeline."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

from mechanism.rng import make_row_rng

from em_hsd import load_em_hsd_config
from em_hsd.core.config import EmHsdConfig
from em_hsd.layer4.orchestrator import Layer4Orchestrator


def _load_rows(path: str) -> list[tuple[str, str]]:
    from em_hsd.csv_compat import read_csv_compat
    _, rows, _ = read_csv_compat(path)
    return [(r["Text"], r.get("Author", str(i))) for i, r in enumerate(rows)]


def _run_single(cfg: EmHsdConfig, rows: Sequence[tuple[str, str]]) -> dict[str, Any]:
    orch = Layer4Orchestrator()
    changed = 0
    fallback = 0
    k_valids: list[int] = []
    for idx, (text, _author) in enumerate(rows):
        cfg.spine.rng = make_row_rng(idx, run_seed="ablations")
        out, audit = orch.privatize(text, cfg)
        if out != text:
            changed += 1
        if audit.get("fallback"):
            fallback += 1
        k_valids.append(audit.get("k_valid", 0))
    n = len(rows)
    return {
        "rows": n,
        "changed": changed,
        "changed_rate": changed / n if n else 0.0,
        "fallback": fallback,
        "fallback_rate": fallback / n if n else 0.0,
        "avg_k_valid": sum(k_valids) / len(k_valids) if k_valids else 0.0,
    }


def _ablation_a1(cfg: EmHsdConfig) -> EmHsdConfig:
    """No epsilon_1 (paraphrase raw x)."""
    c = deepcopy(cfg)
    c.em_hsd_v2.epsilon_split = 0.0
    return c


def _ablation_a2(cfg: EmHsdConfig) -> EmHsdConfig:
    """No protected spans (empty lexicon)."""
    c = deepcopy(cfg)
    c.spine.lexicon.test_terms = []
    return c


def _ablation_a3(cfg: EmHsdConfig) -> EmHsdConfig:
    """No hate floor delta."""
    c = deepcopy(cfg)
    c.em_hsd_v2.hate_floor_delta = 0.0
    return c


def _ablation_a4(cfg: EmHsdConfig) -> EmHsdConfig:
    """No EM (argmax P_hate)."""
    c = deepcopy(cfg)
    c.em_hsd_v2.epsilon_total = 1e-9
    return c


def _ablation_a5(cfg: EmHsdConfig) -> EmHsdConfig:
    """EM-HSD-Naive (delta_u = 1)."""
    c = deepcopy(cfg)
    c.em_hsd_v2.use_refined_delta_u = False
    return c


def _ablation_a6(cfg: EmHsdConfig) -> EmHsdConfig:
    """Semantic-only EM (hate floor very low, high epsilon)."""
    c = deepcopy(cfg)
    c.em_hsd_v2.hate_floor_delta = 0.0
    c.em_hsd_v2.tau_sem_min = 0.9
    return c


def _ablation_a7(cfg: EmHsdConfig) -> EmHsdConfig:
    """No tau_dup prune."""
    c = deepcopy(cfg)
    c.em_hsd_v2.tau_dup = 1.0
    return c


def _ablation_a8(cfg: EmHsdConfig) -> EmHsdConfig:
    """No tau_sem_min."""
    c = deepcopy(cfg)
    c.em_hsd_v2.tau_sem_min = 0.0
    return c


def _ablation_a9(cfg: EmHsdConfig) -> EmHsdConfig:
    """Hand prompt vs optimized prompt (placeholder)."""
    c = deepcopy(cfg)
    c.generation.backend = "mock"
    return c


_ABLATIONS = [
    ("A1", "No epsilon_1 (paraphrase raw x)", _ablation_a1),
    ("A2", "No protected spans", _ablation_a2),
    ("A3", "No hate floor delta", _ablation_a3),
    ("A4", "No EM (argmax P_hate)", _ablation_a4),
    ("A5", "EM-HSD-Naive (delta_u=1)", _ablation_a5),
    ("A6", "Semantic-only EM", _ablation_a6),
    ("A7", "No tau_dup prune", _ablation_a7),
    ("A8", "No tau_sem_min", _ablation_a8),
    ("A9", "Hand vs optimized prompt (placeholder)", _ablation_a9),
]


def run_ablations(config_path: str, in_path: str, output: str) -> int:
    cfg = load_em_hsd_config(config_path)
    rows = _load_rows(in_path)
    baseline = _run_single(cfg, rows)
    ablations: list[dict[str, Any]] = []
    for code, description, mutator in _ABLATIONS:
        mutated = mutator(cfg)
        result = _run_single(mutated, rows)
        ablations.append({
            "code": code,
            "description": description,
            "result": result,
        })
    report = {
        "baseline": baseline,
        "ablations": ablations,
        "config": config_path,
        "input": in_path,
    }
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    print(f"Wrote ablation report to {out_path}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Run EM-HSD v2 ablations A1-A9")
    p.add_argument("--config", required=True)
    p.add_argument("--in", dest="in_path", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    return run_ablations(args.config, args.in_path, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
