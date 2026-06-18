from __future__ import annotations

import argparse
import copy
import random
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml
from mechanism.rng import make_row_rng

from em_hsd import load_em_hsd_config
from em_hsd.core.config import EmHsdConfig
from em_hsd.layer4.orchestrator import Layer4Orchestrator
from em_hsd.layer4.scorer import HFToxicityScorer

_SEARCH_SPACE = {
    "epsilon_total": (3.0, 36.0),
    "hate_floor_delta": (0.0, 0.5),
    "tau_sem_min": (0.1, 0.9),
}


class LocalTradeoffProxy:
    """Local trade-off proxy: a stylometry-overlap privacy estimate combined
    with the REAL hate scorer (unitary/unbiased-toxic-roberta) for utility.

    Not the full SPINE harness (privacy is an n-gram-overlap estimate), but the
    hate/utility axis is the real classifier — no mock scorer.
    """

    def __init__(self, rng: random.Random):
        self.rng = rng

    def _ngram_overlap(self, a: str, b: str, n: int = 3) -> float:
        a_chars = [c for c in (a or "").lower() if c.isalnum() or c.isspace()]
        b_chars = [c for c in (b or "").lower() if c.isalnum() or c.isspace()]
        a_ng = {tuple(a_chars[i : i + n]) for i in range(len(a_chars) - n + 1)}
        b_ng = {tuple(b_chars[i : i + n]) for i in range(len(b_chars) - n + 1)}
        if not a_ng:
            return 0.0
        return len(a_ng & b_ng) / len(a_ng)

    def evaluate(self, original: str, privatized: str, author_id: str, scorer: HFToxicityScorer) -> dict[str, float]:
        utility_original = float(scorer.score(original))
        utility_priv = float(scorer.score(privatized))
        privacy_original = 1.0
        privacy_priv = 1.0 - self._ngram_overlap(original, privatized)
        u_ratio = utility_priv / max(utility_original, 1e-9)
        p_ratio = privacy_priv / max(privacy_original, 1e-9)
        return {
            "TO": u_ratio - p_ratio,
            "U_ratio": u_ratio,
            "P_ratio": p_ratio,
            "utility_original": utility_original,
            "utility_privatized": utility_priv,
            "privacy_original": privacy_original,
            "privacy_privatized": privacy_priv,
        }


def _make_scorer(config: EmHsdConfig) -> HFToxicityScorer:
    return HFToxicityScorer(config.utility.model, score_label=config.utility.score_label)


def _evaluate_dev(
    config: EmHsdConfig,
    dev_rows: Sequence[tuple[str, str]],
    rng: random.Random,
) -> dict[str, float]:
    proxy = LocalTradeoffProxy(rng)
    scorer = _make_scorer(config)
    orch = Layer4Orchestrator()
    tos: list[float] = []
    uratios: list[float] = []
    pratios: list[float] = []
    fallback_count = 0
    for idx, (text, author) in enumerate(dev_rows):
        config.spine.rng = make_row_rng(idx, run_seed="calibrate")
        privatized, audit = orch.privatize(text, config)
        if audit.get("fallback"):
            fallback_count += 1
        metrics = proxy.evaluate(text, privatized, author, scorer)
        tos.append(metrics["TO"])
        uratios.append(metrics["U_ratio"])
        pratios.append(metrics["P_ratio"])
    if not tos:
        return {"TO": -1.0, "U_ratio": 0.0, "P_ratio": 0.0, "fallback_rate": 0.0}
    return {
        "TO": sum(tos) / len(tos),
        "U_ratio": sum(uratios) / len(uratios),
        "P_ratio": sum(pratios) / len(pratios),
        "fallback_rate": fallback_count / len(dev_rows),
    }


def _sample_theta(rng: random.Random) -> dict[str, float]:
    return {
        key: rng.uniform(low, high)
        for key, (low, high) in _SEARCH_SPACE.items()
    }


def _apply_theta(config: EmHsdConfig, theta: dict[str, float]) -> EmHsdConfig:
    cfg = copy.deepcopy(config)
    em = cfg.em_hsd_v2
    if "epsilon_total" in theta:
        em.epsilon_total = theta["epsilon_total"]
    if "hate_floor_delta" in theta:
        em.hate_floor_delta = theta["hate_floor_delta"]
    if "tau_sem_min" in theta:
        em.tau_sem_min = theta["tau_sem_min"]
    return cfg


class CalibrateRunner:
    def __init__(
        self,
        config: EmHsdConfig,
        dev_rows: Sequence[tuple[str, str]],
        *,
        trials: int = 20,
        seed: int = 0,
    ):
        self.base_config = config
        self.dev_rows = list(dev_rows)
        self.trials = trials
        self.rng = random.Random(seed)

    def run(self) -> tuple[dict[str, Any], EmHsdConfig]:
        best_theta: dict[str, float] = {}
        best_score = float("-inf")
        best_metrics: dict[str, float] = {}
        history: list[dict[str, Any]] = []
        for trial in range(self.trials):
            theta = _sample_theta(self.rng)
            cfg = _apply_theta(self.base_config, theta)
            metrics = _evaluate_dev(cfg, self.dev_rows, self.rng)
            score = metrics["TO"]
            entry = {"trial": trial, "theta": theta, "metrics": metrics}
            history.append(entry)
            if score > best_score:
                best_score = score
                best_theta = theta
                best_metrics = metrics
        best_cfg = _apply_theta(self.base_config, best_theta)
        summary = {
            "best_theta": best_theta,
            "best_metrics": best_metrics,
            "history": history,
            "search_space": _SEARCH_SPACE,
            "note": "mock TO proxy — not the real harness",
        }
        return summary, best_cfg


def _load_dev_csv(path: str) -> list[tuple[str, str]]:
    from em_hsd.csv_compat import read_csv_compat
    _, rows, _ = read_csv_compat(path)
    return [(r["Text"], r.get("Author", str(i))) for i, r in enumerate(rows)]


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Calibrate EM-HSD v2 hyperparameters (mock TO).")
    p.add_argument("--dev", required=True, help="dev CSV with Text and Author columns")
    p.add_argument("--config", required=True, help="base YAML config")
    p.add_argument("--trials", type=int, default=5, help="number of random-search trials")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", required=True, help="output YAML path")
    args = p.parse_args(argv)

    base = load_em_hsd_config(args.config)
    dev_rows = _load_dev_csv(args.dev)
    runner = CalibrateRunner(base, dev_rows, trials=args.trials, seed=args.seed)
    summary, best_cfg = runner.run()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary": summary,
        "config": {
            "em_hsd_v2": asdict(best_cfg.em_hsd_v2),
            "generation": asdict(best_cfg.generation),
            "embedding": asdict(best_cfg.embedding),
            "utility": asdict(best_cfg.utility),
            "triage_dp": asdict(best_cfg.triage_dp),
        },
    }
    with out_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(payload, fh, sort_keys=False)

    print(f"Best TO={summary['best_metrics']['TO']:.4f} with theta={summary['best_theta']}")
    print(f"Calibration YAML written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
