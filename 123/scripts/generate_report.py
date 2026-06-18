"""Generate a Markdown evaluation report from ablation/calibration JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


def _load_json_or_yaml(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    text = Path(path).read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return yaml.safe_load(text)


def _row(code: str, desc: str, result: dict[str, Any]) -> str:
    return (
        f"| {code} | {desc} | {result['rows']} | "
        f"{result['changed_rate']:.2%} | {result['fallback_rate']:.2%} | "
        f"{result['avg_k_valid']:.2f} |\n"
    )


def _burrows_delta_stub() -> str:
    return (
        "## Burrows' Delta diagnostic stub\n\n"
        "*Requires character n-gram profiles across the full corpus. "
        "Placeholder: re-identification risk is not computed in mock mode.*\n\n"
    )


def _pareto_stub() -> str:
    return (
        "## Pareto frontier placeholder\n\n"
        "| Utility ratio | Privacy ratio | TO | Config hint |\n"
        "|---|---|---|---|\n"
        "| *mock* | *mock* | *mock* | run calibration with real harness |\n\n"
    )


def generate(ablations_path: str, calibration_path: str | None, output: str) -> int:
    abl = _load_json_or_yaml(ablations_path)
    cal = _load_json_or_yaml(calibration_path) if calibration_path else None

    lines: list[str] = []
    lines.append("# EM-HSD 2.0 Evaluation Report\n")
    lines.append(f"**Config:** `{abl.get('config')}`  \n")
    lines.append(f"**Input:** `{abl.get('input')}`  \n\n")

    baseline = abl.get("baseline", {})
    lines.append("## Baseline\n\n")
    lines.append(
        f"- Rows: {baseline.get('rows', 0)}\n"
        f"- Changed: {baseline.get('changed', 0)} ({baseline.get('changed_rate', 0):.2%})\n"
        f"- Fallback: {baseline.get('fallback', 0)} ({baseline.get('fallback_rate', 0):.2%})\n"
        f"- Avg valid candidates: {baseline.get('avg_k_valid', 0):.2f}\n\n"
    )

    lines.append("## Ablations\n\n")
    lines.append("| Code | Description | Rows | Changed | Fallback | Avg k_valid |\n")
    lines.append("|---|---|---:|---:|---:|---:|\n")
    for entry in abl.get("ablations", []):
        lines.append(_row(entry["code"], entry["description"], entry["result"]))
    lines.append("\n")

    if cal:
        lines.append("## Calibration\n\n")
        best = cal.get("summary", {}).get("best_theta", {})
        metrics = cal.get("summary", {}).get("best_metrics", {})
        for k, v in best.items():
            lines.append(f"- {k}: {v:.4f}\n")
        lines.append(f"- Best TO (mock): {metrics.get('TO', 0):.4f}\n")
        lines.append(f"- U ratio: {metrics.get('U_ratio', 0):.4f}\n")
        lines.append(f"- P ratio: {metrics.get('P_ratio', 0):.4f}\n")
        lines.append(f"- Fallback rate: {metrics.get('fallback_rate', 0):.2%}\n\n")
    else:
        lines.append("## Calibration\n\n*No calibration JSON provided.*\n\n")

    lines.append(_burrows_delta_stub())
    lines.append(_pareto_stub())

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(lines), encoding="utf-8")
    print(f"Wrote report to {out_path}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Generate evaluation report from JSON artifacts")
    p.add_argument("--ablations", required=True)
    p.add_argument("--calibration")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    return generate(args.ablations, args.calibration, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
