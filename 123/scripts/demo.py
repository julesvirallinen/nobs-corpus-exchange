"""Interactive demo of EM-HSD v2 on a single sentence."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from mechanism.rng import make_row_rng

from em_hsd import load_em_hsd_config
from em_hsd.layer4.orchestrator import Layer4Orchestrator


def _format_candidates(audit: dict) -> str:
    parts: list[str] = []
    for c in audit.get("candidates", []):
        mark = "*" if c.get("selected") else " "
        parts.append(
            f"  [{mark}] {c.get('text', '')[:120]!r}  "
            f"score={c.get('score', 0):.3f}"
        )
    return "\n".join(parts) if parts else "  (none)"


def run_demo(text: str, config_path: str, *, show_candidates: bool, json_output: bool) -> int:
    config = load_em_hsd_config(config_path)
    config.spine.rng = make_row_rng(0, run_seed="demo")
    out, audit = Layer4Orchestrator().privatize(text, config)

    if json_output:
        print(json.dumps(audit, indent=2, ensure_ascii=False, default=str))
        return 0

    print(f"original : {text}")
    print(f"x_priv   : {audit['x_priv']}")
    print(f"selected : {out}")
    print(f"fallback : {audit['fallback']} ({audit.get('fallback_reason', '')})")
    print(f"k_valid  : {audit['k_valid']}")
    print(f"utility  : {audit.get('utility_backend', '?')}")
    if show_candidates:
        print("candidates:")
        print(_format_candidates(audit))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Demo EM-HSD v2 on a single sentence")
    p.add_argument("--text", required=True)
    p.add_argument("--config", default="configs/em-hsd-v2-test.yaml")
    p.add_argument("--show-candidates", action="store_true")
    p.add_argument("--json", action="store_true", dest="json_output")
    args = p.parse_args(argv)
    return run_demo(args.text, args.config, show_candidates=args.show_candidates, json_output=args.json_output)


if __name__ == "__main__":
    raise SystemExit(main())
