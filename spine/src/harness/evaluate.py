from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Dict, List

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass

import yaml

from .classifiers import build_classifiers
from .reident import reident_report
from .tradeoff import trade_off
from .utility import utility_report

csv.field_size_limit(10 * 1024 * 1024)


def _read(path: str, required: List[str]) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames or any(c not in reader.fieldnames for c in required):
            raise ValueError(
                f"{path!r}: missing required column(s) {required}; "
                f"found {reader.fieldnames}"
            )
        return list(reader)


def _load_terms(config_path: str) -> List[str]:    if not os.path.exists(config_path):
        return []
    with open(config_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    lex = cfg.get("lexicon", {}) or {}
    if lex.get("source") == "test":
        return list(lex.get("test_terms", []) or [])
    path = lex.get("path")
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fh:
            return [ln.strip() for ln in fh
                    if ln.strip() and not ln.startswith("#")]
    return []


def evaluate(original_path: str, privatized_path: str, utility_backend: str,
             config_path: str, use_transformer: bool,
             transformer_model: str) -> Dict:
    original = _read(original_path, ["ID", "Author", "Text", "HS"])
    privatized = _read(privatized_path, ["ID", "Text"])

    if len(original) != len(privatized):
        raise ValueError(
            f"row count mismatch: original={len(original)} "
            f"privatized={len(privatized)}"
        )
    mismatched = [i for i, (o, p) in enumerate(zip(original, privatized))
                  if o["ID"] != p["ID"]]
    if mismatched:
        raise ValueError(f"ID/order mismatch at rows {mismatched[:5]} ...")

    original_texts = [r["Text"] for r in original]
    privatized_texts = [r["Text"] for r in privatized]
    gold_hs = [r["HS"] for r in original]
    authors = [r["Author"] for r in original]   # <-- Author used ONLY here

    terms = _load_terms(config_path)
    classifiers = build_classifiers(utility_backend, terms=terms)

    util = utility_report(classifiers, original_texts, privatized_texts, gold_hs)
    reid = reident_report(original_texts, privatized_texts, authors,
                          use_transformer=use_transformer,
                          transformer_model=transformer_model)
    to = trade_off(util["utility_original"], util["utility_privatized"],
                   reid["privacy_original"], reid["privacy_privatized"])
    return {
        "original": original_path,
        "privatized": privatized_path,
        "utility_backend": utility_backend,
        "n_rows": len(original),
        "utility": util,
        "reidentification": reid,
        "trade_off": to,
    }


def _print_report(rep: Dict) -> None:
    u, r, to = rep["utility"], rep["reidentification"], rep["trade_off"]
    print("=" * 70)
    print(f"SPINE offline measurement  ({rep['n_rows']} rows, "
          f"utility backend = {rep['utility_backend']})")
    print("=" * 70)
    print("\nUTILITY (HS classification; higher = better):")
    for pc in u["per_classifier"]:
        print(f"  {pc['classifier']:<42} "
              f"F1 orig={pc['f1_original']:.3f}  priv={pc['f1_privatized']:.3f}  "
              f"agree={pc['agreement_priv_vs_orig']:.3f}")
    print(f"  ensemble utility:  original={u['utility_original']:.3f}  "
          f"privatized={u['utility_privatized']:.3f}")
    print("\nRE-IDENTIFICATION (authorship top-1; lower priv = better privacy):")
    for pr in r["probes"]:
        if "error" in pr:
            print(f"  {pr['probe']:<24} ERROR: {pr['error']}")
        else:
            print(f"  {pr['probe']:<24} "
                  f"acc orig={pr['accuracy_original']:.3f}  "
                  f"priv={pr['accuracy_privatized']:.3f}")
    print("\nTRADE-OFF ESTIMATE:")
    print(f"  utility_ratio  (util_priv / util_orig) = {to['utility_ratio']:.3f}")
    print(f"  privacy_ratio  (reid_priv / reid_orig) = {to['privacy_ratio']:.3f}")
    print(f"  TO = utility_ratio - privacy_ratio      = "
          f"{to['trade_off_estimate']:.3f}")
    print(f"\n  *** {to['label']} ***")
    print("  This uses our own probe models. Do NOT overfit to it; the "
          "organiser's\n  hidden evaluator differs. See README 'Evaluation honesty'.")
    print("=" * 70)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m harness.evaluate")
    p.add_argument("--original", required=True)
    p.add_argument("--privatized", required=True)
    p.add_argument("--utility-backend", choices=("proxy", "hf"), default="proxy")
    p.add_argument("--config", default="configs/test.yaml",
                   help="config providing proxy-lexicon terms")
    p.add_argument("--reident-transformer", action="store_true",
                   help="add the slower transformer authorship probe (HF)")
    p.add_argument("--transformer-model", default="distilroberta-base")
    p.add_argument("--json", default=None, help="also write the report as JSON")
    args = p.parse_args(argv)

    try:
        rep = evaluate(args.original, args.privatized, args.utility_backend,
                       args.config, args.reident_transformer,
                       args.transformer_model)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    _print_report(rep)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(rep, fh, indent=2, ensure_ascii=False)
        print(f"wrote {args.json}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
