from __future__ import annotations

import argparse
import json
import os
import sys
import time

from mechanism.rng import make_row_rng

from em_hsd import load_em_hsd_config, privatize_em_hsd_v2
from em_hsd import spine_bootstrap as _spine_bootstrap  # noqa: F401 — path setup
from em_hsd.config import resolve_config_path
from em_hsd.core.config import EmHsdConfig
from em_hsd.csv_compat import (
    DiffCheckError,
    append_csv_row_compat,
    assert_preserved_compat,
    read_csv_compat,
)
from em_hsd.io.audit_io import AuditJsonlWriter
from em_hsd.resources import init_spine_resources
from em_hsd.utility_scorer import get_scorer
from triage_dp.pipeline import TriageDpPipeline

_MODE_DESCRIPTION = "Privatise Text column via EM-HSD 2.0 / TRIAGE-DP Layer 4."


def _default_log_path(out_path: str) -> str:
    return out_path + ".log.jsonl"


def _checkpoint_path(out_path: str) -> str:
    return out_path + ".checkpoint.json"


def _load_checkpoint(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _save_checkpoint(path: str, completed: int, n_changed: int) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump({"completed": completed, "n_changed": n_changed}, fh)
    os.replace(tmp, path)


def _resume_state(
    in_path: str,
    out_path: str,
    log_path: str,
    resume: bool,
) -> tuple[int, int]:    ckpt_path = _checkpoint_path(out_path)
    if not resume or not os.path.exists(ckpt_path):
        return 0, 0

    ckpt = _load_checkpoint(ckpt_path)
    completed = int(ckpt.get("completed", 0))
    n_changed = int(ckpt.get("n_changed", 0))
    if completed <= 0:
        return 0, 0

    if not os.path.exists(out_path):
        print(
            f"WARNING: checkpoint says {completed} rows done but {out_path!r} "
            "is missing; starting from row 0.",
            file=sys.stderr,
        )
        return 0, 0

    _, out_rows, _ = read_csv_compat(out_path)
    if len(out_rows) != completed:
        print(
            f"WARNING: checkpoint completed={completed} but output has "
            f"{len(out_rows)} rows; starting from row 0.",
            file=sys.stderr,
        )
        return 0, 0

    try:
        assert_preserved_compat(in_path, out_path, completed_rows=completed)
    except DiffCheckError as exc:
        print(f"WARNING: partial output failed diff check; restarting.\n{exc}", file=sys.stderr)
        return 0, 0

    print(
        f"Resuming from row {completed}/{_total_rows_hint(in_path)} "
        f"(changed so far: {n_changed}); log appends to {log_path}",
        file=sys.stderr,
    )
    return completed, n_changed


def _total_rows_hint(in_path: str) -> int:
    _, rows, _ = read_csv_compat(in_path)
    return len(rows)


def _privatize_row(
    text: str,
    config: EmHsdConfig,
    mode: str,
    adapter: TriageDpPipeline | None,
) -> tuple[str, dict]:
    if mode == "triage-dp" and adapter is not None:
        return adapter.sanitize(text, original_text=text)
    return privatize_em_hsd_v2(text, config)


def run(
    in_path: str,
    out_path: str,
    config_path: str,
    *,
    mode: str = "em-hsd-v2",
    debug_seed: str | None = None,
    log_path: str | None = None,
    resume: bool = True,
    utility_backend: str | None = None,
    utility_model: str | None = None,
) -> int:
    try:
        fieldnames, rows, column_map = read_csv_compat(in_path)
    except FileNotFoundError:
        print(f"ERROR: input file not found: {in_path!r}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    cfg_path = resolve_config_path(config_path)
    config = load_em_hsd_config(cfg_path)

    if utility_backend is not None:
        config.utility.backend = utility_backend
    if utility_model is not None:
        config.utility.model = utility_model

    if debug_seed is not None:
        print(
            "WARNING: --debug-seed is for development ONLY.",
            file=sys.stderr,
        )

    try:
        init_spine_resources(config)
    except ImportError as exc:
        print(f"ERROR: missing optional deps: {exc}", file=sys.stderr)
        return 4
    except Exception as exc:
        print(f"ERROR: could not initialise resources: {exc}", file=sys.stderr)
        return 4

    if config.utility.backend == "hf":
        try:
            get_scorer(config)
        except ImportError as exc:
            print(f"ERROR: hf utility backend requires torch/transformers: {exc}", file=sys.stderr)
            return 4
        except Exception as exc:
            print(f"ERROR: could not load utility model: {exc}", file=sys.stderr)
            return 4

    adapter: TriageDpPipeline | None = None
    if mode == "triage-dp":
        # build_pipeline wires the real Layers 1–3 when triage_dp.enabled,
        # otherwise the NoOp defaults (standalone Layer 4).
        from triage_dp.pipeline import build_pipeline

        adapter = build_pipeline(config)

    log_path = log_path or _default_log_path(out_path)
    ckpt_path = _checkpoint_path(out_path)
    start_idx, n_changed = _resume_state(in_path, out_path, log_path, resume)

    if start_idx == 0:
        if os.path.exists(out_path):
            os.remove(out_path)
        if os.path.exists(log_path):
            os.remove(log_path)
        if os.path.exists(ckpt_path):
            os.remove(ckpt_path)

    log_mode = "a" if start_idx > 0 else "w"
    start = time.time()
    completed = start_idx

    try:
        with AuditJsonlWriter(log_path, append=(log_mode == "a")) as writer:
            for idx in range(start_idx, len(rows)):
                row = rows[idx]
                text = row["Text"]
                config.spine.rng = make_row_rng(idx, run_seed=debug_seed)
                new_text, audit = _privatize_row(text, config, mode, adapter)
                if new_text != text:
                    n_changed += 1
                new_row = dict(row)
                new_row["Text"] = new_text
                append_csv_row_compat(
                    out_path,
                    fieldnames,
                    new_row,
                    column_map,
                    write_header=(idx == 0),
                )
                completed = idx + 1
                _save_checkpoint(ckpt_path, completed, n_changed)
                writer.write({"row": idx, "mode": mode, "audit": audit})
                if completed % 10 == 0 or completed == len(rows):
                    elapsed = time.time() - start
                    rate = ((completed - start_idx) / elapsed) if elapsed > 0 else 0.0
                    print(
                        f"  [{completed}/{len(rows)}] "
                        f"changed={n_changed} ~{rate:.2f} rows/s",
                        file=sys.stderr,
                        flush=True,
                    )
    except KeyboardInterrupt:
        print(
            f"\nInterrupted at row {completed}/{len(rows)}. "
            f"Re-run with --resume to continue.",
            file=sys.stderr,
        )
        return 1
    except KeyError:
        print("ERROR: a row is missing the 'Text' column value.", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"ERROR: could not write output {out_path!r}: {exc}", file=sys.stderr)
        return 2

    try:
        assert_preserved_compat(in_path, out_path)
    except DiffCheckError as exc:
        print(str(exc), file=sys.stderr)
        return 3

    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)

    elapsed = time.time() - start
    processed = len(rows) - start_idx
    rate = (processed / elapsed) if elapsed > 0 else float("inf")
    print(
        f"OK [{mode}] {len(rows)} rows -> {out_path}  "
        f"(changed in {n_changed} rows)  log: {log_path}",
        file=sys.stderr,
    )
    print(f"diff check PASSED. throughput ~{rate:.2f} rows/s.", file=sys.stderr)
    return 0


def build_parser(prog: str = "em-hsd-run") -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog=prog,
        description=_MODE_DESCRIPTION,
    )
    p.add_argument("--in", dest="in_path", required=True)
    p.add_argument("--out", dest="out_path", required=True)
    p.add_argument("--config", dest="config_path", default="em-hsd-v2-test.yaml")
    p.add_argument(
        "--mode",
        dest="mode",
        choices=("em-hsd-v2", "triage-dp"),
        default="em-hsd-v2",
        help="pipeline mode (default: em-hsd-v2)",
    )
    p.add_argument("--debug-seed", dest="debug_seed", default=None)
    p.add_argument("--logs", dest="log_path", default=None)
    p.add_argument(
        "--resume",
        dest="resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="resume from checkpoint if output was interrupted (default: on)",
    )
    p.add_argument(
        "--utility-backend",
        dest="utility_backend",
        choices=("proxy", "hf"),
        default=None,
        help="override config utility.backend (proxy|hf)",
    )
    p.add_argument(
        "--utility-model",
        dest="utility_model",
        default=None,
        help="override config utility.model (HF model id)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run(
        args.in_path,
        args.out_path,
        args.config_path,
        mode=args.mode,
        debug_seed=args.debug_seed,
        log_path=args.log_path,
        resume=args.resume,
        utility_backend=args.utility_backend,
        utility_model=args.utility_model,
    )


def triage_dp_main(argv: list[str] | None = None) -> int:
    args = build_parser(prog="triage-dp-run").parse_args(argv)
    return run(
        args.in_path,
        args.out_path,
        args.config_path,
        mode="triage-dp",
        debug_seed=args.debug_seed,
        log_path=args.log_path,
        resume=args.resume,
        utility_backend=args.utility_backend,
        utility_model=args.utility_model,
    )


if __name__ == "__main__":
    raise SystemExit(main())
