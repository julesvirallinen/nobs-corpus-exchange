#!/usr/bin/env python
"""Launch the EM-HSD v2 demo API server.

This is the recommended entry point: it ensures the SPINE source dir is on
``sys.path`` *before* importing ``em_hsd`` (whose package import transitively
imports ``mechanism`` at load time).

Usage:
    PYTHONPATH=src python scripts/serve.py [--host H] [--port P] [--reload]

Then open http://127.0.0.1:8000/ for the demo frontend.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _bootstrap_spine() -> str | None:
    """Load the standalone spine bootstrap by file path (no em_hsd import)."""
    bootstrap_file = _SRC / "em_hsd" / "server" / "_bootstrap.py"
    spec = importlib.util.spec_from_file_location("_em_hsd_serve_bootstrap", bootstrap_file)
    if spec is None or spec.loader is None:  # pragma: no cover
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ensure_spine_on_path()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Serve the EM-HSD v2 demo API")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--reload", action="store_true", help="auto-reload on code changes")
    p.add_argument(
        "--config",
        dest="config",
        default=None,
        help="default config name (real models only; defaults to em-hsd-v2-qwen.yaml)",
    )
    args = p.parse_args(argv)

    import os

    # Mocks have been removed — every config uses real models, so always permit
    # loading them (from the HF cache; set HF_HUB_OFFLINE=1 to forbid network).
    os.environ.setdefault("EM_HSD_ALLOW_DOWNLOADS", "1")
    if args.config:
        os.environ["EM_HSD_DEFAULT_CONFIG"] = args.config

    spine = _bootstrap_spine()
    if spine is None:
        print(
            "WARNING: could not locate SPINE src dir; relying on PYTHONPATH for "
            "'mechanism'. Set EM_HSD_SPINE_PATH if import fails.",
            file=sys.stderr,
        )
    else:
        print(f"SPINE: {spine}", file=sys.stderr)

    import uvicorn  # imported after path setup

    print(f"Serving EM-HSD demo at http://{args.host}:{args.port}/", file=sys.stderr)
    uvicorn.run(
        "em_hsd.server.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
