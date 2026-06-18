from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_SIBLING_NAME = "spine"
_ENV_VAR = "EM_HSD_SPINE_PATH"


def _repo_root() -> Path:
    # this file: <root>/src/em_hsd_serve/__init__.py → parents[2] is the repo root.
    return Path(__file__).resolve().parents[2]


def _ensure_spine_on_path() -> str | None:
    candidates: list[Path] = []
    env_path = os.environ.get(_ENV_VAR, "").strip()
    if env_path:
        candidates.append(Path(env_path).expanduser())
    candidates.append(_repo_root().parent / _SIBLING_NAME / "src")
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.exists():
            path_str = str(resolved)
            if path_str not in sys.path:
                sys.path.insert(0, path_str)
            return path_str
    return None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="em-hsd-serve", description="Serve the EM-HSD demo API")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--reload", action="store_true", help="auto-reload on code changes")
    args = p.parse_args(argv)

    spine = _ensure_spine_on_path()
    if spine is None:
        print(
            "WARNING: could not locate SPINE src dir; relying on PYTHONPATH for "
            "'mechanism'. Set EM_HSD_SPINE_PATH if import fails.",
            file=sys.stderr,
        )

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
