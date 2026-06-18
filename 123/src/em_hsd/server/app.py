"""FastAPI server exposing the EM-HSD v2 Layer-4 pipeline to a demo frontend.

Run via ``python scripts/serve.py`` (recommended) or ``em-hsd-serve``. Both
ensure the SPINE source dir is importable before this module loads. If you
import this module directly, make sure ``mechanism`` is already on the path
(e.g. ``PYTHONPATH`` includes ``Johnny t0-1.03/src``).

Endpoints:
    GET  /                – static demo frontend (index.html)
    GET  /health          – liveness + loaded backend info
    GET  /api/configs     – list selectable config files
    POST /api/privatize   – privatise one text string, return audit
"""

from __future__ import annotations

import threading
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from mechanism.rng import make_row_rng
from pydantic import BaseModel, Field

from em_hsd import load_em_hsd_config
from em_hsd.core.config import EmHsdConfig
from em_hsd.layer4.orchestrator import Layer4Orchestrator

# Default demo config: mock generation + proxy utility scorer — fast and
# requires no model downloads. Override per request via the `config` field.
DEFAULT_CONFIG = "em-hsd-v2-test.yaml"

_STATIC_DIR = Path(__file__).resolve().parent / "static"
# app.py lives at <root>/src/em_hsd/server/app.py → parents[3] is the repo root.
_CONFIGS_DIR = Path(__file__).resolve().parents[3] / "configs"

# The orchestrator mutates shared state (config.spine.rng, proposer binding)
# per call and is not thread-safe; serialise privatize calls. The module-level
# orchestrator keeps the scorer/encoder/proposer warm across requests.
_orchestrator = Layer4Orchestrator()
_lock = threading.Lock()


class PrivatizeRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Input sentence to privatise.")
    config: str | None = Field(
        default=None,
        description=f"Config file name from /api/configs. Defaults to {DEFAULT_CONFIG}.",
    )
    seed: int = Field(default=0, description="Row seed for deterministic DP selection.")
    run_seed: str = Field(default="demo", description="Run-level seed label.")


class Candidate(BaseModel):
    text: str
    score: float
    selected: bool


class PrivatizeResponse(BaseModel):
    original: str
    x_priv: str
    selected: str
    fallback: bool
    fallback_reason: str
    config: str
    k_generated: int
    k_valid: int
    p_hate_original: float | None = None
    p_hate_x_priv: float | None = None
    epsilon_total: float | None = None
    epsilon_1: float | None = None
    epsilon_2: float | None = None
    utility_backend: str | None = None
    generation_backend: str | None = None
    candidates: list[Candidate]
    audit: dict[str, Any]


def list_configs() -> list[str]:
    if not _CONFIGS_DIR.is_dir():
        return [DEFAULT_CONFIG]
    return sorted(p.name for p in _CONFIGS_DIR.glob("*.yaml"))


def _resolve_config_name(name: str | None) -> Path:
    """Map a config name to an absolute path inside the configs dir.

    Only basenames within the configs dir are accepted (no traversal).
    """
    chosen = name or DEFAULT_CONFIG
    if Path(chosen).name != chosen:
        raise HTTPException(status_code=400, detail="config must be a bare file name")
    path = _CONFIGS_DIR / chosen
    if not path.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"unknown config {chosen!r}; see GET /api/configs",
        )
    return path


@lru_cache(maxsize=16)
def _load_config_cached(config_path: str) -> EmHsdConfig:
    """Load and cache a config by absolute path.

    A stable config object id keeps the orchestrator's resource cache warm
    (it is keyed by ``id(config)``).
    """
    return load_em_hsd_config(config_path)


def _privatize(text: str, config_path: str, seed: int, run_seed: str) -> tuple[str, dict]:
    config = _load_config_cached(config_path)
    with _lock:
        config.spine.rng = make_row_rng(seed, run_seed=run_seed)
        return _orchestrator.privatize(text, config)


def create_app() -> FastAPI:
    app = FastAPI(
        title="EM-HSD v2 — Layer-4 DP Text Sanitisation",
        description="Demo API wrapping the differential-privacy paraphrase pipeline.",
        version="0.1.0",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "default_config": DEFAULT_CONFIG,
            "configs_available": len(list_configs()),
        }

    @app.get("/api/configs")
    def get_configs() -> dict[str, Any]:
        return {"default": DEFAULT_CONFIG, "configs": list_configs()}

    @app.post("/api/privatize", response_model=PrivatizeResponse)
    def privatize(req: PrivatizeRequest) -> PrivatizeResponse:
        config_path = _resolve_config_name(req.config)
        try:
            selected, audit = _privatize(
                req.text, str(config_path), req.seed, req.run_seed
            )
        except HTTPException:
            raise
        except Exception as exc:  # surface pipeline errors as 500 with detail
            raise HTTPException(status_code=500, detail=f"pipeline error: {exc}") from exc

        return PrivatizeResponse(
            original=req.text,
            x_priv=audit.get("x_priv", ""),
            selected=selected,
            fallback=bool(audit.get("fallback", False)),
            fallback_reason=str(audit.get("fallback_reason", "")),
            config=config_path.name,
            k_generated=int(audit.get("k_generated", 0)),
            k_valid=int(audit.get("k_valid", 0)),
            p_hate_original=audit.get("P_hate_original"),
            p_hate_x_priv=audit.get("P_hate_x_priv"),
            epsilon_total=audit.get("epsilon_total"),
            epsilon_1=audit.get("epsilon_1"),
            epsilon_2=audit.get("epsilon_2"),
            utility_backend=audit.get("utility_backend"),
            generation_backend=_load_config_cached(str(config_path)).generation.backend,
            candidates=[
                Candidate(
                    text=c.get("text", ""),
                    score=float(c.get("score", 0.0)),
                    selected=bool(c.get("selected", False)),
                )
                for c in audit.get("candidates", [])
            ],
            audit=audit,
        )

    if _STATIC_DIR.is_dir():
        @app.get("/")
        def index() -> FileResponse:
            return FileResponse(str(_STATIC_DIR / "index.html"))

        app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    return app


app = create_app()


def main(argv: list[str] | None = None) -> int:
    """Console entry point. Prefer ``python scripts/serve.py`` for path setup."""
    import argparse

    import uvicorn

    p = argparse.ArgumentParser(prog="em-hsd-serve", description="Serve the EM-HSD demo API")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--reload", action="store_true")
    args = p.parse_args(argv)

    uvicorn.run(
        "em_hsd.server.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
