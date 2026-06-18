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
    POST /api/process     – batch: run many rows through the pipeline for the
                            Corpus Exchange upload → review flow
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


# --- Batch processing (Corpus Exchange "Process & validate" flow) -----------

# Demo guardrail: cap rows per request so a large upload can't wedge the
# single-threaded, lock-serialised orchestrator. The UI uploads a sample.
MAX_BATCH_ROWS = 5000
# Decision boundary for the proxy hate score: at/above this a row is flagged.
HATE_THRESHOLD = 0.5


class ProcessRequest(BaseModel):
    rows: list[str] = Field(
        ...,
        min_length=1,
        max_length=MAX_BATCH_ROWS,
        description="Extracted text-column values to privatise & audit, in order.",
    )
    config: str | None = Field(
        default=None,
        description=f"Config file name from /api/configs. Defaults to {DEFAULT_CONFIG}.",
    )
    privatization_level: int = Field(
        default=1,
        ge=0,
        le=3,
        description="0 Light · 1 Standard · 2 High · 3 Maximum. Varies the DP seed.",
    )
    run_seed: str = Field(default="corpus", description="Run-level seed label.")
    classifier: str = Field(
        default="proxy",
        description=(
            "Labelling source: 'proxy' (lexicon-derived, no downloads — default) "
            "or 'hf' (unitary/unbiased-toxic-roberta multi-label, real category + "
            "severity). 'hf' falls back to proxy if the model is unavailable."
        ),
    )


class ProcessedRow(BaseModel):
    id: int
    original: str
    x_priv: str
    p_hate_original: float | None
    p_hate_x_priv: float | None
    flagged: bool
    confidence: float
    confidence_band: str
    severity: str
    category: str | None = None
    tokens_changed: int
    epsilon_total: float | None
    fallback: bool


class ProcessResponse(BaseModel):
    config: str
    privatization_level: int
    classifier: str
    count: int
    flagged: int
    flagged_pct: float
    low_confidence: int
    clusters: int
    severity_hist: dict[str, int]
    category_hist: dict[str, int]
    status_hist: dict[str, int]
    rows: list[ProcessedRow]


def _tokens_changed(original: str, x_priv: str) -> int:
    """Word-level edit count between the original and its anonymised rewrite."""
    a = original.split()
    b = x_priv.split()
    diff = sum(1 for i in range(min(len(a), len(b))) if a[i] != b[i])
    return diff + abs(len(a) - len(b))


def _confidence_band(confidence: float) -> str:
    if confidence >= 0.85:
        return "high"
    if confidence >= 0.70:
        return "medium"
    return "low"


def _derive_row(
    idx: int,
    original: str,
    audit: dict[str, Any],
    clf: dict[str, Any] | None = None,
    selected: str | None = None,
) -> ProcessedRow:
    """Build the review record from the DP audit, optionally overlaying the
    multi-label classifier's flagged/severity/category/confidence labels.

    *selected* is the published privatised output (the chosen paraphrase under
    the exponential mechanism); it is the text shown in the review table. The
    audit's ``x_priv`` is only the intermediate token-sanitised string and
    falls back here when no candidate was selected. When *clf* is given
    (``classifier='hf'``) its labels drive the review fields; otherwise they
    are derived from the proxy ``P_hate``.
    """
    proxy_p = audit.get("P_hate_original")
    x_priv = selected if selected is not None else str(audit.get("x_priv", ""))
    changed = _tokens_changed(original, x_priv)

    if clf is not None:
        flagged = bool(clf["flagged"])
        p_hate = clf["p_hate"]
        confidence = float(clf["confidence"])
        severity = clf["severity"]
        category = clf["category"]
    else:
        p = float(proxy_p) if proxy_p is not None else 0.0
        flagged = p >= HATE_THRESHOLD
        p_hate = proxy_p
        # Confidence = model probability for the predicted class (0.5..1).
        confidence = round(max(p, 1.0 - p), 3)
        # Severity is a direct banding of the proxy hate probability — display
        # mapping only, no scoring logic of our own.
        if not flagged:
            severity = "none"
        elif p >= 0.8:
            severity = "high"
        elif p >= 0.65:
            severity = "medium"
        else:
            severity = "low"
        category = None

    return ProcessedRow(
        id=idx,
        original=original,
        x_priv=x_priv,
        p_hate_original=p_hate,
        p_hate_x_priv=audit.get("P_hate_x_priv"),
        flagged=flagged,
        confidence=confidence,
        confidence_band=_confidence_band(confidence),
        severity=severity,
        category=category,
        tokens_changed=changed,
        epsilon_total=audit.get("epsilon_total"),
        fallback=bool(audit.get("fallback", False)),
    )


def _hf_classifier_available() -> bool:
    """True if the multi-label toxic-roberta classifier can be loaded."""
    try:
        from em_hsd.server.classifier import classifier
    except Exception:
        return False
    return classifier.available


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


@lru_cache(maxsize=16)
def _pipeline_cached(config_path: str):
    """Composed TRIAGE-DP pipeline for a config (Layers 1–4 when enabled).

    Cached so Layer 1's occlusion classifier and Layer 4's resources stay warm
    across requests.
    """
    from triage_dp.pipeline import build_pipeline

    return build_pipeline(_load_config_cached(config_path))


def _privatize(text: str, config_path: str, seed: int, run_seed: str) -> tuple[str, dict]:
    config = _load_config_cached(config_path)
    with _lock:
        config.spine.rng = make_row_rng(seed, run_seed=run_seed)
        # When the config enables TRIAGE-DP, run the composed pipeline
        # (Layer 1 cross-saliency → Layer 2 stylometric → Layer 3 calibrate →
        # Layer 4 rewrite); otherwise run standalone Layer 4.
        if getattr(config.triage_dp, "enabled", False):
            return _pipeline_cached(config_path).sanitize(text)
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

    @app.post("/api/process", response_model=ProcessResponse)
    def process(req: ProcessRequest) -> ProcessResponse:
        config_path = _resolve_config_name(req.config)
        # Privatisation level varies the run-level seed so different levels
        # produce different anonymisations from the same input.
        run_seed = f"{req.run_seed}-L{req.privatization_level}"

        # Resolve the labelling source. 'hf' uses the real multi-label
        # classifier when available, otherwise falls back to the proxy.
        use_hf = req.classifier == "hf" and _hf_classifier_available()
        applied_classifier = "hf" if use_hf else "proxy"

        records: list[ProcessedRow] = []
        for idx, text in enumerate(req.rows):
            stripped = text.strip()
            if not stripped:
                records.append(_derive_row(idx, text, {"x_priv": "", "P_hate_original": 0.0}))
                continue
            try:
                selected, audit = _privatize(stripped, str(config_path), idx, run_seed)
            except Exception as exc:  # surface pipeline errors as 500 with detail
                raise HTTPException(
                    status_code=500, detail=f"pipeline error on row {idx}: {exc}"
                ) from exc
            clf = None
            if use_hf:
                from em_hsd.server.classifier import classifier as _clf

                clf = _clf.classify(stripped)
            records.append(_derive_row(idx, text, audit, clf, selected))

        flagged = [r for r in records if r.flagged]
        severity_hist = {
            "high": sum(1 for r in flagged if r.severity == "high"),
            "medium": sum(1 for r in flagged if r.severity == "medium"),
            "low": sum(1 for r in flagged if r.severity == "low"),
        }
        category_hist: dict[str, int] = {}
        for r in flagged:
            key = r.category or "unassigned"
            category_hist[key] = category_hist.get(key, 0) + 1
        # Light grouping: distinct anonymised text among flagged rows.
        clusters = len({r.x_priv for r in flagged})
        count = len(records)
        return ProcessResponse(
            config=config_path.name,
            privatization_level=req.privatization_level,
            classifier=applied_classifier,
            count=count,
            flagged=len(flagged),
            flagged_pct=round(100.0 * len(flagged) / count, 1) if count else 0.0,
            low_confidence=sum(1 for r in records if r.confidence_band == "low"),
            clusters=clusters,
            severity_hist=severity_hist,
            category_hist=category_hist,
            status_hist={"flagged": len(flagged), "clean": count - len(flagged)},
            rows=records,
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
