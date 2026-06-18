# em-hsd-v2 — Layer-4 DP Text Sanitisation

This repository is the standalone, no-download-ready **Layer-4** module of the TRIAGE-DP pipeline. It takes a hate-adjacent English sentence and returns a privacy-preserving paraphrase using differential-privacy selection over candidate paraphrases.

## Quick start

Clone the repo and run the smoke evaluation without downloading anything:

```bash
PYTHONPATH=src python scripts/demo.py \
  --text "you are a complete dummy" \
  --config configs/em-hsd-v2-test.yaml \
  --json
```

Run the full test suite:

```bash
PYTHONPATH=src pytest tests/ -q
```

Run smoke evaluation on a synthetic dataset:

```bash
PYTHONPATH=src python scripts/smoke_eval.py \
  --in tests/data/synthetic_smoke.csv \
  --out /tmp/smoke_out.csv \
  --config configs/em-hsd-v2-test.yaml
```

## What this repo does

1. **Tokenises** the input and tags protected/hate-lexicon tokens.
2. **Generates** candidate paraphrases (currently via lightweight mock/proxy models).
3. **Scores** candidates on semantic similarity and a hate-score proxy.
4. **Selects** one candidate with the exponential mechanism under `epsilon_total = epsilon_1 + epsilon_2`.
5. **Falls back** to the safe `x_priv` baseline when no candidate is valid, and reports the reason.
6. **Produces** structured JSON/CSV audit records for every decision.

## Repository layout

```
configs/                    # YAML configuration files
scripts/                    # runnable evaluation/demo/report scripts
src/
  em_hsd/
    core/                   # primitives: config, DP, scoring, embedding, resources, paths
    layer4/                 # pipeline: orchestrator, proposer, filter, pruner
    interfaces/             # TRIAGE-DP protocols and adapter
    cli/                    # command-line entry points
    io/                     # CSV + audit JSONL helpers
tests/                      # unit, integration, backward-compat, no-download tests
docs/                       # migration guide, privacy claims, fallback policy
.sisyphus/                  # active plan, session state, evidence
```

## CLI

The package exposes two console scripts:

```bash
# Standalone em-hsd-v2 mode
em-hsd-run --config configs/em-hsd-v2-test.yaml --text "..."

# TRIAGE-DP adapter mode
em-hsd-run --mode triage-dp --config configs/em-hsd-v2-triage-dp-test.yaml --text "..."
```

On a fresh checkout you can also use `python -m em_hsd.cli.run ...` directly.

## Configuration

Key YAML sections:

- `pipeline` — DP budget, candidate counts, semantic/hate thresholds.
- `model` — which lightweight backend is used (default: `proxy` or `mock`).
- `triage_dp` — optional Layer 1–3 adapter configuration for TRIAGE-DP mode.
- `tool.em-hsd` (in `pyproject.toml`) — SPINE path, no-download policy defaults.

See `configs/em-hsd-v2-test.yaml` for a fully commented example.

## No-download policy

By default, **no models are downloaded**. This is enforced by `em_hsd.core.policy.DownloadPolicy` and tested in `tests/test_no_downloads.py`. To explicitly allow downloads (for real local models), set:

```bash
export EM_HSD_ALLOW_DOWNLOADS=1
```

## TRIAGE-DP integration

The repository is designed to merge into the full TRIAGE-DP harness. The adapter is `em_hsd.interfaces.triage_dp.TriageDPLayer4`:

```python
from em_hsd.interfaces.triage_dp import TriageDPLayer4

layer4 = TriageDPLayer4.from_config("configs/em-hsd-v2-triage-dp-test.yaml")
result = layer4.sanitize(text, original_text=text, token_routes=[])
```

When integrated, real Layer 1 (triage), Layer 2 (stylometric prior), and Layer 3 (TO calibration) implementations can be plugged in via the `triage_dp` config section.

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/demo.py` | One-off sanitisation with optional JSON output |
| `scripts/smoke_eval.py` | Evaluate a CSV of synthetic inputs |
| `scripts/run_ablations.py` | Run ablation experiments A1–A9 |
| `scripts/generate_report.py` | Convert ablation + calibration JSON into Markdown |
| `scripts/check_no_downloads.py` | Verify no-download policy is active |
| `scripts/quality_gate.py` | Run ruff + mypy + pytest |

## Testing

```bash
PYTHONPATH=src pytest tests/ -q
```

Expected result on a clean machine: **31 passed, 1 skipped** (the skipped test requires `torch`, which is not installed in the no-download setup).

## Documentation

- `docs/MIGRATION.md` — moving from old flat imports to the new layout
- `docs/PRIVACY_CLAIMS.md` — what privacy is and is not guaranteed
- `docs/FALLBACK_POLICY.md` — behavior when candidates are rejected
- `docs/RESEARCH_NOTE.md` — open questions and evidence

## Development workflow

1. Make changes under `src/em_hsd/` or `tests/`.
2. Run `PYTHONPATH=src pytest tests/ -q`.
3. Run `PYTHONPATH=src python scripts/quality_gate.py` before committing.

## License / Attribution

See the parent `NOBS/TRIAGE-DP` documentation for the full research context and paper references.

