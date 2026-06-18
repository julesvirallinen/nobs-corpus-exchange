# Migration Guide: `em-hsd-v2` → Layer-4 TRIAGE-DP Module

## Overview
The `NOBS/123` repository (`em-hsd-v2`) has been reorganised from a flat package into a layered structure that matches the full TRIAGE-DP architecture. All **old import paths continue to work** via shims, so existing notebooks and scripts do not need to change unless you want the new APIs.

## New package layout

```
src/em_hsd/
├── __init__.py                 # public: EmHsdConfig, Layer4Orchestrator, privatize_em_hsd_v2
├── core/                       # primitives used by all layers
│   ├── config.py               # EmHsdConfig, load_em_hsd_config
│   ├── dp.py                   # exponential mechanism, sensitivity, epsilon accounting
│   ├── embedding.py            # sentence encoder + utility backend abstraction
│   ├── hate_proxy.py           # lightweight hate-score proxy
│   ├── paths.py                # SpinePathResolver
│   ├── policy.py               # DownloadPolicy
│   ├── resources.py            # ResourceManager (lazy model loading + no-download guard)
│   ├── scorer.py               # semantic/hate/length scoring
│   └── utils.py                # text preprocessing helpers
├── layer4/                     # Layer-4 pipeline components
│   ├── orchestrator.py         # Layer4Orchestrator, privatize_em_hsd_v2
│   ├── filter.py               # candidate validity filters
│   ├── proposer.py             # paraphrase candidate generation
│   └── pruner.py               # candidate deduplication / selection
├── interfaces/                 # public adaptors for TRIAGE-DP integration
│   ├── triage.py               # protocols for Layer 1–3
│   ├── mock.py                 # NoOp* implementations for standalone mode
│   └── triage_dp.py            # TriageDPLayer4 single-method adapter
├── cli/
│   └── run.py                  # canonical CLI entrypoint
├── io/
│   ├── csv_io.py               # CSV read/write (moved from csv_compat)
│   └── audit_io.py             # JSONL audit writer
├── calibrate.py                # mock TO calibration harness
├── harness_integration.py      # optional TRIAGE-DP harness wrapper
└── csv_compat.py               # backward-compatible shim to em_hsd.io.csv_io
```

## Import mapping

### Before (still works)
```python
from em_hsd.config import EmHsdConfig, load_em_hsd_config
from em_hsd.privatise import privatize_em_hsd_v2
from em_hsd.csv_compat import read_csv, write_csv
import em_hsd_cli.run as cli
```

### Recommended new paths
```python
from em_hsd import EmHsdConfig, load_em_hsd_config
from em_hsd import privatize_em_hsd_v2, Layer4Orchestrator
from em_hsd.io.csv_io import read_csv, write_csv
from em_hsd.core.resources import ResourceManager
from em_hsd.interfaces.triage_dp import TriageDPLayer4
from em_hsd.cli.run import main
```

### TRIAGE-DP harness integration
```python
from em_hsd.interfaces.triage_dp import TriageDPLayer4
from em_hsd.interfaces.triage import TriageRouter, StylometricPrior, TOOptimizer

layer4 = TriageDPLayer4.from_config("configs/em-hsd-v2-triage-dp.yaml")
sanitized = layer4.sanitize(
    text,
    original_text=text,
    token_routes=[],
)
```

## CLI changes

- Old command: `python -m em_hsd_cli.run --config ...`
- New canonical commands (installed by `pyproject.toml`):
  - `em-hsd-run --config configs/em-hsd-v2-test.yaml --text "..."`
  - `triage-dp-run --config configs/em-hsd-v2-triage-dp-test.yaml --text "..."`
- The old `python -m em_hsd_cli.run` still works as a shim.

## Configuration changes

All existing YAML configs keep working. A new optional section has been added for TRIAGE-DP mode:

```yaml
triage_dp:
  enabled: true
  # layer_1, layer_2, layer_3 entries will point to real modules when merged
  layer_1:
    module: em_hsd.interfaces.mock
    class: NoOpTriageRouter
  layer_2:
    module: em_hsd.interfaces.mock
    class: NoOpStylometricPrior
  layer_3:
    module: em_hsd.interfaces.mock
    class: NoOpTOOptimizer
```

## SPINE path resolution

The hard-coded relative path to `Johnny t0-1.03/src` has been removed. Resolution order:

1. Environment variable `EM_HSD_SPINE_PATH`
2. `tool.em-hsd.spine_path` in `pyproject.toml`
3. Sibling directory `../Johnny t0-1.03/src` from the repository root
4. If none resolve, a clear error is raised only when a SPINE-based component is actually used.

## No-download policy

By default, the module **does not download any models**. To opt in, set:

```bash
export EM_HSD_ALLOW_DOWNLOADS=1
```

This is enforced in `ResourceManager` and checked by `scripts/check_no_downloads.py`.

## What you need to change

If you only use the high-level CLI or `privatize_em_hsd_v2()`: **nothing**.

If you import internal modules directly, update to the new paths or keep the shimmed old paths. Internal modules that moved include:

| Old module | New canonical location | Shim present? |
|------------|------------------------|---------------|
| `em_hsd.config` | `em_hsd.core.config` | yes |
| `em_hsd.privatise` | `em_hsd.layer4.orchestrator` | yes |
| `em_hsd.csv_compat` | `em_hsd.io.csv_io` | yes |
| `em_hsd.scorer` | `em_hsd.core.scorer` | yes |
| `em_hsd.embedding` | `em_hsd.core.embedding` | yes |
| `em_hsd.hate_proxy` | `em_hsd.core.hate_proxy` | yes |
| `em_hsd.dp` | `em_hsd.core.dp` | yes |
| `em_hsd_cli.run` | `em_hsd.cli.run` | yes |

## Testing after migration

```bash
# Run the full suite
PYTHONPATH=src pytest tests/ -q

# Run smoke evaluation
PYTHONPATH=src python scripts/smoke_eval.py --in tests/data/synthetic_smoke.csv --out /tmp/smoke.csv --config configs/em-hsd-v2-test.yaml

# Run demo
PYTHONPATH=src python scripts/demo.py --text "you are a dummy" --config configs/em-hsd-v2-test.yaml --json
```

## Getting help

- Implementation plan: `.sisyphus/plans/building-123-triage-dp.md`
- Evidence directory: `.sisyphus/evidence/`
- Research note: `docs/RESEARCH_NOTE.md`
- Fallback policy: `docs/FALLBACK_POLICY.md`
- Privacy claims: `docs/PRIVACY_CLAIMS.md`

