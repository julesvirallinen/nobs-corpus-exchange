# Comprehensive Plan: Build 123 into a TRIAGE-DP-Compatible Layer-4 Module

## TL;DR

> Harden the existing `NOBS/123` EM-HSD 2.0 MVP so it becomes a **production-quality, standalone Layer-4 pipeline** that can also plug cleanly into the full **TRIAGE-DP** architecture (Layers 1–5). Implement the **interface contracts** for Layers 1–3 without breaking the standalone mode. No model downloads or large data fetches during this work.
>
> **Deliverables:** refactored `em_hsd` package, merge-ready `triage_dp` integration layer, calibration harness skeleton, ablation framework, evaluation report generator, and a research-note outline.
> **Estimated Effort:** Large (5 waves, 32 tasks)
> **Parallel Execution:** YES — 5 waves
> **Critical Path:** T0–T5 (foundation) → T6–T8 (core refactor) → T13–T18 (adapter/integration) → T19–T25 (calibration/eval) → T26–T31 (config/docs/gate) → F1–F4 (final review)

---

## Context

### Original Request
User wants a comprehensive plan to "build 123" so it is **completely mergeable into full TRIAGE-DP**, while **not downloading any big files yet**.

### What 123 Is Today
- `em-hsd-v2` v0.1.0, a Python package implementing **EM-HSD 2.0 v2** (Layer-4-only) from `NOBS/TRIAGE-DP/layer-04-only-proposal-v2.md`.
- Already has: token sanitization, generative proposers (mock/transformers/unsloth), pruning, filters, exponential-mechanism selection, refined Δu, CSV wrapper, configs, and tests.
- Original MVP plan (`NOBS/.cursor/plans/em-hsd_v2_mvp_cc422a5a.plan.md`) is marked complete.

### Where It Must Go
- Become the **Layer 4 component** of the full TRIAGE-DP stack described in `TRIAGE-DP.md` and `layer-04-sentence-level-em.md`.
- Implement **clean interfaces** for Layer 1 (cross-saliency triage), Layer 2 (Biber priors), and Layer 3 (TO calibration) so that future work can drop those in without rewriting Layer 4.
- Remain **standalone runnable** as EM-HSD 2.0 for users who only want Layer 4.

### Research Documents Referenced
- `NOBS/TRIAGE-DP/TRIAGE-DP.md`
- `NOBS/TRIAGE-DP/layer-01-cross-saliency-triage.md`
- `NOBS/TRIAGE-DP/layer-02-stylometric-priors.md`
- `NOBS/TRIAGE-DP/layer-03-to-calibration.md`
- `NOBS/TRIAGE-DP/layer-04-sentence-level-em.md`
- `NOBS/TRIAGE-DP/layer-04-only-proposal-v2.md`
- `NOBS/TRIAGE-DP/layer-05-rights-architecture.md`
- `NOBS/TRIAGE-DP/Layer4 vs PrivRewrite.md`
- `NOBS/.cursor/plans/em-hsd_v2_mvp_cc422a5a.plan.md`

---

## Work Objectives

### Core Objective
Refactor and extend `NOBS/123` into a **dual-mode** package:
1. **Standalone mode:** `em-hsd-run` continues to work exactly as today.
2. **TRIAGE-DP mode:** `triage-dp` can import `em_hsd.layer4` as its sentence-level exponential-mechanism fallback for hard rows.

### Concrete Deliverables
- Refactored package layout: `em_hsd.core` (pure), `em_hsd.layer4` (pipeline), `em_hsd.interfaces` (contracts), optional `triage_dp` adapter.
- Layer 1–3 interface contracts: `TriageRouter`, `StylometricPrior`, `TOOptimizer` protocols.
- Configuration schema that supports both `em-hsd-v2` and `triage-dp` fields.
- Calibration harness skeleton (`em_hsd.calibrate`).
- Ablation framework supporting A1–A9.
- Evaluation report generator (TO, utility F1, re-ID, Burrows' Delta diagnostics).
- Research-note outline and demo script skeleton.
- Full regression test suite with mock backends.

### Definition of Done
- [ ] All mock/backend tests pass without downloading models.
- [ ] CLI produces identical output for existing configs after refactor.
- [ ] New `triage-dp` adapter can be imported without circular imports.
- [ ] Ablation runner produces a JSON report on synthetic data.
- [ ] Plan compliance review passes (F1–F4).

### Must Have
- Preserve current standalone CLI behavior.
- Keep mock proposer path working (no GPU).
- Implement protocols/interfaces for Layers 1–3.
- Add calibration harness skeleton.
- Add ablation runner.
- Add evaluation report generator.
- Document every public API.
- Enforce no-download policy with verification.

### Must NOT Have (Guardrails)
- No large model downloads during development.
- No breaking changes to existing `configs/em-hsd-v2-*.yaml` files without migration path.
- No harness imports inside `em_hsd.core` or `em_hsd.layer4`.
- No hard-coded paths outside the repo.
- No production Reddit data downloads.

---

## Verification Strategy

> **ZERO HUMAN INTERVENTION** — all verification is agent-executed.

### Test Decision
- **Infrastructure exists:** YES (`pytest`, `conftest.py`, existing tests)
- **Automated tests:** YES (tests-after for new code; existing tests must keep passing)
- **Framework:** `pytest`
- **Agent-Executed QA:** MANDATORY for every task

### QA Policy
Every task MUST include agent-executed QA scenarios. Evidence saved to `.sisyphus/evidence/task-{N}-{scenario-slug}.{ext}`.

- **Unit/module:** `pytest` single file
- **CLI:** `python -m em_hsd.cli.run ...` then inspect output CSV + JSONL
- **Integration:** `python scripts/run_ablations.py` or `pytest -m integration`
- **Config:** validate YAML with Python loader
- **Static analysis:** `pytest`, `ruff`/`flake8` if available, `mypy` if configured

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Foundation — refactor + contracts, 7 tasks):
├── T0   Enforce no-download policy with verification gate
├── T1   Audit current 123 and map to TRIAGE-DP layers
├── T2   Create package structure + public API contract + path abstraction
├── T3   Define Layer 1–3 protocols (TriageRouter, StylometricPrior, TOOptimizer)
├── T4   Refactor config loader for dual-mode schema
├── T5   Move CSV compatibility to em_hsd.io + audit JSONL writer
└── T6   Add type annotations to stable public APIs (config, interfaces, protocols)

Wave 2 (Layer 4 hardening — core pipeline, 6 tasks):
├── T7   Bulk refactor pipeline modules into em_hsd.core / layer4
├── T8   Build layer4 orchestrator with clean inputs/outputs + explicit fallback policy
├── T9   Centralize model/resource caching and no-download guards
├── T10  Harden unit tests + type annotations for refactored layout (mock path)
├── T11  Add integration test: Layer4Orchestrator with no-op Layer 1–3
└── T12  Verify backward-compatible imports still work

Wave 3 (Integration + TRIAGE-DP adapter, 6 tasks):
├── T13 Build triage-dp adapter module (em_hsd.interfaces.triage_dp)
├── T14 Wire Layer 1–3 protocols into Layer 4 orchestrator
├── T15 Implement no-op / mock implementations of Layer 1–3 for standalone mode
├── T16 Refactor CLI to support dual entrypoints with --mode triage-dp
├── T17 Add backward-compatibility shims for old import paths
└── T18 Add path abstraction for Johnny t0-1.03 / TRIAGE-DP integration

Wave 4 (Calibration + ablations + evaluation, 7 tasks):
├── T19 Build calibration harness skeleton (mock TO proxy + real-harness integration point)
├── T20 Build ablation runner supporting A1–A9
├── T21 Build evaluation report generator (TO, F1, re-ID, Burrows' Delta stub)
├── T22 Add synthetic-dev smoke evaluation (no model downloads)
├── T23 Add demo script with --show-candidates + research-note outline
├── T24 Add no-download regression test
└── T25 Add harness integration stub (works when Johnny harness available)

Wave 5 (Config + docs + quality gate, 6 tasks):
├── T26 Update all configs to dual-mode schema (keep old ones valid)
├── T27 Write migration guide from old import paths
├── T28 Finalize README / quickstart / integration status
├── T29 Document fallback policy and privacy-claim boundaries
├── T30 Pre-commit quality gate: tests, lint, typecheck, no-download check
└── T31 Final end-to-end mock smoke run + evidence capture

Wave FINAL (4 parallel reviews):
├── F1 Plan compliance audit (oracle)
├── F2 Code quality review
├── F3 Real manual QA
└── F4 Scope fidelity check
```

### Dependency Matrix (abbreviated)

- **T0:** blocks nothing; informs all
- **T1:** blocks T2–T6
- **T2:** blocks T7–T12, T13–T18
- **T3:** blocks T7, T13–T15
- **T4:** blocks T8, T13–T18, T26
- **T5:** blocks T16
- **T6:** blocks T7–T12
- **T7:** blocks T8–T12
- **T8:** blocks T9–T12, T13–T18, T19–T25
- **T9:** blocks T10–T12
- **T10–T12:** blocks T13–T18
- **T13–T18:** blocks T19–T25
- **T19–T25:** blocks T26–T31
- **T26–T31:** blocks F1–F4

---

## TODOs

- [ ] T0. **Enforce no-download policy with verification gate**

  **What to do:**
  - Create `em_hsd/core/policy.py` module with a single source of truth:
    ```python
    class DownloadPolicy:
        @classmethod
        def is_allowed(cls) -> bool: ...
    ```
  - Default returns `False` unless `EM_HSD_ALLOW_DOWNLOADS=1` is set.
  - Add a CI/pre-commit check script `scripts/check_no_downloads.py` that scans for direct download calls:
    - `transformers.AutoModelForCausalLM.from_pretrained`
    - `transformers.AutoModelForSequenceClassification.from_pretrained`
    - `transformers.AutoModelForMaskedLM.from_pretrained`
    - `unsloth.FastLanguageModel.from_pretrained`
    - `sentence_transformers.SentenceTransformer`
    - `torch.hub.download_url_to_file`
  - Assert every call goes through `ResourceManager` and respects `DownloadPolicy.is_allowed()`.
  - Fail if any direct unconditional download is found.
  - Document the no-download rule in `README.md`.

  **Must NOT do:**
  - Do not actually download anything.
  - Do not disable existing backends; only guard them.

  **Recommended Agent Profile:**
  - **Category:** `quick`

  **Parallelization:**
  - **Can Run In Parallel:** YES
  - **Parallel Group:** Wave 1
  - **Blocks:** T9, T24, T30
  - **Blocked By:** None

  **References:**
  - `src/em_hsd/generative_proposer.py` — model loading
  - `src/em_hsd/utility_scorer.py` — model loading
  - `src/em_hsd/embedding.py` — model loading
  - `layer-05-rights-architecture.md` §3.4 — local-only processing principle

  **Acceptance Criteria:**
  - [ ] `DownloadPolicy.is_allowed()` returns `False` by default
  - [ ] `scripts/check_no_downloads.py` passes on current code (or documents exceptions)
  - [ ] Gate is added to quality check T30

  **QA Scenarios:**
  ```
  Scenario: Download policy defaults to false
    Tool: Bash
    Steps:
      1. `python -c "from em_hsd.core.policy import DownloadPolicy; assert not DownloadPolicy.is_allowed()"`
    Expected Result: rc 0
    Evidence: .sisyphus/evidence/T0-policy-default.log

  Scenario: No-download check script runs
    Tool: Bash
    Steps:
      1. `python scripts/check_no_downloads.py`
    Expected Result: rc 0 or documented exceptions list
    Evidence: .sisyphus/evidence/T0-no-download-check.log
  ```

  **Commit:** YES — `feat(policy): add no-download policy gate and checker`

  **Rollback strategy:** If any test legitimately needs a download in future work, gate must be explicitly disabled via `EM_HSD_ALLOW_DOWNLOADS=1` and documented, not removed.

- [ ] T1. **Audit current 123 and map to TRIAGE-DP layers**

  **What to do:**
  - Read every file in `src/em_hsd/`, `src/em_hsd_cli/`, `configs/`, `scripts/`, `tests/`.
  - Produce a mapping table: current module → TRIAGE-DP layer/component.
  - Identify import cycles, hard-coded paths, and boundary violations.
  - Document which parts already match the TRIAGE-DP spec and which need adapters.

  **Must NOT do:**
  - Do not modify code; this is inventory only.
  - Do not download models or data.

  **Recommended Agent Profile:**
  - **Category:** `explore`
  - **Reason:** Codebase archaeology and cross-referencing with research docs.

  **Parallelization:**
  - **Can Run In Parallel:** YES
  - **Parallel Group:** Wave 1
  - **Blocks:** T2–T6
  - **Blocked By:** None

  **References:**
  - `src/em_hsd/__init__.py` — public API
  - `src/em_hsd/em_hsd_v2.py:1` — orchestrator docstring
  - `src/em_hsd/spine_bootstrap.py:8` — hard-coded `Johnny t0-1.03` path
  - `layer-04-only-proposal-v2.md` — target Layer-4 spec
  - `layer-04-sentence-level-em.md` — how Layer 4 plugs into full TRIAGE-DP
  - `TRIAGE-DP.md` §9 — implementation map

  **Acceptance Criteria:**
  - [ ] Markdown inventory file exists at `.sisyphus/evidence/T1-inventory.md`
  - [ ] Table maps every `src/em_hsd/*.py` to a TRIAGE-DP component
  - [ ] List of boundary violations with file:line citations
  - [ ] List of hard-coded paths and external dependencies

  **QA Scenarios:**
  ```
  Scenario: Inventory is complete
    Tool: Bash
    Steps:
      1. Count Python files under src/em_hsd: `Get-ChildItem src/em_hsd/*.py | Measure-Object`
      2. Compare against inventory table
    Expected Result: Inventory table contains every file found; no unlisted runtime modules
    Evidence: .sisyphus/evidence/T1-inventory.md
  ```

  **Commit:** NO (inventory artifact for planning)

- [ ] T2. **Create package structure + public API contract + path abstraction**

  **What to do:**
  - Create new directory layout under `src/em_hsd/`:
    - `core/` — DP primitives, sanitization, config, utilities, policy
    - `layer4/` — orchestrator, proposers, filters, pruners, scorers
    - `interfaces/` — protocol definitions for Layer 1–3 adapters
    - `cli/` — existing CLI entry points
    - `io/` — CSV compatibility and audit writers
  - Add `__init__.py` files with explicit public API exports:
    - `em_hsd.__all__` lists `EmHsdConfig`, `load_em_hsd_config`, `privatize_em_hsd_v2`, `Layer4Orchestrator`
    - `em_hsd.core.__all__` lists `TokenSanitizer`, `DPSelector`, `EmHsdConfig`, `load_em_hsd_config`, `ResourceManager`, `DownloadPolicy`
    - `em_hsd.layer4.__all__` lists `Layer4Orchestrator`, `MockProposer`, `FilterLayer`, `Pruner`, `HateUtilityScorer`
    - `em_hsd.interfaces.__all__` lists protocols
  - Add `SpinePathResolver` **skeleton** in `em_hsd.core.paths` that finds `Johnny t0-1.03` via:
    1. Environment variable `EM_HSD_SPINE_PATH`
    2. Sibling directory `../Johnny t0-1.03`
    3. `pyproject.toml` `[tool.em-hsd-v2] spine-path` setting
  - Keep old module paths working via re-export shims in `em_hsd/` root.
  - **Note:** T18 will wire `SpinePathResolver` into actual SPINE imports; T2 only creates the skeleton and API.

  **Must NOT do:**
  - Do not rewrite algorithms; only move code.
  - Do not change public function signatures yet.

  **Recommended Agent Profile:**
  - **Category:** `quick`
  - **Skills:** `git`

  **Parallelization:**
  - **Can Run In Parallel:** YES with T1
  - **Parallel Group:** Wave 1
  - **Blocks:** T7–T12
  - **Blocked By:** None

  **References:**
  - `layer-04-only-proposal-v2.md` §11.1 — module structure
  - Current `src/em_hsd/` layout
  - `src/em_hsd/spine_bootstrap.py` — current hard-coded path

  **Acceptance Criteria:**
  - [ ] Directory layout exists and imports cleanly
  - [ ] `python -c "import em_hsd; import em_hsd.core; import em_hsd.layer4; import em_hsd.interfaces; import em_hsd.cli; import em_hsd.io"` succeeds
  - [ ] Old imports still work: `from em_hsd import privatize_em_hsd_v2`
  - [ ] `SpinePathResolver` finds sibling `Johnny t0-1.03` when present

  **QA Scenarios:**
  ```
  Scenario: New layout imports cleanly
    Tool: Bash
    Steps:
      1. `python -c "import em_hsd.core, em_hsd.layer4, em_hsd.interfaces, em_hsd.cli, em_hsd.io, em_hsd.policy"`
    Expected Result: No ImportError
    Evidence: .sisyphus/evidence/T2-layout-imports.log

  Scenario: Backward compatibility preserved
    Tool: Bash
    Steps:
      1. `python -c "from em_hsd import privatize_em_hsd_v2, load_em_hsd_config, EmHsdConfig"`
    Expected Result: Imports succeed
    Evidence: .sisyphus/evidence/T2-backcompat.log
  ```

  **Commit:** YES — `refactor(structure): reorganize em_hsd into core/layer4/interfaces/cli/io/policy`

- [ ] T3. **Define Layer 1–3 protocols: `TriageRouter`, `StylometricPrior`, `TOOptimizer`**

  **What to do:**
  - In `em_hsd.interfaces`, create `Protocol` classes:
    - `TriageRouter` with method `route_tokens(text, config) -> List[TokenRoute]`
    - `StylometricPrior` with method `boost(text, token_routes, config) -> List[TokenRoute]`
    - `TOOptimizer` with method `optimize(dev_rows, config) -> OptimizedConfig`
  - Define `TokenRoute` dataclass with fields matching `layer-01-cross-saliency-triage.md` §8.1.
  - Define `OptimizedConfig` dataclass for Layer 3 output.

  **Must NOT do:**
  - Do not implement full occlusion probes (Layer 1) yet — only protocols.
  - Do not implement Biber tagger (Layer 2) yet.
  - Do not implement Optuna calibration (Layer 3) yet.

  **Recommended Agent Profile:**
  - **Category:** `quick`

  **Parallelization:**
  - **Can Run In Parallel:** YES
  - **Parallel Group:** Wave 1
  - **Blocks:** T14–T16
  - **Blocked By:** None

  **References:**
  - `layer-01-cross-saliency-triage.md` §6.2, §8.1 — quadrant actions + TokenRoute dataclass
  - `layer-02-stylometric-priors.md` §5.1 — `biber.py` module design
  - `layer-03-to-calibration.md` §6.2 — `harness/calibrate.py` module structure

  **Acceptance Criteria:**
  - [ ] Protocols defined in `em_hsd/interfaces/triage.py`
  - [ ] `TokenRoute` dataclass exists with all required fields
  - [ ] Unit test: `tests/test_interfaces.py` asserts protocols can be implemented by mock classes

  **QA Scenarios:**
  ```
  Scenario: Mock implementations satisfy protocols
    Tool: pytest
    Steps:
      1. Run `pytest tests/test_interfaces.py -v`
    Expected Result: PASS
    Evidence: .sisyphus/evidence/T3-interfaces-test.log
  ```

  **Commit:** YES — `feat(interfaces): add TriageRouter, StylometricPrior, TOOptimizer protocols`

- [ ] T4. **Refactor config loader for dual-mode schema**

  **What to do:**
  - Extend `EmHsdConfig` to carry optional `triage_dp` section:
    - `triage_dp.enabled: bool`
    - `triage_dp.layer1: dict` (thresholds, probes)
    - `triage_dp.layer2: dict` (Biber boosts)
    - `triage_dp.layer3: dict` (optimizer settings)
    - `triage_dp.layer4: dict` (references existing em_hsd_v2 settings)
  - Ensure existing `em_hsd_v2` configs still load unchanged.
  - Add validation helper `config.is_triage_dp_mode()`.

  **Must NOT do:**
  - Do not break existing YAML files.
  - Do not require new config fields for standalone mode.

  **Recommended Agent Profile:**
  - **Category:** `quick`

  **Parallelization:**
  - **Can Run In Parallel:** YES
  - **Parallel Group:** Wave 1
  - **Blocks:** T8, T14–T18, T26
  - **Blocked By:** None

  **References:**
  - `src/em_hsd/config.py` — current config loader
  - `layer-03-to-calibration.md` §3.1 — θ parameter list
  - `layer-04-only-proposal-v2.md` §6 — hyperparameters

  **Acceptance Criteria:**
  - [ ] Existing configs load and produce same object shape
  - [ ] New `triage-dp-test.yaml` config parses and exposes all sections
  - [ ] `pytest tests/test_em_hsd_v2.py::test_load_config` still passes

  **QA Scenarios:**
  ```
  Scenario: Dual-mode config loads
    Tool: Bash
    Steps:
      1. Create `configs/em-hsd-v2-triage-dp.yaml` with triage_dp section
      2. Run `python -c "from em_hsd import load_em_hsd_config; c = load_em_hsd_config('configs/em-hsd-v2-triage-dp.yaml'); print(c.em_hsd_v2.k_generate, c.triage_dp.enabled)"`
    Expected Result: Both sections parsed, no error
    Evidence: .sisyphus/evidence/T4-config-load.log
  ```

  **Commit:** YES — `feat(config): add triage_dp dual-mode schema`

- [ ] T5. **Move CSV compatibility to `em_hsd.io` + audit JSONL writer**

  **What to do:**
  - Move `csv_compat.py` to `em_hsd/io/csv_io.py`.
  - Preserve backward-compatible import at `em_hsd.csv_compat`.
  - Add audit JSONL writer class `AuditJsonlWriter` in `em_hsd/io/audit_io.py`.
  - The CLI should use `AuditJsonlWriter` instead of inline JSONL formatting.

  **Must NOT do:**
  - Do not change CSV I/O behavior.

  **Recommended Agent Profile:**
  - **Category:** `quick`

  **Parallelization:**
  - **Can Run In Parallel:** YES
  - **Parallel Group:** Wave 1
  - **Blocks:** T16
  - **Blocked By:** None

  **References:**
  - `src/em_hsd/csv_compat.py` — existing CSV compatibility
  - `src/em_hsd_cli/run.py:168-191` — inline JSONL writing

  **Acceptance Criteria:**
  - [ ] `tests/test_csv_compat.py` passes after move
  - [ ] JSONL output format unchanged

  **QA Scenarios:**
  ```
  Scenario: CSV compat tests still pass
    Tool: pytest
    Steps:
      1. `pytest tests/test_csv_compat.py -v`
    Expected Result: PASS
    Evidence: .sisyphus/evidence/T5-csv-tests.log
  ```

  **Commit:** YES — `refactor(io): move csv_compat to em_hsd.io`

- [ ] T6. **Add type annotations to stable public APIs (config, interfaces, protocols)**

  **What to do:**
  - Add `from __future__ import annotations` to stable public modules.
  - Add type hints to:
    - `EmHsdConfig` dataclass fields
    - `TokenRoute`, `FilterBatch`, `FilterResult` fields (copy to interfaces)
    - Protocol method signatures in `em_hsd.interfaces`
    - `ResourceManager` public methods
    - `AuditJsonlWriter` public methods
  - Add `py.typed` marker file.
  - Add minimal `mypy` config in `pyproject.toml`.
  - Leave pipeline implementation files (`token_sanitize`, `dp_select`, etc.) for T10 after refactor.

  **Must NOT do:**
  - Do not annotate files that will be moved in T7 unless trivial.

  **Recommended Agent Profile:**
  - **Category:** `quick`

  **Parallelization:**
  - **Can Run In Parallel:** YES
  - **Parallel Group:** Wave 1
  - **Blocks:** T10
  - **Blocked By:** T2, T4, T5

  **References:**
  - `src/em_hsd/config.py`
  - `em_hsd.interfaces` protocols
  - Python typing best practices

  **Acceptance Criteria:**
  - [ ] `py.typed` exists in `src/em_hsd/`
  - [ ] Stable public functions have type hints
  - [ ] `mypy src/em_hsd/core/config.py src/em_hsd/interfaces/triage.py` runs without errors

  **QA Scenarios:**
  ```
  Scenario: Public APIs have type hints
    Tool: Bash
    Steps:
      1. `python -c "import inspect; from em_hsd.core.config import EmHsdConfig; print(inspect.signature(EmHsdConfig.__init__))"`
    Expected Result: Signature contains typed parameters
    Evidence: .sisyphus/evidence/T6-type-hints.log
  ```

  **Commit:** YES — `feat(types): add type annotations and py.typed marker for stable APIs`

- [ ] T7. **Bulk refactor pipeline modules into `em_hsd.core` / `em_hsd.layer4`**

  **What to do:**
  - Move the following into `em_hsd/core/` (DP primitives + shared utilities):
    - `token_sanitize.py` → `core/sanitize.py`
    - `dp_select.py` → `core/dp_select.py`
    - `config.py` → `core/config.py`
    - `embedding.py` → `core/embedding.py`
    - `resources.py` → `core/resources.py`
    - `sensitivity.py` → `core/sensitivity.py`
  - Move the following into `em_hsd/layer4/` (pipeline components):
    - `generative_proposer.py` → `layer4/proposer.py`
    - `constraints.py` → `layer4/filter.py`
    - `utility_scorer.py` → `layer4/scorer.py`
    - `prune_candidates.py` → `layer4/prune.py`
    - `em_hsd_v2.py` → `layer4/orchestrator.py`
  - Define Protocols in `em_hsd/interfaces/`:
    - `GenerativeProposer`, `CandidateFilter`, `UtilityScorer`, `CandidatePruner`
  - Add backward-compatible shims for every old import path.
  - Ensure no code duplication: shims only re-export.
  - **Internal staging:** Move core first, verify tests; then move layer4; then add protocols. Do one final combined commit.

  **Must NOT do:**
  - Do not change any algorithm in this step.
  - Do not implement real Layer 1–3 logic.
  - Do not download models.

  **Recommended Agent Profile:**
  - **Category:** `quick`
  - **Skills:** `git`
  - **Reason:** Mechanical reorganization, one coherent task.

  **Parallelization:**
  - **Can Run In Parallel:** NO — single coherent refactor
  - **Parallel Group:** Wave 2 (single bulk task)
  - **Blocks:** T8–T12
  - **Blocked By:** T2, T3, T6

  **References:**
  - All current `src/em_hsd/*.py` files
  - `layer-04-only-proposal-v2.md` §11.1 — module structure

  **Acceptance Criteria:**
  - [ ] Every old import path still works via shim
  - [ ] New import paths work
  - [ ] `pytest tests/` passes without code changes to test logic

  **QA Scenarios:**
  ```
  Scenario: New and old import paths both work
    Tool: Bash
    Steps:
      1. `python -c "from em_hsd.core.sanitize import TokenSanitizer; from em_hsd.token_sanitize import token_sanitize; from em_hsd.layer4.proposer import MockProposer; from em_hsd.generative_proposer import get_proposer; from em_hsd.layer4.orchestrator import Layer4Orchestrator; from em_hsd.em_hsd_v2 import privatize_em_hsd_v2"`
    Expected Result: No ImportError
    Evidence: .sisyphus/evidence/T7-import-paths.log

  Scenario: Tests pass after refactor
    Tool: pytest
    Steps:
      1. `pytest tests/ -v`
    Expected Result: PASS
    Evidence: .sisyphus/evidence/T7-tests-pass.log
  ```

  **Commit:** YES — `refactor(structure): bulk move modules into core/layer4 with shims`

  **Rollback strategy:** If `pytest` fails after move, restore old files from git and retry in smaller chunks (core first, then layer4). Do not leave repo in broken state overnight.

- [ ] T8. **Build `Layer4Orchestrator` with clean inputs/outputs and explicit fallback policy**

  **What to do:**
  - In `em_hsd/layer4/orchestrator.py`, implement `Layer4Orchestrator` class with:
    ```python
    def privatize(
        self,
        text: str,
        config: EmHsdConfig,
        *,
        layer1_routes: Optional[List[TokenRoute]] = None,
        layer2_routes: Optional[List[TokenRoute]] = None,
        layer3_overrides: Optional[Dict[str, Any]] = None,
    ) -> tuple[str, dict]:
    ```
  - Document fallback policy inline:
    - ≥2 valid candidates → EM select with ε₂ + refined Δu
    - 1 valid candidate → return it
    - 0 valid / proposer fail → return `x_priv`
    - Never return raw text
  - Keep `privatize_em_hsd_v2(text, config)` as a thin wrapper.

  **Must NOT do:**
  - Do not change pipeline semantics.
  - Do not add real Layer 1–3 logic yet; just accept optional routes.

  **Recommended Agent Profile:**
  - **Category:** `unspecified-high`
  - **Reason:** Critical API design for TRIAGE-DP integration.

  **Parallelization:**
  - **Can Run In Parallel:** NO — depends on T7
  - **Parallel Group:** Wave 2 (finalizer)
  - **Blocks:** T9–T12, T13–T18, T19–T25
  - **Blocked By:** T2, T3, T7

  **References:**
  - `src/em_hsd/em_hsd_v2.py` — current orchestrator
  - `layer-04-only-proposal-v2.md` §3.1, §5.5 — pipeline + fallback
  - `layer-04-sentence-level-em.md` §8 — orchestration class

  **Acceptance Criteria:**
  - [ ] `pytest tests/test_em_hsd_v2.py` all tests pass
  - [ ] `python -m em_hsd_cli.run` on synthetic data still works
  - [ ] New orchestrator method signature documented in docstring

  **QA Scenarios:**
  ```
  Scenario: Full mock pipeline still works
    Tool: pytest
    Steps:
      1. `pytest tests/test_em_hsd_v2.py -v`
    Expected Result: PASS
    Evidence: .sisyphus/evidence/T8-orchestrator-test.log
  ```

  **Commit:** YES — `refactor(layer4): introduce Layer4Orchestrator with explicit fallback policy`

- [ ] T9. **Centralize model/resource caching and no-download guards**

  **What to do:**
  - Refactor the moved `em_hsd/core/resources.py` into a `ResourceManager`:
    - Lazy singletons for scorer, encoder, proposer, MLM backend
    - `allow_downloads: bool` flag (default False for this work)
    - Raises clear `RuntimeError` if a requested backend requires a missing local model
  - Update `get_scorer`, `get_encoder`, `get_proposer` to use `ResourceManager`.
  - Add environment variable `EM_HSD_ALLOW_DOWNLOADS` override.

  **Must NOT do:**
  - Do not trigger downloads during tests or normal import.
  - Do not change backend selection logic.

  **Recommended Agent Profile:**
  - **Category:** `quick`

  **Parallelization:**
  - **Can Run In Parallel:** YES
  - **Parallel Group:** Wave 2
  - **Blocks:** T10–T12
  - **Blocked By:** T2, T7

  **References:**
  - `src/em_hsd/resources.py` — existing resource helpers
  - `src/em_hsd/utility_scorer.py` — scorer caching
  - `src/em_hsd/embedding.py` — encoder caching
  - `src/em_hsd/generative_proposer.py` — proposer caching

  **Acceptance Criteria:**
  - [ ] Mock backend works with `allow_downloads=False`
  - [ ] `ResourceManager` rejects HF/Unsloth backends if model not cached and downloads disallowed
  - [ ] `pytest tests/` still passes

  **QA Scenarios:**
  ```
  Scenario: No-download guard blocks HF backend without local cache
    Tool: Bash
    Steps:
      1. With empty HF cache, attempt `python -c "from em_hsd.core.resources import ResourceManager; rm=ResourceManager(allow_downloads=False); rm.get_proposer(load_config('configs/em-hsd-v2-qwen35-08b.yaml'))"`
    Expected Result: RuntimeError mentioning downloads disabled
    Evidence: .sisyphus/evidence/T9-no-download-guard.log
  ```

  **Commit:** YES — `feat(core): centralize resource manager with no-download guard`

- [ ] T10. **Harden unit tests for refactored layout (mock path)**

  **What to do:**
  - Update `tests/test_em_hsd_v2.py` to import from new paths (or keep shims and add new-path tests).
  - Add explicit tests for:
    - New import paths
    - Old import paths via shims
    - `Layer4Orchestrator` fallback branches
    - `ResourceManager` no-download guard
  - Add `tests/test_layer4_orchestrator.py`.
  - Add type annotations to refactored public APIs (`Layer4Orchestrator`, `FilterLayer`, `Pruner`, etc.).
  - Update `tests/conftest.py` with fixtures for new paths/config if needed.
  - Add `pytest` markers: `@pytest.mark.slow` for generation tests, `@pytest.mark.gpu` for Unsloth/transformers tests.
  - Add marker definitions to `pyproject.toml`.

  **Must NOT do:**
  - Do not remove existing tests.

  **Recommended Agent Profile:**
  - **Category:** `quick`

  **Parallelization:**
  - **Can Run In Parallel:** YES
  - **Parallel Group:** Wave 2
  - **Blocks:** T12
  - **Blocked By:** T7, T8, T9

  **References:**
  - Existing `tests/test_em_hsd_v2.py`
  - `layer-04-only-proposal-v2.md` §10.2 — ablation-style tests

  **Acceptance Criteria:**
  - [ ] `pytest tests/` passes
  - [ ] New tests cover orchestrator fallback + resource guard

  **QA Scenarios:**
  ```
  Scenario: Refactored test suite passes
    Tool: pytest
    Steps:
      1. `pytest tests/ -v`
    Expected Result: PASS, including new tests
    Evidence: .sisyphus/evidence/T10-refactored-tests.log
  ```

  **Commit:** YES — `test(layer4): add refactored-layout unit tests and type annotations`

- [ ] T11. **Add integration test: Layer4Orchestrator with no-op Layer 1–3**

  **What to do:**
  - Create `tests/test_triage_dp_integration.py`.
  - Instantiate `Layer4Orchestrator` with no-op `TriageRouter`, `StylometricPrior`, `TOOptimizer`.
  - Verify output equals standalone mode output.
  - Verify audit dict contains expected fields.

  **Must NOT do:**
  - Do not require real models.

  **Recommended Agent Profile:**
  - **Category:** `quick`

  **Parallelization:**
  - **Can Run In Parallel:** YES
  - **Parallel Group:** Wave 2
  - **Blocks:** T13–T18
  - **Blocked By:** T8

  **References:**
  - `em_hsd.interfaces` protocols
  - `em_hsd.layer4.orchestrator`

  **Acceptance Criteria:**
  - [ ] Test passes and asserts identical output between standalone and no-op triage-dp paths

  **QA Scenarios:**
  ```
  Scenario: No-op triage-dp path matches standalone
    Tool: pytest
    Steps:
      1. `pytest tests/test_triage_dp_integration.py -v`
    Expected Result: PASS
    Evidence: .sisyphus/evidence/T11-triage-dp-integration.log
  ```

  **Commit:** YES — `test(integration): add no-op Layer 1–3 integration test`

- [ ] T12. **Verify backward-compatible imports still work**

  **What to do:**
  - Add `tests/test_backward_compat.py` that imports every old public symbol.
  - List old import paths and assert they still work.

  **Must NOT do:**
  - Do not add new features.

  **Recommended Agent Profile:**
  - **Category:** `quick`

  **Parallelization:**
  - **Can Run In Parallel:** YES
  - **Parallel Group:** Wave 2
  - **Blocks:** T17
  - **Blocked By:** T7

  **References:**
  - All public imports used in original tests and scripts

  **Acceptance Criteria:**
  - [ ] `pytest tests/test_backward_compat.py` passes

  **QA Scenarios:**
  ```
  Scenario: Backward compatibility test passes
    Tool: pytest
    Steps:
      1. `pytest tests/test_backward_compat.py -v`
    Expected Result: PASS
    Evidence: .sisyphus/evidence/T12-backward-compat.log
  ```

  **Commit:** YES — `test(compat): add backward-compatibility import test`

- [ ] T13. **Build `triage-dp` adapter module (`em_hsd.interfaces.triage_dp`)**

  **What to do:**
  - Create `em_hsd/interfaces/triage_dp.py` as a **temporary adapter** (will move to a dedicated `triage_dp` package when full TRIAGE-DP is built).
  - In it, define `TriageDPLayer4` class that wraps `Layer4Orchestrator`.
    - Method `sanitize(text, config, token_routes=None)` returning `(text, audit)`
    - Glue to convert Layer 1 `TokenRoute` list into protected spans and optional ε overrides
  - This adapter is what the future full TRIAGE-DP repo will import from 123.

  **Must NOT do:**
  - Do not create a separate `triage_dp` package yet; keep adapter inside `em_hsd.interfaces`.
  - Do not require TRIAGE-DP dependencies.

  **Recommended Agent Profile:**
  - **Category:** `quick`

  **Parallelization:**
  - **Can Run In Parallel:** YES
  - **Parallel Group:** Wave 3
  - **Blocks:** T16
  - **Blocked By:** T3, T8

  **References:**
  - `layer-04-sentence-level-em.md` §8 — orchestration class integration
  - `layer-01-cross-saliency-triage.md` §8.1 — TokenRoute schema
  - `TRIAGE-DP.md` §3 architecture diagram

  **Acceptance Criteria:**
  - [ ] Adapter can be imported: `from em_hsd.interfaces.triage_dp import TriageDPLayer4`
  - [ ] Adapter can be instantiated with a config and `.sanitize(text, config)` runs the mock pipeline

  **QA Scenarios:**
  ```
  Scenario: Adapter runs in standalone mode
    Tool: Bash
    Steps:
      1. `python -c "from em_hsd.interfaces.triage_dp import TriageDPLayer4; from em_hsd import load_em_hsd_config; cfg = load_em_hsd_config('configs/em-hsd-v2-test.yaml'); layer4 = TriageDPLayer4(cfg); out, audit = layer4.sanitize('hello world', cfg); print(out)"`
    Expected Result: Output string printed, no exception
    Evidence: .sisyphus/evidence/T13-adapter-smoke.log
  ```

  **Commit:** YES — `feat(interfaces): add triage-dp Layer4 adapter`

- [ ] T14. **Wire Layer 1–3 protocols into Layer 4 orchestrator**

  **What to do:**
  - Modify `Layer4Orchestrator.privatize` to accept optional `layer1_routes`, `layer2_routes`, `layer3_overrides`.
  - If `layer1_routes` provided:
    - Convert routes to `protected_override` set of canonical skeletons.
    - Pass `protected_override` to `TokenSanitizer` so Q3/Q1 tokens are frozen.
    - For Q2-only tokens, use `epsilon_Q2` from `layer3_overrides` or config.
  - If `layer2_routes` provided, no-op (boost already applied by Layer 2 upstream).
  - If `layer3_overrides` provided, override `epsilon_total`, `epsilon_split`, `hate_floor_delta`, `tau_sem_min`.
  - If no Layer 1–3 input, behave exactly as today (standalone mode).

  **Must NOT do:**
  - Do not implement real Layer 1–3 computations.
  - Do not break standalone behavior.

  **Recommended Agent Profile:**
  - **Category:** `unspecified-high`

  **Parallelization:**
  - **Can Run In Parallel:** YES
  - **Parallel Group:** Wave 3
  - **Blocks:** T15–T16
  - **Blocked By:** T3, T8, T13

  **References:**
  - `layer-04-sentence-level-em.md` §8 — how token-level pass feeds Layer 4
  - `layer-01-cross-saliency-triage.md` §6.2 — quadrant actions
  - `layer-03-to-calibration.md` §3.1 — θ parameters

  **Acceptance Criteria:**
  - [ ] Orchestrator accepts optional Layer 1–3 inputs without error
  - [ ] Standalone mode tests still pass

  **QA Scenarios:**
  ```
  Scenario: Orchestrator ignores unknown Layer 1–3 inputs gracefully in standalone mode
    Tool: pytest
    Steps:
      1. `pytest tests/test_layer4_integration.py -v`
    Expected Result: PASS
    Evidence: .sisyphus/evidence/T14-wire-test.log
  ```

  **Commit:** YES — `feat(layer4): wire Layer 1–3 protocol inputs into orchestrator`

- [ ] T15. **Implement no-op / mock implementations of Layer 1–3 for standalone mode**

  **What to do:**
  - In `em_hsd.interfaces.mock`:
    - `NoOpTriageRouter` — returns empty routes; causes standalone behavior
    - `NoOpStylometricPrior` — returns routes unchanged
    - `NoOpTOOptimizer` — returns config as-is
  - Wire these as defaults when `triage_dp.enabled: false`.

  **Must NOT do:**
  - Do not implement real probes.

  **Recommended Agent Profile:**
  - **Category:** `quick`

  **Parallelization:**
  - **Can Run In Parallel:** YES
  - **Parallel Group:** Wave 3
  - **Blocks:** T16
  - **Blocked By:** T3, T14

  **References:**
  - `layer-01-cross-saliency-triage.md` §8.1 — TokenRoute dataclass
  - `layer-02-stylometric-priors.md` §5.1 — `apply_priors(routes, config)`
  - `layer-03-to-calibration.md` §6.2 — `apply_theta(config, theta)`

  **Acceptance Criteria:**
  - [ ] No-op classes implement protocols
  - [ ] Standalone config uses no-op classes by default

  **QA Scenarios:**
  ```
  Scenario: No-op classes are protocol-compliant
    Tool: pytest
    Steps:
      1. `pytest tests/test_interfaces.py -k no_op -v`
    Expected Result: PASS
    Evidence: .sisyphus/evidence/T15-noop-test.log
  ```

  **Commit:** YES — `feat(interfaces): add no-op Layer 1–3 implementations`

- [ ] T16. **Refactor CLI to support dual entrypoints with `--mode triage-dp`**

  **What to do:**
  - Move `src/em_hsd_cli/run.py` to `src/em_hsd/cli/run.py`.
  - Update `pyproject.toml` `[project.scripts]`:
    - `em-hsd-run = "em_hsd.cli.run:main"`
    - `triage-dp-run = "em_hsd.cli.run:main"` (default `--mode triage-dp`)
  - Keep console script `em-hsd-run` working.
  - Add `--mode triage-dp` (no-op path: uses no-op Layer 1–3, same output as today).
  - Use `em_hsd.io.audit_io.AuditJsonlWriter`.

  **Must NOT do:**
  - Do not remove `--mode em-hsd-v2`.
  - Do not implement full TRIAGE-DP CLI yet.

  **Recommended Agent Profile:**
  - **Category:** `quick`

  **Parallelization:**
  - **Can Run In Parallel:** YES
  - **Parallel Group:** Wave 3
  - **Blocks:** T26
  - **Blocked By:** T5, T8, T13, T15

  **References:**
  - `src/em_hsd_cli/run.py` — current CLI
  - `layer-04-only-proposal-v2.md` §11.3 — quickstart
  - `TRIAGE-DP.md` §13 — quickstart

  **Acceptance Criteria:**
  - [ ] `em-hsd-run` still works on test config
  - [ ] `python -m em_hsd.cli.run --mode triage-dp ...` produces identical output to `--mode em-hsd-v2` with no-op Layer 1–3

  **QA Scenarios:**
  ```
  Scenario: CLI dual modes produce same output on no-op config
    Tool: Bash
    Steps:
      1. Create 1-row CSV `tests/data/smoke.csv`
      2. `python -m em_hsd.cli.run --in tests/data/smoke.csv --out /tmp/a.csv --config configs/em-hsd-v2-test.yaml --mode em-hsd-v2`
      3. `python -m em_hsd.cli.run --in tests/data/smoke.csv --out /tmp/b.csv --config configs/em-hsd-v2-test.yaml --mode triage-dp`
      4. `diff /tmp/a.csv /tmp/b.csv`
    Expected Result: Files identical except timestamps/log row numbers
    Evidence: .sisyphus/evidence/T16-cli-dual-mode.log
  ```

  **Commit:** YES — `feat(cli): support dual em-hsd-v2 / triage-dp modes`

- [ ] T17. **Add backward-compatibility shims for old import paths**

  **What to do:**
  - Add `src/em_hsd/csv_compat.py` shim to `em_hsd.io.csv_io`.
  - Add `src/em_hsd/token_sanitize.py` shim.
  - Add `src/em_hsd/generative_proposer.py` shim.
  - Add `src/em_hsd/constraints.py`, `src/em_hsd/prune_candidates.py`, `src/em_hsd/dp_select.py`, `src/em_hsd/utility_scorer.py`, `src/em_hsd/em_hsd_v2.py`, `src/em_hsd/sensitivity.py`, `src/em_hsd/embedding.py`, `src/em_hsd/resources.py`, `src/em_hsd/config.py` shims.
  - Keep `src/em_hsd_cli/run.py` as a thin wrapper.
  - Run full test suite.

  **Must NOT do:**
  - Do not duplicate logic in shims.

  **Recommended Agent Profile:**
  - **Category:** `quick`

  **Parallelization:**
  - **Can Run In Parallel:** NO — must follow T7
  - **Parallel Group:** Wave 3
  - **Blocks:** T26–T31
  - **Blocked By:** T7, T12

  **References:**
  - All existing `src/em_hsd/*.py` public import paths

  **Acceptance Criteria:**
  - [ ] `pytest tests/` passes with all shims
  - [ ] `python -m em_hsd_cli.run` still works
  - [ ] `python -c "import em_hsd; from em_hsd import privatize_em_hsd_v2; from em_hsd.csv_compat import read_csv_compat; from em_hsd_cli.run import main"` raises no ImportError and no circular import
  - [ ] `python -c "from em_hsd.interfaces.triage_dp import TriageDPLayer4; print(TriageDPLayer4)"` works from both new and legacy packages

  **QA Scenarios:**
  ```
  Scenario: Old import paths work
    Tool: Bash
    Steps:
      1. `python -c "import em_hsd; from em_hsd import privatize_em_hsd_v2; from em_hsd.csv_compat import read_csv_compat; from em_hsd_cli.run import main; from em_hsd.interfaces.triage_dp import TriageDPLayer4; print(TriageDPLayer4)"`
    Expected Result: rc 0; no ImportError
    Evidence: .sisyphus/evidence/T17-shims.log
  ```

  **Commit:** YES — `feat(compat): add backward-compatible import shims`

- [ ] T18. **Add path abstraction for `Johnny t0-1.03` / TRIAGE-DP integration**

  **What to do:**
  - Replace `spine_bootstrap.py` hard-coded path with `SpinePathResolver` from T2.
  - Add environment variable `EM_HSD_SPINE_PATH` support.
  - Add `pyproject.toml` `[tool.em-hsd-v2]` config section for `spine-path`.
  - Add `EmHsdConfig.spine_path` property that uses resolver.
  - If `Johnny t0-1.03` is missing, raise clear error pointing to setup instructions.

  **Must NOT do:**
  - Do not require `Johnny t0-1.03` at plan time.
  - Do not change SPINE import behavior when path exists.

  **Recommended Agent Profile:**
  - **Category:** `quick`

  **Parallelization:**
  - **Can Run In Parallel:** YES
  - **Parallel Group:** Wave 3
  - **Blocked By:** T2
  - **Blocks:** T19–T25

  **References:**
  - `src/em_hsd/spine_bootstrap.py`
  - `TRIAGE-DP.md` §9 — implementation map referencing `Johnny t0-1.03`

  **Acceptance Criteria:**
  - [ ] `SpinePathResolver` resolves sibling directory by default
  - [ ] Custom path via env var works
  - [ ] Missing SPINE path raises helpful RuntimeError

  **QA Scenarios:**
  ```
  Scenario: Spine path resolves to sibling
    Tool: Bash
    Steps:
      1. `python -c "from em_hsd.core.paths import SpinePathResolver; print(SpinePathResolver.resolve())"`
    Expected Result: Path printed if sibling exists; otherwise clear error
    Evidence: .sisyphus/evidence/T18-spine-path.log
  ```

  **Commit:** YES — `feat(core): add spine path resolver for TRIAGE-DP integration`

- [ ] T19. **Build calibration harness skeleton (`em_hsd.calibrate`)**

  **What to do:**
  - Create `em_hsd/calibrate.py` with:
    - `CalibrateRunner` class
    - `search_space()` returns bounds for `epsilon_total`, `hate_floor_delta`, `tau_sem_min`
    - `objective(theta, dev_rows, config)` runs mock pipeline and returns TO proxy
    - `run(...)` orchestrates random search
  - Implement a **mock TO proxy** using `ProxyHateScorer` + synthetic `Author` labels.
    - Generate synthetic `Author` IDs deterministically from row index modulo 4.
    - Compute `Privacy_privatized` as a simple stylometry proxy: 1 - normalized character n-gram overlap between original and privatized text.
    - Compute `Privacy_original` as a baseline using original text vs itself (≈1.0).
    - Compute `Utility_privatized` as average `ProxyHateScorer` score on privatized texts.
    - Compute `Utility_original` as average score on original texts.
    - Return `TO = U_ratio - P_ratio`.
  - Clearly document this is a **mock proxy**, not the real harness.
  - Cache saliency/config-independent results across trials.
  - Export best config to YAML.

  **Must NOT do:**
  - Do not require large downloads.
  - Do not depend on real harness `harness.evaluate` (optional thin wrapper allowed).
  - Do not claim the mock TO equals real TO.

  **Recommended Agent Profile:**
  - **Category:** `deep`

  **Parallelization:**
  - **Can Run In Parallel:** YES
  - **Parallel Group:** Wave 4
  - **Blocks:** T21, T22
  - **Blocked By:** T8, T15

  **References:**
  - `layer-03-to-calibration.md` §6.2 — module structure
  - `layer-03-to-calibration.md` §4.4 — search algorithms
  - `layer-04-only-proposal-v2.md` §6 — user dial levels

  **Acceptance Criteria:**
  - [ ] `python -m em_hsd.calibrate --dev tests/data/synthetic_smoke.csv --config configs/em-hsd-v2-test.yaml --trials 3 --output /tmp/calibrated.yaml` succeeds
  - [ ] Output YAML is loadable by `load_em_hsd_config`

  **QA Scenarios:**
  ```
  Scenario: Calibration runs in mock mode
    Tool: Bash
    Steps:
      1. Create tiny dev CSV with 4 synthetic rows
      2. `python -m em_hsd.calibrate --dev tests/data/synthetic_smoke.csv --config configs/em-hsd-v2-test.yaml --trials 3 --output /tmp/calibrated.yaml`
    Expected Result: rc 0; /tmp/calibrated.yaml exists and is valid YAML
    Evidence: .sisyphus/evidence/T19-calibrate-mock.log
  ```

  **Commit:** YES — `feat(calibrate): add TO calibration harness skeleton`

- [ ] T20. **Build ablation runner supporting A1–A9**

  **What to do:**
  - Create `scripts/run_ablations.py`:
    - A1: No ε₁ (paraphrase raw x)
    - A2: No protected spans
    - A3: No hate floor δ
    - A4: No EM (argmax P_hate)
    - A5: EM-HSD-Naive (Δu=1)
    - A6: Semantic-only EM
    - A7: No τ_dup prune
    - A8: No τ_sem_min
    - A9: Hand prompt vs optimized prompt (placeholder)
  - Each ablation flips one config flag and runs on dev CSV.
  - Output JSON report with per-ablation diagnostics.

  **Must NOT do:**
  - Do not implement real prompt optimization for A9; add placeholder config field.
  - Do not require GPU.

  **Recommended Agent Profile:**
  - **Category:** `unspecified-high`

  **Parallelization:**
  - **Can Run In Parallel:** YES
  - **Parallel Group:** Wave 4
  - **Blocks:** T21, T23
  - **Blocked By:** T8, T15

  **References:**
  - `layer-04-only-proposal-v2.md` §10.2 — ablations A1–A8
  - `layer-04-only-proposal-v2.md` §15.9 — ablation A9

  **Acceptance Criteria:**
  - [ ] `python scripts/run_ablations.py --config configs/em-hsd-v2-test.yaml --output /tmp/ablations.json` succeeds
  - [ ] JSON report contains A1–A9 entries (A9 marked placeholder)

  **QA Scenarios:**
  ```
  Scenario: Ablations run on mock config
    Tool: Bash
    Steps:
      1. `python scripts/run_ablations.py --config configs/em-hsd-v2-test.yaml --output /tmp/ablations.json`
      2. `python -c "import json; d=json.load(open('/tmp/ablations.json')); assert len(d['ablations']) >= 8"`
    Expected Result: rc 0
    Evidence: .sisyphus/evidence/T20-ablations.log
  ```

  **Commit:** YES — `feat(ablations): add A1–A9 ablation runner`

- [ ] T21. **Build evaluation report generator**

  **What to do:**
  - Create `scripts/generate_report.py`:
    - Read ablation JSON and/or calibration JSON
    - Produce Markdown report with tables: TO estimate, utility ratio, privacy ratio, valid-candidate rate, fallback rate, latency
    - Include Burrows' Delta diagnostic stub
    - Add Pareto frontier table placeholder
  - Add `reports/` directory to `.gitignore`.

  **Must NOT do:**
  - Do not download real evaluation data.
  - Do not depend on external harness if not present.

  **Recommended Agent Profile:**
  - **Category:** `writing`

  **Parallelization:**
  - **Can Run In Parallel:** YES
  - **Parallel Group:** Wave 4
  - **Blocks:** T23
  - **Blocked By:** T20

  **References:**
  - `layer-03-to-calibration.md` §7 — calibration report JSON schema
  - `layer-04-only-proposal-v2.md` §8 — audit log schema

  **Acceptance Criteria:**
  - [ ] `python scripts/generate_report.py --ablations /tmp/ablations.json --output .sisyphus/evidence/T21-report.md` succeeds
  - [ ] `.sisyphus/evidence/T21-report.md` contains tables and all expected sections

  **QA Scenarios:**
  ```
  Scenario: Report generator produces markdown
    Tool: Bash
    Steps:
      1. `python scripts/generate_report.py --ablations /tmp/ablations.json --output .sisyphus/evidence/T21-report.md`
      2. `python -c "assert open('.sisyphus/evidence/T21-report.md').read().startswith('#')"`
    Expected Result: rc 0
    Evidence: .sisyphus/evidence/T21-report.log
  ```

  **Commit:** YES — `feat(reports): add evaluation report generator`

- [ ] T22. **Add synthetic-dev smoke evaluation (no model downloads)**

  **What to do:**
  - Add `tests/data/synthetic_smoke.csv` (8–10 rows, mix of hate/benign, long/short).
  - Add `scripts/smoke_eval.py`:
    - Runs mock pipeline on smoke CSV
    - Prints diagnostics: changed rows, avg candidates, fallback rate, valid rate
    - Verifies `ID/Author/HS` preserved
  - Add pytest integration test.

  **Must NOT do:**
  - Do not depend on `Johnny t0-1.03/data/synthetic_dev.csv` unless it exists.
  - Do not require real models.

  **Recommended Agent Profile:**
  - **Category:** `quick`

  **Parallelization:**
  - **Can Run In Parallel:** YES
  - **Parallel Group:** Wave 4
  - **Blocks:** T23
  - **Blocked By:** T8, T17

  **References:**
  - `layer-04-only-proposal-v2.md` §11.2 — engineering order
  - `tests/test_csv_compat.py` — CSV contract tests

  **Acceptance Criteria:**
  - [ ] `python scripts/smoke_eval.py` completes on test config
  - [ ] `pytest tests/test_smoke_eval.py` passes

  **QA Scenarios:**
  ```
  Scenario: Smoke eval runs end-to-end
    Tool: Bash
    Steps:
      1. `python scripts/smoke_eval.py --config configs/em-hsd-v2-test.yaml --in tests/data/synthetic_smoke.csv --out /tmp/smoke_out.csv`
    Expected Result: rc 0; /tmp/smoke_out.csv exists with same row count as input
    Evidence: .sisyphus/evidence/T22-smoke-eval.log
  ```

  **Commit:** YES — `feat(eval): add synthetic-dev smoke evaluation`

- [ ] T23. **Add demo script with `--show-candidates` + research-note outline**

  **What to do:**
  - Create `scripts/demo.py`:
    - Takes one input sentence
    - Prints original, x_priv, candidate list with validity/reject reasons, selected output, audit fields
    - `--show-candidates` flag prints full candidate table
    - `--json` flag outputs audit dict as JSON for programmatic use
    - Works in mock mode by default
  - Create `docs/RESEARCH_NOTE.md` outline.

  **Must NOT do:**
  - Do not write full research note prose yet.
  - Do not require real model for default demo.

  **Recommended Agent Profile:**
  - **Category:** `writing`

  **Parallelization:**
  - **Can Run In Parallel:** YES
  - **Parallel Group:** Wave 4
  - **Blocks:** T28
  - **Blocked By:** T8

  **References:**
  - `layer-04-only-proposal-v2.md` §13 — research note outline
  - `layer-05-rights-architecture.md` §8 — research note outline

  **Acceptance Criteria:**
  - [ ] `python scripts/demo.py --text "you are a dummy" --config configs/em-hsd-v2-test.yaml` runs
  - [ ] `python scripts/demo.py --text "you are a dummy" --config configs/em-hsd-v2-test.yaml --json > /tmp/demo.json` and `python -c "import json; json.load(open('/tmp/demo.json'))"` succeeds
  - [ ] `docs/RESEARCH_NOTE.md` exists with all sections

  **QA Scenarios:**
  ```
  Scenario: Demo script runs
    Tool: Bash
    Steps:
      1. `python scripts/demo.py --text "you are a dummy" --config configs/em-hsd-v2-test.yaml`
    Expected Result: rc 0; output contains "original", "selected", "fallback"
    Evidence: .sisyphus/evidence/T23-demo.log

  Scenario: Demo script outputs JSON
    Tool: Bash
    Steps:
      1. `python scripts/demo.py --text "you are a dummy" --config configs/em-hsd-v2-test.yaml --json > /tmp/demo.json`
      2. `python -c "import json; json.load(open('/tmp/demo.json'))"`
    Expected Result: rc 0
    Evidence: .sisyphus/evidence/T23-demo-json.log
  ```

  **Commit:** YES — `feat(docs): add demo script and research-note outline`

- [ ] T24. **Add no-download regression test**

  **What to do:**
  - Add `tests/test_no_downloads.py`.
  - Patches `transformers.AutoModel*.from_pretrained` and `unsloth.FastLanguageModel.from_pretrained` to fail if called.
  - Runs mock pipeline and asserts no download function called.
  - Verifies `DownloadPolicy` defaults to False.

  **Must NOT do:**
  - Do not actually download anything.

  **Recommended Agent Profile:**
  - **Category:** `quick`

  **Parallelization:**
  - **Can Run In Parallel:** YES
  - **Parallel Group:** Wave 4
  - **Blocks:** T30
  - **Blocked By:** T0, T9

  **References:**
  - `em_hsd.core.policy`
  - `em_hsd.core.resources`

  **Acceptance Criteria:**
  - [ ] `pytest tests/test_no_downloads.py` passes

  **QA Scenarios:**
  ```
  Scenario: No-download regression test passes
    Tool: pytest
    Steps:
      1. `pytest tests/test_no_downloads.py -v`
    Expected Result: PASS
    Evidence: .sisyphus/evidence/T24-no-downloads-regression.log
  ```

  **Commit:** YES — `test(policy): add no-download regression test`

- [ ] T25. **Add harness integration stub (works when Johnny harness available)**

  **What to do:**
  - Create `em_hsd/harness_integration.py`:
    - Function `evaluate_with_harness(original_csv, privatized_csv, config)`
    - Tries to import `harness.evaluate` from `Johnny t0-1.03/src`
    - If unavailable, returns a structured dict with error + suggestion
    - If available, calls harness and returns report dict
  - Add `--evaluate` flag to `smoke_eval.py` that uses this stub.

  **Must NOT do:**
  - Do not require harness to be present.
  - Do not import harness inside core/layer4 modules.

  **Recommended Agent Profile:**
  - **Category:** `quick`

  **Parallelization:**
  - **Can Run In Parallel:** YES
  - **Parallel Group:** Wave 4
  - **Blocks:** T28
  - **Blocked By:** T18

  **References:**
  - `layer-04-only-proposal-v2.md` §11.2 — functional validation
  - `Johnny t0-1.03/src/harness/evaluate.py`

  **Acceptance Criteria:**
  - [ ] `python scripts/smoke_eval.py --evaluate` runs and reports gracefully when harness missing
  - [ ] `evaluate_with_harness` can be imported without importing harness eagerly

  **QA Scenarios:**
  ```
  Scenario: Harness stub degrades gracefully
    Tool: Bash
    Steps:
      1. `python -c "from em_hsd.harness_integration import evaluate_with_harness; print(evaluate_with_harness('a.csv','b.csv',None))"`
    Expected Result: Returns dict without crashing
    Evidence: .sisyphus/evidence/T25-harness-stub.log
  ```

  **Commit:** YES — `feat(harness): add optional harness integration stub`

- [ ] T26. **Update all configs to dual-mode schema (keep old ones valid)**

  **What to do:**
  - Add `triage_dp` section to all 6 configs with `enabled: false`.
  - Create `configs/em-hsd-v2-triage-dp-test.yaml` with `enabled: true` and mock Layer 1–3.
  - Add `tests/test_configs.py` validating every config.

  **Must NOT do:**
  - Do not change existing runtime behavior.

  **Recommended Agent Profile:**
  - **Category:** `quick`

  **Parallelization:**
  - **Can Run In Parallel:** YES
  - **Parallel Group:** Wave 5
  - **Blocks:** T30
  - **Blocked By:** T4, T17

  **References:**
  - `src/em_hsd/config.py`
  - `layer-03-to-calibration.md` §6.1 — calibrated config example

  **Acceptance Criteria:**
  - [ ] All existing configs still parse
  - [ ] New triage-dp test config parses
  - [ ] `pytest tests/test_configs.py` passes

  **QA Scenarios:**
  ```
  Scenario: All configs load
    Tool: pytest
    Steps:
      1. `pytest tests/test_configs.py -v`
    Expected Result: PASS
    Evidence: .sisyphus/evidence/T26-configs.log
  ```

  **Commit:** YES — `feat(configs): add triage_dp sections and test config`

- [ ] T27. **Write migration guide from old import paths**

  **What to do:**
  - Create `docs/MIGRATION.md` with old → new import mapping table, package layout diagram, direct `em_hsd.layer4` usage, `em_hsd.interfaces.triage_dp` usage, and deprecation timeline.

  **Must NOT do:**
  - Do not delete old paths yet.

  **Recommended Agent Profile:**
  - **Category:** `writing`

  **Parallelization:**
  - **Can Run In Parallel:** YES
  - **Parallel Group:** Wave 5
  - **Blocks:** T28
  - **Blocked By:** T2, T17

  **References:**
  - Current and new import paths

  **Acceptance Criteria:**
  - [ ] `docs/MIGRATION.md` exists with all mappings

  **QA Scenarios:**
  ```
  Scenario: Migration doc covers all paths
    Tool: Bash
    Steps:
      1. `python -c "import re; text=open('docs/MIGRATION.md').read(); assert 'em_hsd.csv_compat' in text and 'em_hsd.io.csv_io' in text"`
    Expected Result: No assertion error
    Evidence: .sisyphus/evidence/T27-migration.log
  ```

  **Commit:** YES — `docs(migration): add import-path migration guide`

- [ ] T28. **Finalize README / quickstart / integration status**

  **What to do:**
  - Create/update `README.md` with quickstart (mock mode), install, CLI usage, TRIAGE-DP integration status, links to docs.

  **Must NOT do:**
  - Do not invent performance claims.

  **Recommended Agent Profile:**
  - **Category:** `writing`

  **Parallelization:**
  - **Can Run In Parallel:** YES
  - **Parallel Group:** Wave 5
  - **Blocks:** T30
  - **Blocked By:** T23, T27

  **References:**
  - `layer-04-only-proposal-v2.md` §11.3
  - `layer-05-rights-architecture.md` §6

  **Acceptance Criteria:**
  - [ ] README quickstart runs in mock mode

  **QA Scenarios:**
  ```
  Scenario: README quickstart is runnable
    Tool: Bash
    Steps:
      1. Copy quickstart command from README
      2. Run it on tests/data/synthetic_smoke.csv
    Expected Result: rc 0
    Evidence: .sisyphus/evidence/T28-readme.log
  ```

  **Commit:** YES — `docs(readme): finalize README with quickstart and integration status`

- [ ] T29. **Document fallback policy and privacy-claim boundaries**

  **What to do:**
  - Add `docs/PRIVACY_CLAIMS.md` documenting:
    - What is claimed: ε₁-DP token sanitize on non-protected positions; ε₂-DP EM selection
    - What is not claimed: DP on generation, document-level DP, normalization/canonicalization formal ε
    - Fallback returns `x_priv` preserving ε₁
  - Add `docs/FALLBACK_POLICY.md` with decision tree.

  **Must NOT do:**
  - Do not over-promise privacy.

  **Recommended Agent Profile:**
  - **Category:** `writing`

  **Parallelization:**
  - **Can Run In Parallel:** YES
  - **Parallel Group:** Wave 5
  - **Blocks:** T30
  - **Blocked By:** T8

  **References:**
  - `layer-04-only-proposal-v2.md` §7, §9 — privacy claim
  - `layer-05-rights-architecture.md` §3.5 — honest privacy claims

  **Acceptance Criteria:**
  - [ ] Both docs exist and accurately reflect code behavior

  **QA Scenarios:**
  ```
  Scenario: Privacy claims doc exists
    Tool: Bash
    Steps:
      1. `python -c "assert 'epsilon-DP' in open('docs/PRIVACY_CLAIMS.md').read()"`
    Expected Result: No assertion error
    Evidence: .sisyphus/evidence/T29-privacy-claims.log
  ```

  **Commit:** YES — `docs(privacy): document fallback policy and privacy-claim boundaries`

- [ ] T30. **Pre-commit quality gate: tests, lint, typecheck, no-download check**

  **What to do:**
  - Ensure `pytest tests/` passes.
  - Add `ruff` to `[project.optional-dependencies] test` (or new `dev` extra).
  - Add `mypy` to `test` extra.
  - Update `pyproject.toml` with `[tool.mypy]` and `[tool.ruff]` minimal config.
  - Add `scripts/quality_gate.py` local task runner.
  - Gate runs:
    - `pytest tests/`
    - `ruff check src/em_hsd scripts`
    - `mypy src/em_hsd` (or `--ignore-missing-imports`)
    - `python scripts/check_no_downloads.py`

  **Must NOT do:**
  - Do not fix unrelated lint issues without documenting.

  **Recommended Agent Profile:**
  - **Category:** `quick`

  **Parallelization:**
  - **Can Run In Parallel:** NO — final gate
  - **Parallel Group:** Wave 5 (finalizer)
  - **Blocks:** T31
  - **Blocked By:** T0, T24, T26–T29

  **References:**
  - `pyproject.toml`
  - `tests/conftest.py`

  **Acceptance Criteria:**
  - [ ] `pytest tests/` PASS
  - [ ] Lint passes on public API files
  - [ ] No-download check passes
  - [ ] Gate command documented

  **QA Scenarios:**
  ```
  Scenario: Quality gate passes
    Tool: Bash
    Steps:
      1. `python scripts/quality_gate.py`
    Expected Result: rc 0
    Evidence: .sisyphus/evidence/T30-quality-gate.log
  ```

  **Commit:** YES — `chore(quality): add pre-commit quality gate`

- [ ] T31. **Final end-to-end mock smoke run + evidence capture**

  **What to do:**
  - Run full mock pipeline on synthetic smoke CSV.
  - Run ablations on smoke CSV.
  - Generate report.
  - Run demo script.
  - Capture all evidence files.
  - Ensure no-download guard active.

  **Must NOT do:**
  - Do not download models.

  **Recommended Agent Profile:**
  - **Category:** `unspecified-high`

  **Parallelization:**
  - **Can Run In Parallel:** NO — final capstone
  - **Parallel Group:** Wave 5 (capstone)
  - **Blocks:** F1–F4
  - **Blocked By:** T19–T30

  **References:**
  - All previous tasks

  **Acceptance Criteria:**
  - [ ] All commands in verification section produce expected output
  - [ ] Evidence directory contains one file per task QA scenario
  - [ ] No model downloads occurred

  **QA Scenarios:**
  ```
  Scenario: Full mock end-to-end
    Tool: Bash
    Steps:
      1. `python scripts/smoke_eval.py --config configs/em-hsd-v2-test.yaml --in tests/data/synthetic_smoke.csv --out .sisyphus/evidence/final_out.csv`
      2. `python scripts/run_ablations.py --config configs/em-hsd-v2-test.yaml --output .sisyphus/evidence/final_ablations.json`
      3. `python scripts/generate_report.py --ablations .sisyphus/evidence/final_ablations.json --output .sisyphus/evidence/final_report.md`
      4. `python scripts/demo.py --text "you are a dummy" --config configs/em-hsd-v2-test.yaml`
    Expected Result: All rc 0
    Evidence: .sisyphus/evidence/T31-final-smoke.log
  ```

  **Commit:** YES — `test(e2e): final mock end-to-end smoke run`

---

## Commit Strategy

- One commit per task (T0–T31), with concise `type(scope): description` messages.
- Final review wave F1–F4 does not produce code commits; it produces review artifacts in `.sisyphus/evidence/`.
- No empty commits.
- Pre-commit per task: `pytest tests/`

## Success Criteria

### Verification Commands
```bash
python scripts/quality_gate.py                                                                 # Expected: rc 0
pytest tests/                                                                                   # Expected: PASS
python -m em_hsd.cli.run --in tests/data/synthetic_smoke.csv --out .sisyphus/evidence/out.csv --config configs/em-hsd-v2-test.yaml --mode em-hsd-v2  # Expected: rc 0
python -m em_hsd.cli.run --in tests/data/synthetic_smoke.csv --out .sisyphus/evidence/out2.csv --config configs/em-hsd-v2-test.yaml --mode triage-dp  # Expected: rc 0
python scripts/run_ablations.py --config configs/em-hsd-v2-test.yaml --output .sisyphus/evidence/ablations.json  # Expected: rc 0
python scripts/generate_report.py --ablations .sisyphus/evidence/ablations.json --output .sisyphus/evidence/report.md        # Expected: rc 0
python scripts/demo.py --text "you are a dummy" --config configs/em-hsd-v2-test.yaml             # Expected: rc 0
python scripts/demo.py --text "you are a dummy" --config configs/em-hsd-v2-test.yaml --json     # Expected: valid JSON, rc 0
```

### Final Checklist
- [ ] All Must Have present
- [ ] All Must NOT Have absent
- [ ] All tests pass
- [ ] No big file downloads during work
- [ ] Old import paths still work
- [ ] New TRIAGE-DP adapter imports cleanly
- [ ] Plan passes F1–F4

---

## Final Verification Wave

### F1. Plan Compliance Audit — `oracle`
Read plan end-to-end. For each Must Have verify implementation exists; for each Must NOT Have search codebase and reject with file:line if found.
Output: `Must Have [N/N] | Must NOT Have [N/N] | Tasks [N/N] | VERDICT: APPROVE/REJECT`

### F2. Code Quality Review — `unspecified-high`
Run `pytest`, linter, typecheck. Review changed files for anti-patterns.
Output: `Build [PASS/FAIL] | Lint [PASS/FAIL] | Tests [N pass/N fail] | Files [N clean/N issues] | VERDICT`

### F3. Real Manual QA — `unspecified-high`
Note: "Manual" here means agent-executed, not human. Execute every QA scenario from every task. Test cross-task integration and edge cases. Save evidence to `.sisyphus/evidence/final-qa/`.
Output: `Scenarios [N/N pass] | Integration [N/N] | Edge Cases [N tested] | VERDICT`

### F4. Scope Fidelity Check — `deep`
For each task: read "What to do", read actual diff. Verify 1:1 spec-to-code, no scope creep.
Output: `Tasks [N/N compliant] | Contamination [CLEAN/N issues] | Unaccounted [CLEAN/N files] | VERDICT`