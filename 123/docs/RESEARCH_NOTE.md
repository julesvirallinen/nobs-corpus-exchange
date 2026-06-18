# Research Note: Layer-4 DP Text Sanitization for `em-hsd-v2`

## 1. Goal
Harded the standalone `em-hsd-v2` module into a TRIAGE-DP-compatible **Layer-4 differential-privacy text sanitiser** that can later be merged into the full TRIAGE-DP harness without losing its current independent test/run capability.

## 2. What was changed
- Refactored the flat `em_hsd` package into `core/`, `layer4/`, `interfaces/`, `cli/`, and `io/`.
- Defined clean protocols for **Layer 1 (Cross-Saliency Triage)**, **Layer 2 (Stylometric Prior)**, and **Layer 3 (TO Calibration)**.
- Built a `Layer4Orchestrator` that consumes those three inputs and returns `x_priv` plus a full audit token log.
- Added `TriageDPLayer4`, an adapter that exposes a single `sanitize()` entry point for the future TRIAGE-DP harness.
- Centralised model/encoder/proposer loading behind `ResourceManager` with a hard **no-download** policy unless `EM_HSD_ALLOW_DOWNLOADS=1`.
- Replaced hard-coded paths to the `Johnny t0-1.03` SPINE code with `SpinePathResolver` (`EM_HSD_SPINE_PATH` env / config).
- Added a calibration harness, ablation runner, smoke evaluator, report generator, and demo script — all running on mock/proxy models so nothing heavy is downloaded.
- Preserved all old public imports via shims in the original package root.

## 3. Why these changes
- **Separation of concerns**: Layer 4 should not know how Layers 1–3 are implemented, only what they output.
- **Merge readiness**: The full TRIAGE-DP pipeline will supply real Layer 1–3 components; `TriageDPLayer4` is the plug point.
- **No-download safety**: The repository can be cloned, tested, and demonstrated on a clean machine without accidentally pulling multi-gigabyte models from HuggingFace or Unsloth.
- **Reproducibility**: Every script outputs structured JSON/CSV/Markdown and writes evidence to `.sisyphus/evidence/`.

## 4. Open research questions
1. Does the current EM mechanism (Exponential over semantic-similarity × hate-score penalty) preserve utility under real hate-classifiers?
2. How should Layer 2 stylometric priors be represented when tokens are replaced by paraphrases rather than single words?
3. What is the right utility proxy when no production model is available locally?
4. How do we calibrate `epsilon_1`, `epsilon_2`, `delta_u` jointly with Layer 3 rather than independently?
5. What guarantees does the fallback policy (`x_priv` returned when candidates are invalid) provide, and how should it be audited?

## 5. Evidence captured
- Full test suite: `.sisyphus/evidence/T22-full-suite-after-harness-fix.log`
- Smoke evaluation: `.sisyphus/evidence/T22-smoke-eval.csv`
- Ablation run: `.sisyphus/evidence/T20-ablations.json`
- Calibration run: `.sisyphus/evidence/T19-calibration*.yaml`
- Demo output: `.sisyphus/evidence/T23-demo-utf8.json`
- No-download gate: `.sisyphus/evidence/T24-no-downloads.log`

## 6. Next research steps
1. Swap the mock utility/hate proxies for small local models (e.g., DistilBERT hate detection) once download policy is explicitly enabled.
2. Integrate real Layer 1–3 modules from the full TRIAGE-DP harness and run the same test matrix.
3. Add formal DP accounting and confidence intervals around the reported `epsilon_total`.
4. Evaluate the fallback-to-`x_priv` policy against a held-out adversarial test set.

