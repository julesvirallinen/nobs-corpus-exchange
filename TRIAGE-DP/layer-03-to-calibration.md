# Layer 3 — TO-Calibrated Budget Optimizer

Detailed design document for **TRIAGE-DP**. See the overview in [`../TRIAGE-DP.md`](../TRIAGE-DP.md). Depends on Layer 1 routing ([`layer-01-cross-saliency-triage.md`](layer-01-cross-saliency-triage.md)) and optionally Layer 2 priors ([`layer-02-stylometric-priors.md`](layer-02-stylometric-priors.md)).

---

## 1. Purpose and role in the stack

Organizers note that dpmlm is **grounded in privacy theory and parameterized—but introduces complexity**. Layer 3 answers that criticism operationally:

> **Human operators choose a privacy level; the system chooses θ.**

Layer 3 searches for a parameter vector **θ** that maximizes the hackathon **Trade-Off (TO)** metric on a development split, then exports:

1. Frozen config for submission/production
2. Pareto frontier (utility vs privacy)
3. Mapping from user-facing dial (1–5) to θ profiles

Layer 3 runs **offline** during development/calibration. It is **not** invoked per row at submission time (unless adaptive online calibration is explicitly enabled—a research extension).

---

## 2. The TO objective (formal)

From PrivHSD / `harness/tradeoff.py`:

```
TO(θ) = U_ratio(θ) − P_ratio(θ)

where:
  U_ratio(θ) = Utility_privatized(θ) / Utility_original
  P_ratio(θ) = Privacy_privatized(θ) / Privacy_original
```

| Component | Harness definition | Direction |
|-----------|-------------------|-----------|
| `Utility_*` | Ensemble hate-classification **macro-F1** | Higher better |
| `Privacy_*` | Authorship re-ID attacker **top-1 accuracy** | Lower better after privatization |

**Goal:** maximize TO(θ) subject to constraints (minimum utility floor, maximum ε spend, etc.).

### 2.1 Local vs official score

The harness explicitly labels TO as a **local approximation** of the organizer's hidden evaluator:

```
*** LOCAL APPROXIMATION (not the official score) ***
```

Layer 3 must:

- Use dev split **not identical** to repeated tuning set if risk of overfitting
- Hold out a **calibration-test** fold never used in search
- Report sensitivity to probe choice (proxy vs HF backends)
- Never claim optimized θ guarantees official leaderboard rank

---

## 3. Parameter vector θ

### 3.1 Full parameter list

```yaml
θ:
  # Layer 1 thresholds
  tau_H: 0.12              # hate saliency quadrant boundary
  tau_A: 0.18              # authorship saliency quadrant boundary

  # Per-quadrant DP budgets (exponential mechanism ε)
  epsilon_Q2: 8.0          # aggressive privacy (lower ε = more noise if inverted—see §3.2)
  epsilon_Q1_context: 25.0 # optional neighbor context rewrite
  epsilon_sentence: 15.0 # Layer 4 sentence-level EM

  # Layer 2 boosts (optional learnable scale)
  biber_scale: 1.0         # multiplies all β_f in biber.boosts

  # MLM / DP mechanics (usually fixed, optionally tuned)
  mlm_top_k: 48
  mlm_clip: 5.0

  # Gates
  hard_row_min_tokens: 40
  stretch_enabled: true

  # User dial preset id
  privacy_level: 3         # 1=utility-first .. 5=privacy-first
```

### 3.2 ε semantics (critical)

In DP-MLM / `mechanism/dp.py`:

```
P(i) ∝ exp(ε · u_i / (2Δ))
```

**Higher ε** → distribution concentrates on high-utility candidate → **less noise** → **weaker per-token privacy, higher utility**.

| Quadrant | Typical ε range | Intent |
|----------|-----------------|--------|
| Q2 (aggressive privacy) | **Lower** ε (e.g. 3–15) | More stochastic rewrite |
| Q1 context | **Medium** ε (e.g. 15–40) | Light touch near conflict tokens |
| Q3/Q4 | `skip` | No ε spent |

Layer 3 search bounds must respect this direction to avoid inverted optimization.

### 3.3 Fixed vs tunable parameters

| Parameter | Tune in Layer 3? | Rationale |
|-----------|------------------|-----------|
| `tau_H`, `tau_A` | **Yes** | Core routing |
| `epsilon_Q2` | **Yes** | Primary privacy dial |
| `epsilon_Q1_context` | Optional | Secondary |
| `biber_scale` | Optional | Layer 2 strength |
| `mlm_clip`, `top_k` | Rarely | DP theory stable defaults |
| Model checkpoints | **No** | Keep pinned for reproducibility |

---

## 4. Optimization procedure

### 4.1 Data splits

```
D_full (dev.csv from organizers or proxy)
├── D_calib (60%)  — used in search loop
├── D_select (20%) — select best θ each generation
└── D_holdout (20%) — report once at end; never tune on
```

If only one dev file: **k-fold cross-validation** on D_calib; holdout carved first.

**Author column:** used **only** in harness during evaluation, never passed to `privatize()`.

### 4.2 Objective function

```python
def objective(theta, D_eval, config_base) -> dict:
    config = apply_theta(config_base, theta)
    privatized = [privatize(row.Text, config) for row in D_eval]
    report = harness.evaluate(original=D_eval, privatized=privatized, ...)
    return {
        "TO": report.trade_off.trade_off_estimate,
        "U_ratio": report.trade_off.utility_ratio,
        "P_ratio": report.trade_off.privacy_ratio,
        "utility_f1": report.utility.utility_privatized,
        "privacy_acc": report.reidentification.privacy_privatized,
    }
```

### 4.3 Constraints (recommended)

```
Utility_privatized >= 0.85 * Utility_original   # floor
Privacy_privatized <= 0.70 * Privacy_original   # minimum privacy gain
epsilon_Q2 >= 2.0                                # don't collapse to noise-only broken text
```

Constrained optimization: penalize violations:

```
J(θ) = TO(θ) − λ_u · max(0, U_floor − U_priv) − λ_p · max(0, P_priv − P_ceiling)
```

### 4.4 Search algorithms

| Method | Pros | Cons | When to use |
|--------|------|------|-------------|
| **Grid search** | Simple, reproducible | Curse of dimensionality | 2–3 params only |
| **Random search** | Good for sparse gains | Sample inefficient | Initial exploration |
| **Bayesian optimization** (Optuna, skopt) | Sample efficient | Black-box | **Default recommendation** |
| **CMA-ES** | Continuous θ | Needs bounds | Fine-tune after BO |
| **Pareto MOO** (NSGA-II) | Multi-objective frontier | Harder to pick single θ | Research plots |

**Recommended pipeline:**

1. Random 30 trials — wide bounds
2. Optuna 100 trials — TPE sampler on promising region
3. Pareto extract — top 5 θ by TO + non-dominated (U, 1/P)
4. Holdout evaluate once

### 4.5 Cost model

Each trial = full pass privatize + evaluate on D_select.

| Dataset size | Trials | Approx cost |
|--------------|--------|-------------|
| 500 rows, L1 occlusion | 100 | Hours–days CPU |
| 500 rows, L1 cached saliency | 100 | Faster if saliency cached per θ-independent pass |

**Optimization:** For θ that only changes ε and τ (not occlusion paths), cache H(t) and A(t) per row **once**, reuse across trials—massive speedup.

---

## 5. User-facing privacy dial

### 5.1 Mapping privacy levels to θ

Pre-compute 5 profiles on Pareto frontier:

| Level | Label | Target | Typical θ profile |
|-------|-------|--------|-------------------|
| 1 | Utility-first | Max U_ratio | High ε_Q2, high τ_H (few Q3), low stretch |
| 2 | Balanced-light | TO high, slight privacy | Medium ε_Q2 |
| 3 | **Default** | Max TO on holdout | Optimized θ* |
| 4 | Privacy-strong | Low P_ratio | Low ε_Q2, low τ_A |
| 5 | Privacy-max | Min P_ratio subject to U floor | Lowest ε_Q2 + stretch on |

Submission config points to `privacy_level: 3` unless judges request sensitivity analysis.

### 5.2 Config export

```yaml
# configs/triage-dp-calibrated.yaml
# Generated by: python -m harness.calibrate --dev dev.csv ...
calibration:
  date: "2026-06-15"
  holdout_TO: 0.52
  holdout_U_ratio: 0.94
  holdout_P_ratio: 0.42
  n_trials: 130
  probe_backend: hf

privacy_level: 3

triage:
  tau_H: 0.11
  tau_A: 0.19

epsilon:
  Q2: 6.5
  Q1_context: 30.0
  sentence: 18.0

biber:
  scale: 0.9
```

---

## 6. Proposed implementation: `harness/calibrate.py`

### 6.1 CLI interface

```bash
python -m harness.calibrate \
  --dev data/dev.csv \
  --config configs/triage-dp.yaml \
  --output configs/triage-dp-calibrated.yaml \
  --trials 100 \
  --holdout-frac 0.2 \
  --utility-backend hf \
  --cache-saliency saliency_cache.npz \
  --json reports/calibration_run.json
```

### 6.2 Module structure

```
harness/calibrate.py
├── load_dev_csv()
├── split_holdout()
├── build_search_space() -> optuna distributions
├── apply_theta(config, theta) -> Config
├── objective(trial) -> float
├── extract_pareto_front(results) -> list[theta]
├── export_profiles(front, levels=5) -> yaml
└── main()
```

### 6.3 Saliency caching

Occlusion H(t), A(t) depend on text and probes, **not** on ε.

```
Phase A (once): for each row, compute and store H, A per token
Phase B (many trials): triage with τ from θ; DP with ε from θ; evaluate
```

Cache invalidation: only when text normalization rules or probe models change.

---

## 7. Outputs and artifacts

### 7.1 Calibration report (`calibration_run.json`)

```json
{
  "best_theta": { "tau_H": 0.11, "epsilon_Q2": 6.5, ... },
  "holdout": { "TO": 0.52, "U_ratio": 0.94, "P_ratio": 0.42 },
  "calib_best_TO": 0.55,
  "pareto_points": [ ... ],
  "trials": [ { "theta": {}, "TO": 0.48, "trial_id": 1 }, ... ]
}
```

### 7.2 Figures for research note

1. **TO vs trial** — convergence plot
2. **Pareto frontier** — U_ratio vs P_ratio colored by TO
3. **τ_H × τ_A heatmap** — TO on grid (if 2D slice)
4. **ε_Q2 sensitivity** — TO vs ε curve
5. **Privacy dial table** — levels 1–5 metrics

### 7.3 Ablation integration

Run calibration with flags:

| Flag | Effect |
|------|--------|
| `--no-biber` | Layer 2 off |
| `--no-stretch` | Layer 4 off |
| `--fixed-tau` | Only tune ε |
| `--spine-mode` | Baseline SPINE parameterization |

Compare holdout TO to justify each layer.

---

## 8. Interaction with Layers 1, 2, 4

```
                    ┌─────────────────┐
                    │  Layer 3 (θ)    │
                    │  tau_H, tau_A   │
                    │  epsilon_Q*     │
                    │  biber_scale    │
                    └────────┬────────┘
                             │
         ┌───────────────────┼───────────────────┐
         ▼                   ▼                   ▼
   Layer 1 triage      Layer 2 boosts     Layer 4 gates
   (thresholds)        (scaled β)         (epsilon_sentence,
                                         hard_row_min_tokens)
```

Layer 3 does **not** retrain hate or authorship probes during search (fixed probes = stable objective). Probe training is a separate offline step.

---

## 9. Avoiding overfitting

| Risk | Mitigation |
|------|------------|
| Tune to local SVM attacker | Report HF transformer probe on holdout |
| Tune to proxy hate classifier | `--utility-backend hf` for final θ |
| Too many trials on small dev | Holdout + cross-validation SE |
| θ exploits harness bugs | Manual review of worst/best rows |
| Official evaluator differs | Document as limitation; qualitative examples |

**Rule:** Report **holdout TO** in research note, not calib-best TO.

---

## 10. Worked calibration example (synthetic)

Assume calib set 200 rows, 10 authors.

| Trial | τ_H | τ_A | ε_Q2 | TO | U_ratio | P_ratio |
|-------|-----|-----|------|-----|---------|---------|
| baseline dpmlm | — | — | uniform 25 | 0.31 | 0.78 | 0.47 |
| SPINE | — | — | content 25 | 0.38 | 0.91 | 0.53 |
| T42 | 0.15 | 0.20 | 8 | 0.49 | 0.92 | 0.43 |
| T87 | 0.11 | 0.19 | 6.5 | **0.55** | 0.94 | 0.41 |
| T99 | 0.08 | 0.15 | 4 | 0.51 | 0.81 | 0.35 |

Pick T87 if holdout confirms; T99 wins privacy but violates U floor.

---

## 11. Comparison to related work

| Approach | Similarity | Difference |
|----------|------------|------------|
| dpmlm manual ε | Single dial | Layer 3 multi-dimensional θ + auto search |
| STAMP group profiles | Per-group ε | θ includes triage thresholds + TO objective |
| Loiseau GEPA prompts | Optimizes anonymization | Layer 3 optimizes formal DP parameters, not LLM prompts |
| DP-MGTD adaptive budget | Frequency-based entity budgets | Layer 3 optimizes occlusion + stylometric routing for IA-HSD |

---

## 12. Testing strategy

### 12.1 Unit tests

- `apply_theta` round-trip YAML
- Objective returns finite TO on synthetic_dev.csv
- Holdout rows never appear in trial logs for calib split
- ε bounds enforced (Q2 ≥ ε_min)

### 12.2 Integration tests

- End-to-end calibrate with 5 trials on synthetic data
- Cached saliency matches uncached TO within tolerance

---

## 13. Limitations

- Calibration data may not match test distribution
- TO is ratio-based— unstable when `Privacy_original` ≈ 0 (few authors)
- Multi-objective tension: single θ cannot dominate entire Pareto front
- Computationally expensive with full Layer 1 occlusion per trial unless cached

---

## 14. References

- `Johnny t0-1.03/src/harness/tradeoff.py` — TO definition
- `Johnny t0-1.03/src/harness/evaluate.py` — evaluation pipeline
- `resources/Trade-off.png` — organizer metric
- Meisenbacher et al., DP-MLM — ε semantics
- Wang et al., DP-MGTD — adaptive budget motivation

---

## 15. Open design decisions

| Decision | Recommendation |
|----------|----------------|
| Primary optimizer | Optuna with 100 trials |
| Cache saliency | Yes, default on |
| Tune τ and ε jointly | Yes |
| Multi-objective pick | Max TO on holdout with U floor |
| Recalibrate at submission | Once; freeze θ in repo |
