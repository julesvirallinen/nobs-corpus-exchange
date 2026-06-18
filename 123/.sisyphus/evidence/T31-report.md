# EM-HSD 2.0 Evaluation Report
**Config:** `configs/em-hsd-v2-test.yaml`  
**Input:** `tests/data/synthetic_smoke.csv`  

## Baseline

- Rows: 10
- Changed: 10 (100.00%)
- Fallback: 1 (10.00%)
- Avg valid candidates: 0.90

## Ablations

| Code | Description | Rows | Changed | Fallback | Avg k_valid |
|---|---|---:|---:|---:|---:|
| A1 | No epsilon_1 (paraphrase raw x) | 10 | 100.00% | 0.00% | 1.00 |
| A2 | No protected spans | 10 | 100.00% | 10.00% | 0.90 |
| A3 | No hate floor delta | 10 | 100.00% | 10.00% | 0.90 |
| A4 | No EM (argmax P_hate) | 10 | 100.00% | 10.00% | 0.90 |
| A5 | EM-HSD-Naive (delta_u=1) | 10 | 100.00% | 10.00% | 0.90 |
| A6 | Semantic-only EM | 10 | 100.00% | 50.00% | 0.50 |
| A7 | No tau_dup prune | 10 | 100.00% | 10.00% | 3.60 |
| A8 | No tau_sem_min | 10 | 100.00% | 10.00% | 0.90 |
| A9 | Hand vs optimized prompt (placeholder) | 10 | 100.00% | 10.00% | 0.90 |

## Calibration

- epsilon_total: 12.3006
- hate_floor_delta: 0.3779
- tau_sem_min: 0.5947
- Best TO (mock): 0.3568
- U ratio: 0.9059
- P ratio: 0.5491
- Fallback rate: 10.00%

## Burrows' Delta diagnostic stub

*Requires character n-gram profiles across the full corpus. Placeholder: re-identification risk is not computed in mock mode.*

## Pareto frontier placeholder

| Utility ratio | Privacy ratio | TO | Config hint |
|---|---|---|---|
| *mock* | *mock* | *mock* | run calibration with real harness |

