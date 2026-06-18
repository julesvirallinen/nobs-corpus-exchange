# Fallback Policy

## Purpose

The fallback policy defines what `em-hsd-v2` does when the candidate-generation and selection pipeline cannot produce a valid candidate. It exists so that the module always returns a safe, deterministic output rather than leaking information through an error or an empty response.

## Trigger conditions

A fallback occurs when any of the following is true after candidate generation:

1. **Zero candidates generated** — the proposer failed or the input is too short.
2. **All candidates filtered out** — every candidate violates at least one validity rule.
3. **Exponential-mechanism failure** — the score distribution is degenerate or numerically unstable.
4. **Runtime error** during scoring or selection.

## Fallback output

When a fallback is triggered, `Layer4Orchestrator.privatize()` returns:

```python
{
    "x_priv": <canonicalised input with protected tokens retained>,
    "fallback": True,
    "fallback_reason": <one of the reason strings below>,
    "token_log": [...],        # full per-token decision log
    "candidates": [],          # or populated with filter details
    "k_valid": 0,
    "P_hate_x_priv": <proxy hate score>,
}
```

## Fallback reason codes

| Reason | Meaning |
|--------|---------|
| `no_candidates_generated` | The proposer produced zero candidates. |
| `no_valid_candidates` | Candidates were generated but all were filtered. |
| `selection_failed` | The exponential mechanism could not sample. |
| `runtime_error` | An unexpected exception occurred. |

## Why this is safe

The fallback output is the **canonicalised protected baseline**. It contains only tokens that were already tagged as protected or retained by deterministic rules, and does not depend on the private candidate set. Therefore, releasing it does not consume additional privacy budget beyond what was already spent to produce the baseline.

## Configuration

Fallback behavior is controlled by the `pipeline` section of the config:

```yaml
pipeline:
  protected_override: "dummy"      # replacement for protected tokens (deterministic)
  epsilon_1: 9.0
  epsilon_2: 9.0
  delta_u: 0.5
  min_semantic_score: 0.6
  max_hate_score: 0.9
  min_length: 3
```

`protected_override` is applied deterministically when a token is in the hate lexicon. If the original token is already canonical, the token is kept as-is.

## Audit fields

Every fallback response includes:

- `fallback: true`
- `fallback_reason`
- `k_generated`
- `k_after_prune`
- `k_valid`
- `filter_details` for each rejected candidate

These fields allow downstream auditors to determine why no candidate was selected.

## Future improvements

1. Add a **partial-fallback** mode that returns the best-scoring invalid candidate with a warning, when utility is critical.
2. Make fallback reason codes stable and localisable.
3. Expose fallback statistics in the evaluation report.

