# Privacy Claims for `em-hsd-v2`

## What this module actually does

`em-hsd-v2` is a **Layer-4 differential-privacy text sanitiser**. Given an input sentence, it generates a set of candidate paraphrases, scores them for semantic utility and hate-content risk, and selects one via the **exponential mechanism** with privacy budget `epsilon_total = epsilon_1 + epsilon_2`.

## Claims

### 1. DP selection over candidates
The final candidate is selected with the exponential mechanism using a score function that combines semantic similarity to the original text and a hate-score penalty. Under the standard exponential-mechanism analysis, this step satisfies **(epsilon_2, 0)-differential privacy** for the reported sensitivity.

### 2. Composition
The pipeline currently accounts privacy budget as the sum of two stages:

- `epsilon_1` — used by the candidate-generation / token-classification stage.
- `epsilon_2` — used by the exponential-mechanism selection stage.

Reported total: `epsilon_total = epsilon_1 + epsilon_2`.

### 3. Fallback when no candidate is valid
If every generated candidate is filtered out, the pipeline returns the baseline `x_priv` (the canonicalised/protected-token-only version of the input) and marks `fallback: true`. This is safe by construction: no private choice is released beyond the protected-token baseline.

### 4. No model downloads by default
The module will not download models from HuggingFace, Unsloth, or any other remote source unless `EM_HSD_ALLOW_DOWNLOADS=1` is set. This prevents accidental privacy-relevant network leakage on a fresh checkout.

## Non-claims

### 1. Not formally proven end-to-end DP for the whole pipeline
This implementation uses deterministic mock/proxy models and lightweight score functions. A formal end-to-end DP proof would require:
- A clearly defined neighbouring-dataset model (e.g., one-sentence substitution).
- Verified sensitivity bounds for every feature used in the score function.
- A calibrated `epsilon_1` for any token-level randomisation.
- Composition analysis that includes deterministic post-processing.

### 2. Not robust against adaptive attacks
The current code does not include mechanisms for adaptive composition, subsampling amplification, or privacy-amplification-by-iteration.

### 3. No certified semantic utility
Semantic similarity is measured by a proxy model. Higher similarity usually correlates with better utility, but there is no formal guarantee that the output preserves downstream task performance.

### 4. No certified hate-speech reduction
The hate-score proxy is a lightweight heuristic. It does not guarantee that the output is non-toxic, non-offensive, or compliant with any content policy.

## How to interpret the reported numbers

- `epsilon_total`: the sum of configured stage budgets. Lower is stronger privacy.
- `delta_u`: the allowed semantic-utility drop threshold used during candidate filtering.
- `P_hate_original` / `P_hate_x_priv`: proxy hate scores, not calibrated probabilities.

## Future work

- Replace proxy models with formally analysed DP mechanisms (e.g., objective perturbation, DP-SGD) if end-to-end certification is required.
- Add confidence intervals and composition accounting for repeated queries.
- Evaluate the pipeline against a held-out adversarial set with explicit attack models.

