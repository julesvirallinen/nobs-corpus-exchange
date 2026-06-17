# Layer 1 — Cross-Saliency Token Triage

Detailed design document for **TRIAGE-DP**.

## 1. Purpose and role in the stack

Layer 1 is the **core decision engine** of TRIAGE-DP. Before any differentially private rewrite runs, it answers one question **for every token** in the input text:

> *How should this token be treated to maximize hate-speech detection utility while reducing authorship re-identifiability?*

Layer 1 does **not** perform the DP rewrite itself. It produces a **routing decision** (quadrant + action + ε budget) that downstream layers consume:

| Downstream | Uses Layer 1 output |
|------------|---------------------|
| Normalization | Runs **before** Layer 1 (deterministic pre-processing) |
| Layer 2 (Biber priors) | Boosts `A(t)` before quadrant assignment |
| DP-MLM (`mechanism/dp.py`) | Executes Q1 moderate / Q2 aggressive rewrites |
| Canonicalization | Executes Q1 spelling de-obfuscation |
| Layer 4 (sentence EM) | Receives hard-row flag when token-level gain is insufficient |
| Audit log | Records H(t), A(t), quadrant, action per token |

Without Layer 1, TRIAGE-DP collapses to uniform dpmlm or fixed-rule SPINE—both insufficient for the PrivHSD research claim.

---

## 2. Problem Layer 1 solves

### 2.1 Failure modes of uniform DP-MLM

Plain **dpmlm** applies the same privacy budget to (almost) every content token:

| Failure | Mechanism | Effect on TO |
|---------|-----------|--------------|
| **Utility destruction** | Hate-evidence tokens resampled to neutral synonyms | Utility ratio ↓ |
| **Budget misallocation** | ε spent on function words and generic tokens | Privacy ratio barely ↓ |
| **Register blindness** | Involved/persuasive markers rewritten uniformly | HSD loses context-specific signal |

### 2.2 Failure modes of rule-based SPINE

SPINE (`Johnny t0-1.03`) improves on dpmlm with:

- Lexicon-protected hate tokens
- Skipped function words
- Uniform `epsilon.content` on remaining content

This is **static**: the same rules apply to every post, author, and context. It cannot distinguish:

- A generic insult (`moron`) → high utility, low authorship saliency → should skip DP
- A writer's habitual opener (`honestly mate`) → low utility impact, high authorship → should get aggressive DP
- An obfuscated slur (`d00fus`) → high on **both** axes → canonicalize, do not neutralize

Layer 1 replaces static classes with **measured, per-token, per-text** routing.

### 2.3 Relation to STAMP (literature)

STAMP (Tian et al., 2026) also uses a 2×2 token grid:

| Axis | STAMP | Layer 1 (TRIAGE-DP) |
|------|-------|---------------------|
| Privacy | PII / NER sensitivity | Authorship occlusion `A(t)` |
| Utility | Task embedding similarity | Hate occlusion `H(t)` |
| Mechanism | Polar embedding LDP | DP-MLM exponential mechanism |

Same **structure**, different **operationalization** for IA-HSD.

---

## 3. Core concept: occlusion saliency

### 3.1 Definition
"For each token: “How much does this specific word matter for what we care about?”
The importance of a token is measured in the context of the full sentence, using a model that reads the whole text — not from an isolated word list.
For probe score function `S(text)` and token `t`:

```
Δ(t) = S(text) − S(text \ {t})
```

where `text \ {t}` is the text with token `t` removed or replaced by whitespace (occluded).

| Δ(t) | Interpretation |
|------|----------------|
| **Large positive** | Token is **salient** for what the probe measures |
| **Near zero** | Token is ** irrelevant** to the probe |
| **Negative** | Removing token **helps** the probe (rare; often noise) |

Layer 1 uses **two probes** → two saliency scores per token:

- **H(t)** — utility (hate) saliency
- **A(t)** — privacy (authorship) saliency

### 3.2 Occlusion procedure (pseudocode)

```
function score_all_tokens(text, segments, probe):
    surfaces = [seg.text for seg in segments if seg.is_word]
    full_text = reassemble(segments)
    S_full = probe.score(full_text)

    scores = {}
    for i, seg in enumerate(word_segments):
        occluded_surfaces = surfaces.copy()
        occluded_surfaces[i] = " "   # or [MASK] for MLM-compatible probes
        S_occluded = probe.score(reassemble(occluded_surfaces))
        scores[seg] = S_full - S_occluded
    return scores
```

### 3.3 Occlusion variants

| Variant | When to use | Trade-off |
|---------|-------------|-----------|
| **Blank removal** | Fast; matches existing `saliency.py` | May break tokenization boundaries |
| **Space substitution** | Preserves word count | Slight grammar artifact |
| **Mask token** | Transformer probes | More realistic for MLM context |
| **Phrase occlusion** | Multi-word hate phrases | Higher cost; better H(t) for context-dependent hate |

**Default:** blank/space substitution on word segments after normalization.

### 3.4 Which tokens to score

Not every segment needs full occlusion:

| Segment kind | H(t) | A(t) | Rationale |
|--------------|------|------|-----------|
| `WORD` (content) | Yes | Yes | Primary routing targets |
| `WORD` (function, pre-classified) | Skip | Optional light | Layer 1 can inherit SPINE function-word skip |
| `PROTECTED` (lexicon) | Optional confirm | Yes | Lexicon → likely Q1 or Q3 |
| `URL`, `MENTION`, `@handle` | Low | **High** | Strong identity leak; often Q2 or redact |
| `HASHTAG` | Context | High | May carry author community signal |
| `SEP` (punctuation, whitespace) | No | No | Kept as-is |

---

## 4. Utility score H(t)

### 4.1 Probe specification

| Property | Recommended value |
|----------|-------------------|
| Model | `cardiffnlp/twitter-roberta-base-hate-latest` or ensemble hate classifier |
| Output | P(hateful) = sum of softmax probs over hate/offensive/toxic labels |
| Location | `mechanism/` only — **never import from `harness/`** |
| Training data | Public hate-detection checkpoint; no PrivHSD dev labels in mechanism |

Existing partial implementation: `Johnny t0-1.03/src/mechanism/saliency.py` (`OcclusionSaliency`).

### 4.2 Formal definition

```
H(t) = P_hate(T) − P_hate(T \ {t})
```

where `T` is the normalized full text string.

### 4.3 Interpretation bands (before thresholding)

| H(t) range (illustrative) | Meaning | Typical routing |
|---------------------------|---------|-----------------|
| H(t) ≥ 0.25 | Strong hate-evidence | Q1 or Q3 (with A(t)) |
| 0.05 ≤ H(t) < 0.25 | Moderate contribution | Context-dependent |
| H(t) < 0.05 | Neutral for hate | Q2 or Q4 (with A(t)) |

Thresholds `τ_H` are set by Layer 3 calibration, not hand-tuned.

### 4.4 Examples

**Text:** `"you total dummy, you ruined the whole project"`

| Token | Expected H(t) | Why |
|-------|---------------|-----|
| `dummy` | High (~0.3+) | Direct insult; removal drops hate score sharply |
| `ruined` | Medium | Hostile verb in context |
| `project` | Low | Neutral noun |
| `the` | ~0 | Function word |

**Text:** `"go back where you came from"` (no single slur)

| Token | Expected H(t) | Why |
|-------|---------------|-----|
| `back` | Medium–high | Part of exclusion phrase |
| `where` | Medium | Context-dependent |
| Individual function words | Low | Phrase-level hate may need phrase occlusion extension |

### 4.5 H(t) beyond lexicon

Lexicon matching fails on:

- Novel slurs not in list
- Obfuscated spellings (`d00fus`, `d u m m y`)
- Implicit hate without keyword hits
- Sarcasm + target group references

H(t) catches these because it measures **classifier behavior**, not string matching.

### 4.6 Failure modes and guards

| Risk | Mitigation |
|------|------------|
| Classifier false positive on benign text | Combine H(t) with lexicon; cap Q3 protection |
| Classifier blind to dialect | Use ensemble or domain-fine-tuned probe |
| Occlusion noise on short texts | Require minimum text length for H(t) trust |
| Adversarial tokens gaming H(t) | Layer 3 calibrates on dev; audit log review |

---

## 5. Privacy score A(t)

### 5.1 Threat model

PrivHSD privacy = **authorship re-identification via stylometry**, not PII extraction.

The evaluation harness trains an attacker on original texts with `Author` labels, then measures top-1 accuracy on privatized texts (`harness/reident.py`). Layer 1 must **approximate what tokens enable that attack** without seeing `Author` at runtime.

### 5.2 Author-blind constraint

The mechanism receives only `Text`. Therefore A(t) **cannot** be:

```
❌ A(t) = 1[ predict(Author | T) = bob ] − 1[ predict(Author | T\{t}) = bob ]
```

That requires knowing the true author at sanitization time.

### 5.3 Recommended A(t) definitions (author-blind)

#### Option A — Confidence drop (primary)

Train a multi-class authorship-style classifier on **public stylometry data** or on a **frozen snapshot** of in-domain authors (labels used only at train time for the probe weights, not per-row at inference):

```
A(t) = conf(T) − conf(T \ {t})
```

where `conf(T) = max_c P(author_class=c | T)` or SVM decision margin magnitude.

**Intuition:** If removing `t` collapses the classifier's confidence, `t` carried identity information.

#### Option B — Embedding shift

```
A(t) = ‖ embed(T) − embed(T \ {t}) ‖₂
```

using a frozen authorship encoder (e.g. LUAR-style or mean-pooled transformer). No author label needed at inference.

#### Option C — Self-supervised rarity

```
A(t) = rarity(t) × idf(t)
```

Cheap proxy; less accurate than Options A/B. Use only as pre-filter.

### 5.4 Probe architecture (mechanism-side)

Mirror harness attacker family but **separate instance**:

| Tier | Model | Cost | Use |
|------|-------|------|-----|
| Fast | Char n-gram TF-IDF + linear model | Very low | All tokens |
| Slow | Frozen transformer + linear probe | High | Tie-break on borderline A(t) |

Implementation target: `mechanism/authorship_saliency.py`.

### 5.5 Interpretation

| A(t) | Meaning |
|------|---------|
| **High** | Distinctive vocabulary, dialect, spelling habits, @handles, rare bigrams |
| **Low** | Generic words appearing in many authors' posts |
| **Medium** | Context-dependent (e.g. `you` is generic alone but part of author tic) |

### 5.6 Examples

**Same text:** `"Honestly you're such a d00fus, only a moron would believe that scam"`

| Token | Expected A(t) | Why |
|-------|---------------|-----|
| `Honestly` | High | Distinctive discourse opener if habitual for author |
| `d00fus` | Medium–high | Idiosyncratic leet spelling = stylometric fingerprint |
| `moron` | Low–medium | Common insult; many authors use it |
| `you're` | Low | Generic |
| `@bob_handle` | Very high | Direct identifier (if present) |

---

## 6. Quadrant routing

### 6.1 Thresholding

Given calibrated thresholds `τ_H`, `τ_A` (from Layer 3):

```
high_H = (H(t) >= τ_H)
high_A = (A(t) >= τ_A)
```

Assign quadrant:

| high_A | high_H | Quadrant |
|--------|--------|----------|
| T | T | **Q1** — conflict |
| T | F | **Q2** — privacy-only |
| F | T | **Q3** — utility-only |
| F | F | **Q4** — neutral |

Optional: **soft routing** with continuous ε interpolation near boundaries (advanced; default is hard quadrants).

### 6.2 Quadrant actions (complete specification)

#### Q1 — High A, High H (privacy + utility conflict)

**Scenario:** Token is both hate-evidence and authorship-leaking (obfuscated slur, author-specific insult spelling).

| Step | Action |
|------|--------|
| 1 | **Canonicalize** surface form (de-leet, de-space, de-elongate, lowercase) |
| 2 | Mark as `PROTECTED` — **no DP resampling** to neutral synonym |
| 3 | Optional: moderate DP on **adjacent** Q2 tokens only |
| ε on token | `skip` (canonicalization is empirical obfuscation, not formal DP) |

**Must not:** Replace `d00fus` → `person` (destroys utility).

**Implementation hook:** `mechanism/canonicalize.py`, `mechanism/lexicon.py`.

#### Q2 — High A, Low H (primary privacy work)

**Scenario:** Stylometric identity carrier with negligible hate impact.

| Step | Action |
|------|--------|
| 1 | DP-MLM rewrite via exponential mechanism |
| 2 | Use **low ε** (aggressive noise) from config `ε_Q2` |
| 3 | MLM candidate set includes semantically similar alternatives |

**Budget:** Majority of formal DP ε should land here.

**Implementation hook:** `mechanism/spine.py` loop + `mechanism/dp.py` + `mechanism/mlm.py`.

#### Q3 — Low A, High H (utility protection)

**Scenario:** Hate-evidence without strong author fingerprint.

| Step | Action |
|------|--------|
| 1 | Skip DP rewrite |
| 2 | Apply normalization only if not already done |
| ε on token | `skip` |

**Implementation hook:** Same as SPINE protected-in-place without lexicon requirement.

#### Q4 — Low A, Low H (neutral)

**Scenario:** Function words, generic glue, semantically inert tokens.

| Step | Action |
|------|--------|
| 1 | Skip DP |
| 2 | Normalization only (lowercase, whitespace) |
| ε on token | `skip` |

**Optimization:** Skip occlusion entirely for pre-identified function words (`salient.py` `FUNCTION_WORDS`).

### 6.3 Per-quadrant ε map

| Quadrant | Typical ε | DP applied? |
|----------|-----------|-------------|
| Q1 | `skip` on token; optional `ε_Q1_context` on neighbors | Canonicalize only on token |
| Q2 | `ε_Q2` (low numeric = aggressive) | Yes |
| Q3 | `skip` | No |
| Q4 | `skip` | No |

All numeric ε values calibrated in Layer 3.

---

## 7. End-to-end pipeline (Layer 1 only)

```
Input: raw Text string
  │
  ▼
[Normalize]  ← deterministic; not Layer 1 but prerequisite
  │
  ▼
[Segment]    ← tokenize.py
  │
  ▼
[Lexicon pre-filter]  ← optional fast path to Q1/Q3 candidates
  │
  ▼
[Compute H(t) for candidate word tokens]  ← hate occlusion probe
  │
  ▼
[Compute A(t) for candidate word tokens]  ← authorship occlusion probe
  │
  ▼
[Layer 2: boost A(t) for Biber tags]  ← see layer-02 doc
  │
  ▼
[Assign quadrant using τ_H, τ_A]
  │
  ▼
[Emit routing table per token]
  │
  ▼
[Hand off to DP-MLM / canonicalize / skip executors]
```

### 7.1 Worked example (full trace)

**Input:** `"Honestly you're such a d00fus, only a moron would believe that scam"`

**After normalization:** same surface (no elongation in this example)

| # | Token | H(t) | A(t) | Quadrant | Action | Output token |
|---|-------|------|------|----------|--------|--------------|
| 1 | Honestly | 0.02 | 0.31 | Q2 | Aggressive DP | Really |
| 2 | you're | 0.01 | 0.08 | Q4 | Skip | you're |
| 3 | such | 0.00 | 0.05 | Q4 | Skip | such |
| 4 | a | 0.00 | 0.00 | Q4 | Skip | a |
| 5 | d00fus | 0.41 | 0.22 | Q1 | Canonicalize | dummy |
| 6 | only | 0.03 | 0.04 | Q4 | Skip | only |
| 7 | a | 0.00 | 0.00 | Q4 | Skip | a |
| 8 | moron | 0.38 | 0.11 | Q3 | Skip | moron |
| 9 | would | 0.00 | 0.02 | Q4 | Skip | would |
| 10 | believe | 0.01 | 0.06 | Q4 | Skip | believe |
| 11 | that | 0.00 | 0.01 | Q4 | Skip | that |
| 12 | scam | 0.12 | 0.09 | Q3 | Skip | scam |

**Output:** `"Really you're such a dummy, only a moron would believe that scam"`

**Expected effects:**

- Hate classifier: still fires (dummy, moron, scam preserved)
- Authorship probe: confidence drop from `Honestly` → `Really`, `d00fus` → `dummy`

---

## 8. Data structures and API

### 8.1 Proposed module: `mechanism/triage.py`

```python
@dataclass
class TokenRoute:
    segment_index: int
    original: str
    H: float
    A: float
    H_boost: float          # from Layer 2; 0 if disabled
    A_effective: float      # A + H_boost (or max)
    quadrant: Literal["Q1", "Q2", "Q3", "Q4"]
    action: Literal["canonicalize", "dp_rewrite", "skip", "normalize_only"]
    epsilon: float | None
    reason: str

def triage_segments(
    segments: list[Segment],
    config: Config,
    hate_probe: HateProbe,
    auth_probe: AuthorshipProbe,
) -> list[TokenRoute]:
    ...
```

### 8.2 Audit log schema (per token)

```json
{
  "row_index": 14,
  "token_index": 5,
  "original": "d00fus",
  "normalized": "d00fus",
  "H": 0.41,
  "A": 0.22,
  "A_effective": 0.28,
  "biber_tags": ["lexical_substitution"],
  "quadrant": "Q1",
  "action": "canonicalized",
  "epsilon": null,
  "replacement": "dummy",
  "reason": "Q1: high hate saliency + high authorship saliency; canonicalized not resampled"
}
```

---

## 9. Computational cost

### 9.1 Complexity

For `n` word tokens and two probes:

- **Worst case:** `2 × (1 + n)` probe forward passes ≈ **O(n)** per text
- **With function-word skip:** `2 × (1 + n_content)` where `n_content ≤ n`

### 9.2 Mitigations (production order)

1. **Function-word skip** — no occlusion for `FUNCTION_WORDS`
2. **Lexicon short-circuit** — known hate terms → skip H(t) full scan or assign Q1 directly
3. **Two-tier A(t)** — fast char-SVM for all; transformer only if A_SVM borderline
4. **Batch occlusions** — stack occluded texts in one GPU batch
5. **Cache S_full** — one baseline per probe per text
6. **Max tokens cap** — for very long posts, sample content tokens or delegate to Layer 4

### 9.3 Expected latency (order of magnitude)

| Config | ~40-token post | ~200-token post |
|--------|----------------|-----------------|
| H(t) transformer only | ~2–5 s CPU | ~10–25 s CPU |
| + A(t) char-SVM | +0.1 s | +0.5 s |
| + both transformer | ~4–10 s | ~20–50 s |

Layer 4 hard-row path handles cases where token-level cost is prohibitive.

---

## 10. Integration with existing codebase

| Existing file | Layer 1 relationship |
|---------------|---------------------|
| `mechanism/saliency.py` | **Extend** — export H(t) scores; remove binary-only threshold |
| `mechanism/authorship_saliency.py` | **New** — A(t) probe |
| `mechanism/triage.py` | **New** — quadrant routing |
| `mechanism/spine.py` | **Modify** — call triage before DP loop |
| `mechanism/salient.py` | **Demote** — function-word pre-filter only |
| `mechanism/lexicon.py` | **Keep** — pre-filter + Q1 canonicalization |
| `harness/reident.py` | **Separate** — evaluation only; never imported by mechanism |

### 10.1 Migration from SPINE mode

| SPINE behavior | Layer 1 equivalent |
|----------------|-------------------|
| `protected: skip` | Q1 + Q3 |
| `function_word: skip` | Q4 pre-filter |
| `content: ε` uniform | Q2 with `ε_Q2`; Q1 neighbors optional |
| `saliency.enabled` binary | Full H(t) routing |

---

## 11. Testing strategy

### 11.1 Unit tests

- Quadrant assignment given fixed H, A, τ values
- Q1 canonicalize never calls `dp.select`
- Q3/Q4 never receive ε > 0
- Author-blind: mechanism test suite never passes `Author` column

### 11.2 Integration tests

- Synthetic CSV rows with known insult + distinctive opener
- Verify output preserves insult tokens, rewrites opener
- Log completeness: every token has H, A, quadrant

### 11.3 Regression vs SPINE

- On lexicon-hit-only rows, Layer 1 should **match or beat** SPINE utility
- On stylometric-heavy benign rows, Layer 1 should **beat** SPINE privacy

---

## 12. Ablation: what Layer 1 alone contributes

| Config | Description |
|--------|-------------|
| L1-full | H(t) + A(t) + quadrants |
| L1-H-only | Protect high H(t); uniform DP elsewhere → SPINE-like utility, weak privacy targeting |
| L1-A-only | Aggressive DP on high A(t); no hate protection → dpmlm-like utility loss |
| L1-no-occlusion | Fixed rules only → equals SPINE |

Layer 1's research claim requires **L1-full** to dominate on TO vs ablations.

---

## 13. References

- Meisenbacher et al., DP-MLM — exponential mechanism substrate
- Tian et al., STAMP — 2×2 partitioning precedent (different axes)
- Arnold, DP Rewriting reshapes Linguistic Style — register-blind failure mode
- `resources/case study.png` — dpmlm vs Presidio vs GPT baselines
- `Johnny t0-1.03/src/mechanism/saliency.py` — existing H(t) partial impl

---

## 14. Open design decisions

| Decision | Options | Recommendation |
|----------|---------|----------------|
| A(t) probe | Confidence drop vs embedding shift | Confidence drop (char-SVM) + embedding tie-break |
| Phrase-level H(t) | Off vs n-gram phrases | Phase 2: bigrams for known hate constructions |
| Soft quadrants | Hard vs interpolated ε | Hard for v1; soft for calibration fine-tuning |
| URL/@mention | Q2 always vs redact | Q2 DP or replace with generic `<USER>` (log as high-A) |
