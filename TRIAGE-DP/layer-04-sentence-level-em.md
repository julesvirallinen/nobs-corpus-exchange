# Layer 4 — Sentence-Level Exponential Mechanism for Hard Rows

Detailed design document for **TRIAGE-DP**. See the overview in [`../TRIAGE-DP.md`](../TRIAGE-DP.md). Builds on Layer 1 ([`layer-01-cross-saliency-triage.md`](layer-01-cross-saliency-triage.md)) and Layer 3 parameters ([`layer-03-to-calibration.md`](layer-03-to-calibration.md)).

Existing scaffold: `Johnny t0-1.03/src/stretch/candidate_selection.py`.

---

## 1. Purpose and role in the stack

Layers 1–3 perform **token-level** privatization: each word is routed (Q1–Q4) and optionally rewritten by DP-MLM one token at a time.

Token-level DP has limits:

| Limitation | Why it matters |
|------------|----------------|
| **Local context only** | MLM sees left/right window; cannot restructure whole argument |
| **Long posts** | Many tokens → composition of many ε draws; stylometric patterns persist at phrase level |
| **Phrase-level identity** | Catchphrases, repeated constructions, paragraph rhythm survive token swaps |
| **Occlusion cost** | Layer 1 on 500-word post is expensive |

**Layer 4** activates for **hard rows** where token-level sanitization yields insufficient privacy gain (or excessive cost). It:

1. Proposes **k** full-text candidate rewrites from a **local** generative model
2. Scores each candidate for **hate detection utility**
3. Selects one candidate via the **exponential mechanism** over utility scores (formal DP at sentence level)

Layer 4 is **optional** (`stretch.enabled: false` for ablation) but important for long Reddit posts and for beating dpmlm on privacy-heavy configs without destroying utility.

---

## 2. When Layer 4 activates (gating)

### 2.1 Hard-row definition

Combine structural and outcome-based gates:

#### Gate A — Length (implemented)

From `stretch/candidate_selection.py`:

```python
def is_hard(text: str, hard_row_min_tokens: int) -> bool:
    return len(re.findall(r"[^\W_]+", text)) >= hard_row_min_tokens
```

Default `hard_row_min_tokens: 40` (configurable by Layer 3).

#### Gate B — Privacy residual (recommended)

After Layer 1–3 token pipeline, estimate whether privacy target met:

```
if is_hard_length(text) AND privacy_gain(row) < target:
    trigger Layer 4
```

where `privacy_gain = 1 − (A_proxy_priv / A_proxy_orig)` using mechanism-side authorship proxy on **this row only** (not harness Author labels).

#### Gate C — Q2 saturation (optional)

If >60% of content tokens already Q2-rewritten but stylometric embedding distance to original still below threshold → trigger Layer 4.

### 2.2 When Layer 4 must NOT run

| Condition | Reason |
|-----------|--------|
| Short posts (<40 tokens) | Token-level sufficient; avoid over-smoothing |
| Layer 4 disabled in config | Ablation baseline |
| Generative proposer unavailable | Fail closed to token-level output + warning |
| Utility-critical minimal texts | Risk of holistically destroying hate signal |

---

## 3. Pipeline overview

```
Input text (single row)
  │
  ▼
[Layers 1–3 token-level privatization]  → intermediate text T'
  │
  ▼
[Hard-row gate]
  │ no ──► output T'
  │ yes
  ▼
[Extract protected spans from Layer 1 log]
  │  (Q1 canonical forms + Q3 skip tokens)
  ▼
[Generative proposer: k constrained paraphrases of T']
  │
  ▼
[Filter candidates violating constraints]
  │
  ▼
[Utility scorer: hate classifier score per candidate]
  │
  ▼
[Exponential mechanism: select one candidate]
  │
  ▼
[Merge protected spans back if proposer drifted]
  │
  ▼
Output text + sentence-level audit entry
```

---

## 4. Component 1 — Generative proposer

### 4.1 Design requirements

| Requirement | Rationale |
|-------------|-----------|
| **Local open-weight** | No API leakage of hate speech content |
| **Small / CPU-viable** | Hackathon reproducibility |
| **Pinned version** | Reproducible submission |
| **Constrained decoding** | Must not remove protected hate evidence |

### 4.2 Proposed module: `stretch/generative_proposer.py`

Implements `GenerativeProposer` protocol from scaffold:

```python
class LocalParaphraseProposer:
    def propose(self, text: str, k: int, protected_spans: list[Span]) -> list[str]:
        ...
```

### 4.3 Model candidates (examples)

| Model class | Size | Notes |
|-------------|------|-------|
| FLAN-T5-small/base | 80M–250M | Instruction paraphrase |
| TinyLlama / Qwen2.5-0.5B | 500M | Needs strict prompting |
| Pegasus paraphrase | 570M | Paraphrase-specialized |

**Not allowed for core claim:** GPT-4o API (PrivRewrite style)—conflicts with local processing narrative.

### 4.4 Prompt template (conceptual)

```
Rewrite the following social media post to express the same meaning in different words.
Rules:
- Do NOT remove or soften insults or offensive terms listed in PROTECTED: [{protected}]
- Change stylistic choices (openers, filler words, distinctive phrases)
- Keep similar length ±20%
- Output only the rewritten post

Post: {text}
```

Generate k outputs with temperature sampling **before** DP selection (sampling is not the privacy step—the **selection** is).

### 4.5 Candidate diversity

Ensure k candidates differ:

- Sample with temperatures `[0.7, 0.9, 1.0, 1.1]`
- Or use different prompt variants ("formal", "neutral", "casual")
- Deduplicate by normalized string hash

---

## 5. Component 2 — Constraint enforcement

### 5.1 Protected span list

From Layer 1 token log for this row:

```python
protected_spans = [
  {"surface": "dummy", "canonical": "dummy", "quadrant": "Q1"},
  {"surface": "moron", "quadrant": "Q3"},
  ...
]
```

### 5.2 Validation rules

Reject candidate `c` if:

| Rule | Check |
|------|-------|
| **Lexical preservation** | Each Q3 token (case-insensitive skeleton) appears in `c` |
| **Category preservation** | Q1 canonical form appears (not neutralized to "person") |
| **Min hate score** | `P_hate(c) >= P_hate(T') - δ` |
| **Max length ratio** | `0.5 <= len(c)/len(T') <= 2.0` |

If fewer than 2 valid candidates remain, **fall back** to token-level `T'` without sentence EM.

### 5.3 Post-selection merge

If valid candidate passes but dropped a protected token due to model error:

- Replace candidate with token-level output **or**
- Surgical insert of missing canonical tokens at logged positions

Prefer conservative: **reject candidate** rather than silently lose hate signal.

---

## 6. Component 3 — Utility scorer

### 6.1 Scoring function

For each candidate `c_i`:

```
u_i = score(c_i) = P_hate(c_i)
```

Alternative composite (research extension):

```
u_i = α · P_hate(c_i) + (1−α) · semantic_sim(c_i, T')   # sim capped, no external API
```

Default: hate probability only—aligned with harness utility.

### 6.2 Scorer implementation

Reuse mechanism-side hate classifier (same family as Layer 1 H(t) probe, separate instance):

```python
class HateUtilityScorer:
    def score(self, candidate: str) -> float:
        return self.probe.hate_prob(candidate)
```

Implements `CandidateScorer` protocol in scaffold.

---

## 7. Component 4 — Exponential mechanism selection

### 7.1 DP guarantee scope

Selection step satisfies **ε_sentence-DP** with respect to the **utility score vector** `(u_1, …, u_k)` clipped to `[-C, C]`, using same math as token-level DP:

From `mechanism/dp.py`:

```
P(select c_i) ∝ exp(ε · u_i / (2Δ))    where Δ = 2C
```

**Composition note:** Sentence-level ε adds to token-level ε budget on hard rows. Report both in audit log; document sequential composition (narrow claim).

### 7.2 Implemented function

From `stretch/candidate_selection.py`:

```python
def select_rewrite(candidates, scores, epsilon, clip, rng):
    return dp.select(candidates, scores, epsilon, clip, rng)
```

### 7.3 Parameter `epsilon_sentence`

Set by Layer 3 calibration (`epsilon.sentence` in config). Typical range: 10–30 (same semantics as DP-MLM—higher = more utility-preserving selection).

### 7.4 Edge cases

| Case | Behavior |
|------|----------|
| k=1 valid candidate | Return it; EM degenerates (log ε=N/A) |
| All scores equal | Uniform selection (maximum privacy randomness) |
| One score dominates, high ε | Near-argmax selection |

---

## 8. Orchestration class

Scaffold provides `CandidateSelectionLayer`:

```python
class CandidateSelectionLayer:
    def rewrite(self, text, rng):
        candidates = self.proposer.propose(text, self.k)
        scores = [self.scorer.score(c) for c in candidates]
        return select_rewrite(candidates, scores, self.epsilon, self.clip, rng)
```

**TRIAGE-DP extension:** wire into `spine.py` or `wrapper/run.py`:

```python
if config.stretch.enabled and is_hard(text, config.stretch.hard_row_min_tokens):
    text, sel = stretch_layer.rewrite(text_after_token_pass, protected, rng)
    log_sentence_level(..., sel)
```

---

## 9. Worked example

### 9.1 Input (hard row, 45 tokens)

Long rant mixing distinctive author tic + hate:

> `"Honestly honestly you guys are all d00fuses, I swear every single time you people believe this scam it's because you're morons, I've said it before here in this town..."`

### 9.2 After token-level pass (Layers 1–3)

- `d00fus` → `dummy` (Q1)
- `morons` kept (Q3)
- `Honestly` → `Really` (Q2)
- Residual: repeated "honestly", "here in this town", rhythmic phrasing remain → authorship proxy still high

### 9.3 Layer 4

**Protected:** `dummy`, `morons`, `scam`

**Candidates (k=4):**

| # | Candidate (abbrev) | P_hate | Valid? |
|---|-------------------|--------|--------|
| 1 | "Really you all are dummies... morons... scam..." | 0.89 | ✓ |
| 2 | "You people are individuals... morons... scam" | 0.41 | ✗ (neutralized) |
| 3 | "Every time you believe this scam, morons, ..." | 0.85 | ✓ |
| 4 | "In this city you always fall for scams, morons..." | 0.82 | ✓ |

**EM selection** (ε_sentence=18): likely picks candidate 1 or 3.

**Privacy gain:** removes "here in this town", double "honestly" tic, restructures author-specific rhythm.

---

## 10. Comparison to PrivRewrite

| Aspect | PrivRewrite (Kim 2025) | Layer 4 |
|--------|------------------------|---------|
| Model access | Black-box LLM API | Local open-weight |
| Candidates | Random prompts to API | Local paraphrase + constraints |
| Selection | EM with refined sensitivity | EM (same family) |
| Utility | Semantic / fluency | **Hate classifier utility** |
| Constraints | General | **Protected hate spans from Layer 1** |
| When | Always sentence-level | **Hard rows only** |

Layer 4 borrows **structure** (candidates + EM), not deployment model.

---

## 11. Privacy and honesty

### 11.1 What Layer 4 claims

- Exponential-mechanism **selection among k candidates** satisfies ε_sentence-DP w.r.t. clipped utility scores.
- **Proposer generation** (sampling paraphrases) is **not** claimed as DP unless generation itself is privatized (future work).

### 11.2 What Layer 4 does not claim

- End-to-end document DP across token + sentence steps without composition accounting
- DP on generative model decoding (only on selection step in v1)

### 11.3 Audit log (sentence-level entry)

```json
{
  "level": "sentence",
  "hard_row": true,
  "k_requested": 4,
  "k_valid": 3,
  "candidates_hash": ["a1b2...", "c3d4...", "e5f6..."],
  "scores": [0.89, 0.85, 0.82],
  "epsilon_sentence": 18.0,
  "selected_index": 0,
  "protected_spans": ["dummy", "morons", "scam"],
  "reason": "hard row: token-level privacy gain below target"
}
```

---

## 12. Computational cost

| Stage | Cost driver |
|-------|-------------|
| Proposer | k × forward passes generative model |
| Scorer | k × hate classifier forward pass |
| Selection | O(k) negligible |

Example: k=4, 45-token post, small T5 → ~2–10 s CPU per hard row.

**Mitigation:** Layer 4 only on gated rows (~10–30% of dataset if threshold=40 tokens).

---

## 13. Testing strategy

### 13.1 Existing tests (`tests/test_stretch.py`)

- `select_rewrite` high ε → picks best score
- `select_rewrite` low ε → varied selection
- `NotImplementedProposer` raises

### 13.2 New tests required

- Constraint filter removes neutralizing candidate
- Protected token `moron` present in all valid candidates
- Hard-row gate false for short text
- Integration: hard synthetic row triggers Layer 4 log entry
- Fallback when all candidates invalid

---

## 14. Ablation

| Config | Expected |
|--------|----------|
| Token-only (stretch off) | Good utility; weak privacy on long posts |
| Layer 4 on, no constraints | Utility collapse |
| Layer 4 on, full constraints | Best TO on long-post subset |
| Layer 4 on all rows | Over-smoothing; utility risk |

Report TO stratified by row length quartile.

---

## 15. Limitations

- Generative model may introduce bias or fluency artifacts
- k and prompt design affect candidate pool quality strongly
- Sentence-level rewrite may desync from token-level audit narrative
- Additional model download and memory footprint
- English-centric prompts

---

## 16. Implementation checklist

| Task | Status in repo |
|------|----------------|
| `select_rewrite` EM | ✅ Done |
| `is_hard` gate | ✅ Done |
| `CandidateSelectionLayer` | ✅ Done |
| `GenerativeProposer` | ❌ NotImplemented |
| Protected-span constraints | ❌ To build |
| Integration in `spine.py` | ❌ To build |
| Sentence audit log | ❌ To build |
| Layer 3 `epsilon_sentence` tune | ❌ To build |

---

## 17. References

- Kim, **PrivRewrite** — candidate pool + black-box EM
- Meisenbacher et al., **DP-MLM** — shared exponential mechanism math
- `Johnny t0-1.03/src/stretch/candidate_selection.py` — scaffold
- `Johnny t0-1.03/src/mechanism/dp.py` — `dp.select`
- [`layer-01-cross-saliency-triage.md`](layer-01-cross-saliency-triage.md) — protected spans
- [`layer-03-to-calibration.md`](layer-03-to-calibration.md) — `epsilon_sentence`

---

## 18. Open design decisions

| Decision | Recommendation |
|----------|----------------|
| Run Layer 4 on T or T' | **T'** (after token pass) |
| k | 4 default |
| Proposer model | FLAN-T5-base pinned |
| Invalid candidates | Fallback to T' |
| Compose ε | Log token_ε_sum + epsilon_sentence separately |
