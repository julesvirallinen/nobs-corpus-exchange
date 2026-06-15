# Layer 2 — Stylometric Priors

Detailed design document for **TRIAGE-DP**. See the overview in [`../TRIAGE-DP.md`](../TRIAGE-DP.md). Layer 1 routing is defined in [`layer-01-cross-saliency-triage.md`](layer-01-cross-saliency-triage.md).

---

## 1. Purpose and role in the stack

Layer 2 **refines** the privacy score `A(t)` from Layer 1 using **linguistic theory** about which token types carry authorship identity vs task (hate) signal.

| Without Layer 2 | With Layer 2 |
|-----------------|--------------|
| A(t) depends only on occlusion probe outputs | A(t) boosted for known stylometric identity carriers |
| Probe may miss rare but systematic identity markers | Biber-style features catch pronoun/register patterns |
| Uniform treatment of e.g. all pronouns | Targeted boost only on identity-heavy subclasses |

Layer 2 does **not** replace occlusion measurement. It is a **prior** that:

1. Increases recall of privacy-relevant tokens (fewer identity leaks in Q4)
2. Aligns DP spending with Arnold et al.'s finding that dpmlm is **register-blind**
3. Provides **interpretable reasons** in the audit log (`biber_tags`)

Layer 2 does **not** modify `H(t)` directly (utility axis stays classifier-driven).

---

## 2. Motivation from literature

### 2.1 Arnold et al. — register-blind sanitization

*Differentially-Private Text Rewriting reshapes Linguistic Style* (Arnold) shows that DP rewriting systematically shifts text toward:

- **Non-involved** register (loss of personal engagement)
- **Non-persuasive** register (loss of direct argumentation)
- Reduced **2nd-person pronouns**, **place/time adverbials**, **discourse particles**
- Simplified **subordination** patterns

These shifts are **stylistic homogenization**—exactly what reduces Burrows' Delta distance from author fingerprint—but they also destroy linguistic material common in **direct attack** and **online argument**, which hate classifiers may use.

**Key insight for TRIAGE-DP:**

| Feature type | Identity leakage | Hate utility |
|--------------|------------------|--------------|
| 2nd-person `you` in generic grammar | Low–medium alone | Often needed for direct attack |
| 2nd-person in author tic (`you guys always...`) | High | May be neutral for hate |
| Obfuscated slur | High (spelling) | High (semantics) |
| Place adverbial (`here`, `downtown`) | High (situational) | Usually low for hate |
| Discourse particle (`honestly`, `lol`) | High if habitual | Usually low for hate |

Layer 1 occlusion may under-estimate A(t) for **systematic grammatical classes** when individual token removal causes small confidence drops. Layer 2 adds **feature-class boosts** so Q2 captures register-level identity without waiting for probe sensitivity.

### 2.2 Biber (1991) multidimensional register framework

Biber's analysis decomposes register into dimensions including:

| Dimension | Poles | Online hate relevance |
|-----------|-------|----------------------|
| Involved vs informational | Personal vs impersonal | Hate often involved |
| Narrative vs expository | Story vs explanation | Mixed |
| Situational vs elaborated reference | Concrete context vs abstract | Situational = identity |
| Persuasive vs neutral | Argument vs report | Hate often persuasive |

Arnold's paper operationalizes **67 lexico-grammatical features** from Biber (1995 inventory)—computationally tractable with rule-based taggers, no ML required.

### 2.3 Why not boost H(t) symmetrically?

Stylometric priors are **asymmetric**:

- Identity lives in **how** something is said (register, dialect, habitual markers)
- Hate evidence lives in **what** is said (semantics, targets, slurs)—better captured by hate classifier occlusion H(t)

Boosting H(t) from Biber risks **protecting too much** involved/persuasive material and blocking needed Q2 rewrites.

---

## 3. Feature inventory (implementation subset)

Full Biber inventory = 67 features. For TRIAGE-DP Layer 2, prioritize features Arnold identifies as **degraded by dpmlm** and **identity-relevant**:

### 3.1 Tier 1 — high priority boosts

| Feature ID | Biber category | Examples | Default boost `β` |
|------------|----------------|----------|-------------------|
| B07 | 2nd person pronouns | you, your, yours | +0.15 A(t) |
| B04 | Place adverbials | here, there, downtown | +0.20 A(t) |
| B05 | Time adverbials | today, yesterday, always | +0.15 A(t) |
| B33 | Discourse particles | well, oh, lol, honestly | +0.20 A(t) |
| B06 | 1st person pronouns | I, me, my, we | +0.10 A(t) |
| B13 | Direct WH-questions | what, why (sentence-initial) | +0.10 A(t) |

### 3.2 Tier 2 — medium priority

| Feature ID | Category | Examples | Boost |
|------------|----------|----------|-------|
| B21–B32 | Subordination markers | because, although, if | +0.08 A(t) |
| B14–B16 | Nominalizations, gerunds | -tion, -ing noun forms | +0.05 A(t) |
| B17–B18 | Passive constructions | was done, by agent | +0.08 A(t) |

### 3.3 Tier 3 — do not boost (or negative boost)

| Feature | Reason |
|---------|--------|
| Protected hate lexicon tokens | Handled by Q1/Q3 via H(t) |
| Pure digits | Low identity in Reddit context |
| URLs, @mentions | Separate high-A rule in Layer 1 |
| Insult lexicon terms | High H(t); boosting A(t) risks wrong Q1 |

### 3.4 Feature–token assignment

Each **word token** receives zero or more tags:

```python
def biber_tags(token: str, context: TokenContext) -> list[str]:
    tags = []
    lower = token.lower()
    if lower in SECOND_PERSON:
        tags.append("B07_second_person")
    if lower in PLACE_ADVERBIALS:
        tags.append("B04_place")
    ...
    return tags
```

`TokenContext` may include previous/next token for multi-word patterns (e.g. `what` + `?` → WH-question).

---

## 4. Mathematical formulation

### 4.1 Effective privacy score

Let `A_raw(t)` = occlusion-based authorship saliency from Layer 1.

Let `B(t)` = sum of boosts from matched Biber tags:

```
B(t) = Σ_{f ∈ tags(t)} β_f
```

**Effective score:**

```
A_eff(t) = clamp(A_raw(t) + B(t), 0, A_max)
```

Default `A_max = 1.0` if scores normalized; else use percentile calibration on dev set.

### 4.2 Routing uses A_eff, not A_raw

Quadrant assignment (Layer 1 §6) replaces `A(t)` with `A_eff(t)`:

```
high_A = (A_eff(t) >= τ_A)
```

Audit log records both `A` and `A_effective` plus `biber_tags`.

### 4.3 Interaction with H(t)

No change to H(t) thresholding unless **Q1 conflict resolution** needs tie-break:

If `high_H` and `high_A_eff` but token is **canonicalizable insult** (lexicon or high H + leet pattern):

→ Force **Q1 canonicalize**, not Q2 DP.

Rule:

```
if high_H and high_A_eff and is_obfuscated_insult(token):
    quadrant = Q1
elif high_H and not high_A_eff:
    quadrant = Q3
...
```

---

## 5. Implementation design

### 5.1 Proposed module: `mechanism/biber.py`

```
biber.py
├── feature_lexicons.py    # static word lists per Biber feature
├── tagger.py              # tag_token(token, context) -> list[str]
├── boosts.yaml            # β_f per tag (Layer 3 may override)
└── apply_priors(routes, config) -> routes  # mutates A_effective
```

### 5.2 Rule-based tagger (no ML)

Advantages:

- Deterministic, auditable
- Zero inference cost
- Matches Arnold's interpretable feature analysis

Disadvantages:

- English-specific
- Misses contextual uses (disambiguation via simple context rules)

Example disambiguation:

| Token | Context | Tag |
|-------|---------|-----|
| `that` | before noun | demonstrative (B10) — low boost |
| `that` | complementizer | subordination (B21) — medium boost |
| `you` | after insult | still B07 but Q1/Q3 from H(t) dominates |

### 5.3 Config schema (`configs/triage-dp.yaml`)

```yaml
biber:
  enabled: true
  boosts:
    B07_second_person: 0.15
    B04_place: 0.20
    B05_time: 0.15
    B33_discourse: 0.20
  clamp_max: 1.0
  skip_if_quadrant_locked: false   # if Layer 1 lexicon already Q3
```

Layer 3 calibration may **learn** boost scales or disable weak features.

---

## 6. Worked examples

### 6.1 Benign post with strong author voice

**Text:** `"I was here yesterday and honestly the weather was lovely"`

| Token | A_raw | Tags | B(t) | A_eff | H(t) | Quadrant |
|-------|-------|------|------|-------|------|----------|
| I | 0.05 | B06 | +0.10 | 0.15 | ~0 | Q4–borderline Q2 |
| here | 0.08 | B04 | +0.20 | 0.28 | ~0 | **Q2** |
| yesterday | 0.06 | B05 | +0.15 | 0.21 | ~0 | **Q2** |
| honestly | 0.12 | B33 | +0.20 | 0.32 | ~0 | **Q2** |
| weather | 0.02 | — | 0 | 0.02 | ~0 | Q4 |

**Effect:** Layer 2 pushes situational/discourse tokens into Q2 even when individual occlusion drops are small—privacy gain on benign authorial posts without touching hate-neutral nouns.

### 6.2 Hate post with direct attack

**Text:** `"you are such a dummy"`

| Token | A_raw | Tags | A_eff | H(t) | Quadrant |
|-------|-------|------|-------|------|----------|
| you | 0.06 | B07 (+0.15) | 0.21 | 0.02 | Q2 if only A; but H low → Q2 |
| dummy | 0.05 | — | 0.05 | 0.45 | **Q3** (H dominates) |

**Critical:** High H(t) on `dummy` overrides B07 boost on `you` for the insult token. For `you`, low H(t) + boosted A_eff → Q2 rewrite (`ye` → generic) may occur—acceptable if hate signal remains on `dummy`.

Design check: monitor utility on 2nd-person hate constructions during Layer 3 calibration; raise `τ_H` for tokens adjacent to high-H neighbors if needed.

### 6.3 Obfuscated slur (Q1)

**Text:** `"what a d00fus"`

| Token | A_raw | Tags | A_eff | H(t) | Quadrant |
|-------|-------|------|-------|------|----------|
| d00fus | 0.22 | — | 0.22 | 0.41 | **Q1** |

Biber priors don't boost leet slurs; Q1 comes from joint H+A occlusion.

---

## 7. Relationship to Arnold's empirical findings

Arnold reports dpmlm **under-represents**:

- Place/time adverbials (~0.6–0.9 ratio vs human baseline)
- 2nd person pronouns (~0.9 ratio vs human)
- Discourse particles (~0.4 ratio)

TRIAGE-DP Layer 2 **pre-identifies** these as identity-heavy **before** dpmlm homogenizes them accidentally. Instead:

- **Q2 tokens** get **controlled** DP with explicit ε
- **Q3 hate tokens** in involved register stay protected

Research claim:

> dpmlm homogenizes register blindly; TRIAGE-DP **targets** register features that carry identity while preserving hate-evidence tokens identified by H(t).

Validate with **Burrows' Delta** on author corpus before/after—Layer 2+1 should reduce Delta more than uniform dpmlm at equal utility.

---

## 8. Evaluation and ablation

### 8.1 Ablation matrix

| Config | Layer 1 | Layer 2 | Expected |
|--------|---------|---------|----------|
| Baseline dpmlm | off | off | Register-blind homogenization |
| L1 only | on | off | Good hate protection; may miss weak A(t) tokens |
| L1+L2 | on | on | Better privacy at equal utility |
| L2 only (broken) | off | on | **Invalid** — boosts without routing |

### 8.2 Metrics specific to Layer 2

| Metric | Method |
|--------|--------|
| Feature retention rate | Frequency of B07/B04/B33 in output vs input |
| Burrows' Delta | Arnold methodology on author subset |
| Q2 token count | Should increase vs L1-only on stylometric posts |
| Utility F1 | Must not drop vs L1-only |

### 8.3 Failure mode: over-boosting

If β values too high:

- Benign `you` in hate posts pushed to Q2 → grammar damage
- Utility F1 drops on involved hate speech

**Mitigation:** Layer 3 learns β scaling; cap `B(t) ≤ 0.25` per token; disable boost when `H(t) > τ_H_neighbor` (adjacent to hate evidence).

---

## 9. Computational cost

| Component | Cost |
|-----------|------|
| Biber tagger | O(n) dictionary lookups per token — **microseconds** |
| Boost application | O(n × avg_tags) — negligible vs Layer 1 occlusion |

Layer 2 adds **no neural inference** in default configuration.

---

## 10. Testing strategy

### 10.1 Unit tests

- `you` → tag B07, boost applied
- `dummy` insult → no Biber boost (unless mis-tagged)
- A_eff clamped at A_max
- Audit log includes `biber_tags` array

### 10.2 Golden files

Fixed strings with expected tags and A_eff deltas documented in test fixtures.

---

## 11. Limitations

- English-centric lexicons; multilingual IA-HSD needs new feature inventories
- Rule tagger misses sarcasm and multi-token constructions
- Biber features correlate—double-counting if many tags stack (mitigate with `B(t) = max(β_f)` variant instead of sum)
- Stylometric priors encode **population-level** identity cues; may over-privatize AAVE or dialect markers—**fairness audit** required in deployment narrative (Layer 5)

---

## 12. References

- Arnold, *Differentially-Private Text Rewriting reshapes Linguistic Style* — register shift under DP
- Biber, *Variation across Speech and Writing* (1991); Biber (1995) feature list
- Egbert et al., CORE corpus — register diversity context
- Meisenbacher et al., DP-MLM — bidirectional rewriting behavior Arnold compares
- [`layer-01-cross-saliency-triage.md`](layer-01-cross-saliency-triage.md) — A(t), quadrants

---

## 13. Open design decisions

| Decision | Recommendation |
|----------|----------------|
| Boost aggregation | Sum with cap `B_max=0.30` per token |
| 1st person boost | Lower β than 2nd person (less identity in short posts) |
| Learned β_f | Layer 3 optional; start with hand defaults from Arnold Table 2 |
| Negative boosts on hate lexicon | Zero boost if lexicon hit or H(t) > τ_H |
