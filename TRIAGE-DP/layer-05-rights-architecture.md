# Layer 5 — Rights-Based Architecture

Detailed design document for **TRIAGE-DP**. See the overview in [`../TRIAGE-DP.md`](../TRIAGE-DP.md). Technical layers 1–4 are in companion documents in this folder.

Layer 5 is not a separate algorithmic pass—it is the **normative and operational framing** that connects TRIAGE-DP to PrivHSD evaluation criterion **Human Rights-Centered Innovation**, the Council of Europe hate speech context, and trustworthy deployment narrative for judges and publication.

---

## 1. Purpose and role in the stack

| Layer | Answers |
|-------|---------|
| 1–4 | **How** to sanitize text (algorithm) |
| **5** | **Why** this design respects human rights and societal needs; **what** operators and affected users can inspect |

Layer 5 deliverables:

1. **Rights-based architecture document** (research note section)
2. **Operational policies** encoded in system design (author-blindness, audit logs)
3. **Limitations and impact reflection** (required by evaluation criteria)
4. **Judge/demo narrative** linking CoE hate speech mandate to IA-HSD

Without Layer 5, TRIAGE-DP risks appearing as a pure ML engineering entry—strong on execution, weak on **human-centered** and **impact** criteria.

---

## 2. Normative foundation

### 2.1 Council of Europe — combating hate speech

The CoE *Recommendation on combating hate speech* (2018) (`resources/CoE_2018_Recommendation_on_combating_hate_speech.pdf`) emphasizes:

- Hate speech harms individuals and democratic participation
- Responses must balance **effective counter-speech and moderation** with **rule of law and fundamental rights**
- States and platforms need **proportionate, transparent** measures

TRIAGE-DP aligns with the **dual need**:

| Societal need | TRIAGE-DP response |
|---------------|-------------------|
| Detect and address hate speech | Preserve HSD utility (Layer 1 H(t), Q3/Q1) |
| Protect individuals from harm | Do not publish re-identifiable toxic speech corpora |
| Avoid surveillance expansion | Strip authorship signal; identity-agnostic moderation |
| Accountability | Per-token and sentence-level audit logs |

### 2.2 Identity-agnostic moderation principle

**Principle:** A moderation system should determine **whether content violates norms**, not **who wrote it in a recoverable form**.

| Traditional pipeline | Risk | IA-HSD / TRIAGE-DP |
|---------------------|------|---------------------|
| Collect raw posts + usernames | Builds stylometric + identity linkage | Sanitize before aggregation |
| Share "anonymized" PII-redacted text | PII ≠ identity; stylometry remains | Target stylometry explicitly |
| Train HSD on raw user history | Surveillance coupling | Train on privatized text |

**Hackathon framing (from `resources/hackathon_prompt.md`):**

> Innovate a new HSD method that operates **agnostically to the author's identity**.

Layer 5 makes this principle **explicit in architecture**, not an accidental side effect of dpmlm noise.

### 2.3 Privacy as anti–re-identification (not anti-moderation)

PrivHSD privacy metric = **re-identification attacker success**, not GDPR-style PII removal alone.

Rights implication:

- **Protecting authors** of hateful content is not endorsement of hate—it reflects **data minimization** for anyone whose speech is processed
- Bystanders, targets, and quoted individuals also appear in posts; stylometric leakage can harm **non-authors** mentioned in context
- Shared research corpora should not become **authorship attribution benchmarks** without consent

---

## 3. Architecture principles (technical embodiment)

### 3.1 Principle: Data minimization by design

```
Raw CSV row:  ID, Author, Text, HS
                    │
Mechanism sees:      Text only
                    │
Output changes:      Text only (sanitized)
Preserved byte-for-byte: ID, Author, HS fields in wrapper contract
```

**Author field preserved in CSV** for organizer evaluation harness only—the **sanitization function** cannot access it (enforced by API and tests).

Reference: `Johnny t0-1.03` wrapper contract in PKG-INFO.

### 3.2 Principle: Separation of moderation utility and identity

| Concern | Component | Sees Author? |
|---------|-----------|--------------|
| Sanitization | `mechanism/*` | **No** |
| Evaluation | `harness/evaluate.py`, `reident.py` | **Yes** (harness only) |

This separation is a **rights-relevant boundary**: production deployment would not ship Author to the sanitization service.

### 3.3 Principle: Explainability and auditability

Every sanitization decision is logged:

| Log field | Rights function |
|-----------|-----------------|
| `original` / `replacement` | Show what changed |
| `H`, `A`, `quadrant` | Explain trade-off reasoning |
| `epsilon` | Show formal privacy spend |
| `reason` | Human-readable justification |
| `biber_tags` | Linguistic interpretability (Layer 2) |

**Demo script for judges:**

1. Show raw hate post with obfuscated slur
2. Show log: Q1 canonicalize `d00fus` → `dummy` (utility + privacy spelling)
3. Show Q2 rewrite of author tic "honestly" → "really" (privacy)
4. Show Q3 skip on `moron` (utility preserved)
5. Show HSD still positive; re-ID accuracy dropped

Transparency supports **human oversight**—moderators or researchers can audit automated sanitization.

### 3.4 Principle: No closed-box external processing of sensitive text

TRIAGE-DP uses **local open-weight models** only (MLM, hate classifier, optional local paraphraser).

| Approach | Rights concern | TRIAGE-DP |
|----------|----------------|-----------|
| GPT-4o API sanitization | Sensitive hate speech sent to third party | **Rejected** (case study TO negative) |
| Black-box PrivRewrite API | Same | Layer 4 local only |
| On-device DP-MLM | Data stays local | **Default path** |

Aligns with adaptive anonymization literature critique of closed-model processing of sensitive data (Loiseau et al.).

### 3.5 Principle: Honest privacy claims

Narrow claim (from SPINE / DP-MLM):

- ε-DP applies to **exponential-mechanism selection steps** (token and sentence)
- Normalization and canonicalization = **empirical obfuscation**, no formal ε
- **No** document-level DP guarantee

Rights benefit: **avoid over-promising** privacy to users or regulators—a form of procedural honesty required for trustworthy AI narratives.

---

## 4. Mapping to hackathon evaluation criteria

From `resources/evaluation criteria.png`:

### 4.1 Problem understanding

Layer 5 contribution:

- Document IA-HSD as distinct from PII anonymization and federated training
- Explain stylometry threat with re-ID probe methodology
- Cite literature table (TRIAGE-DP.md §2) in research note

### 4.2 Human rights-centered innovation

Layer 5 contribution:

- Identity-agnostic moderation principle (§2.2)
- Explainability via audit logs (§3.3)
- Usability: privacy dial from Layer 3 (operators don't tune ε manually)
- Understandable demo narrative (§6)

### 4.3 Execution and feasibility

Layer 5 does not add code paths—it **documents** how execution serves rights goals:

- Open, reusable repo (Apache-2.0 SPINE lineage)
- Reproducible configs
- Tests enforcing Author isolation

### 4.4 Impact and alignment

Layer 5 contribution:

- Societal need: moderation datasets without surveillance tooling
- Limitations section (§7)—required explicitly by criteria
- Follow-ups: multilingual fairness, community review of sanitization policies

---

## 5. Stakeholder perspectives

### 5.1 Platform moderators

**Need:** Reliable hate detection on shared corpora without storing identifiable rants linked to real identities.

**TRIAGE-DP offers:** Privatized text with preserved hate signal + audit trail for appeals ("why was this label still valid after sanitization?").

### 5.2 Researchers

**Need:** Publishable hate speech datasets with reduced re-identification risk.

**TRIAGE-DP offers:** Sanitized release + documented threat model + TO metric—not perfect anonymity, but **quantified** privacy–utility trade-off.

### 5.3 Authors of processed content (including hateful speech)

**Tension:** Hate speech causes harm; authors have reduced claim to privacy in some legal frames.

**TRIAGE-DP position (document carefully):**

- Sanitization reduces **identifiability**, not **accountability** in operational moderation (platforms may still act on account-level signals outside this dataset release)
- Dataset release context assumes **research/moderation pipeline**, not public shaming of individuals
- IA-HSD prevents released corpora from training **stylistic surveillance** models

### 5.4 Targets of hate speech

**Need:** Hate detected and removed; targets not re-victimized via dataset leaks linking them to posts.

**TRIAGE-DP:** Utility preservation (Q3) keeps detection effective; privacy axis reduces collateral identity leakage in surrounding context (place adverbials, community markers in Layer 2 Q2).

---

## 6. Demo and presentation script (judges)

### 6.1 Three-panel demo

| Panel | Content |
|-------|---------|
| **A — Problem** | Show TO formula; explain wrong baselines (Presidio, naive GPT) from case study |
| **B — Method** | Walk one post through Layer 1 quadrants with log |
| **C — Impact** | Before/after re-ID accuracy + HSD F1; rights narrative §2 |

### 6.2 Talking points

1. "We don't ask *who wrote this* to sanitize—we ask *what carries hate* vs *what carries identity*."
2. "Every token decision is logged with dual saliency scores—moderation remains accountable."
3. "Formal DP applies where we claim it; we don't overstate privacy."
4. "Goal is shared hate speech research **without** building author fingerprint datasets."

### 6.3 Visual assets

- Trade-off diagram from `resources/Trade-off.png`
- Quadrant grid from Layer 1 doc
- Pareto plot from Layer 3 calibration
- CoE citation one-liner

---

## 7. Limitations and follow-ups (impact section)

Required for evaluation criterion 4. Document prominently in research note.

### 7.1 Technical limitations

| Limitation | Impact | Mitigation path |
|------------|--------|-----------------|
| Per-token ε, no full document DP | Residual leakage possible | Layer 4 hard rows; future composition analysis |
| English Reddit focus | Global South languages underserved | Multilingual IA-HSD (REACT FL-HSD literature) |
| Hate classifier bias | Utility axis may under-protect dialect | Fairness audit; ensemble probes |
| Stylometric priors (Layer 2) | May over-privatize dialect markers | Community review; reduce β on contested features |
| Local TO ≠ official score | Ranking uncertainty | Holdout honesty; qualitative examples |
| Occlusion cost | Excludes low-resource deployers | Cache; tiered probes |

### 7.2 Rights-related limitations

| Limitation | Discussion |
|------------|------------|
| Sanitized hate still hateful | Sanitization ≠ endorsement; content remains offensive by design for utility |
| Imperfect anonymity | Re-identification risk reduced, not eliminated—communicate responsibly |
| Author field in eval CSV | Organizer harness artifact; not production pattern |
| Moderation automation | Human review still needed; TRIAGE-DP is dataset tool, not autonomous censor |

### 7.3 Follow-up research

- Fairness-aware triage (adjust τ_H, τ_A by demographic parity constraints)
- Community participatory definition of protected utility spans
- Legal analysis under GDPR / CoE framework for sanitized hate corpora
- Adaptive attacker co-evolution (DP-MLM adaptive setting, Meisenbacher et al.)
- Publication with TUM mentor alignment

---

## 8. Research note outline (Layer 5 sections)

Suggested structure for hackathon write-up:

1. **Introduction** — IA-HSD problem; CoE context
2. **Threat model** — stylometry attacker; HSD utility
3. **Method** — Layers 1–4 summary + diagram
4. **Rights-based architecture** — this document §2–3
5. **Experiments** — TO, ablations, baselines
6. **Human-centered design** — privacy dial, audit logs, demo
7. **Impact and limitations** — §7
8. **Conclusion** — moderation without surveillance tooling

Target length: 4–8 pages.

---

## 9. Policy and governance hooks (optional extensions)

Not required for hackathon MVP but strengthen publication:

| Hook | Description |
|------|-------------|
| **Data release policy** | Only privatized Text field published; Author never in public set |
| **Audit log retention** | JSONL alongside release for reproducibility |
| **Red team protocol** | Manual re-ID attempt on sanitized sample |
| **Ethics checklist** | Offensive content warning; no real slurs in public repo tests |

Reference SPINE safety note: synthetic/test lexicons only in git; real lexicons git-ignored.

---

## 10. Comparison to alternative "rights" framings

| Framing | Weakness for PrivHSD | TRIAGE-DP Layer 5 |
|---------|---------------------|-------------------|
| "We use differential privacy" | Technical jargon; may over-promise | Narrow ε claim + empirical steps labeled |
| "We anonymize data" | Implies PII removal | Explicit stylometry threat model |
| "We use federated learning" | Training privacy ≠ release privacy | Inference-time sanitization focus |
| "We don't store usernames" | Usernames ≠ stylometry | Identity-agnostic **text** processing |

---

## 11. Checklist before submission

- [ ] Research note includes IA-HSD definition and CoE citation
- [ ] Demo shows audit log with H, A, quadrant
- [ ] Limitations section lists §7.1 and §7.2 items
- [ ] No false claim of document-level DP
- [ ] Author-blind mechanism verified by tests
- [ ] Impact paragraph: who benefits, who remains at risk
- [ ] Follow-ups listed (multilingual, fairness, legal)

---

## 12. References

- Council of Europe (2018), *Recommendation on combating hate speech*
- `resources/hackathon_prompt.md` — identity-agnostic HSD goal
- `resources/evaluation criteria.png` — Human Rights-Centered Innovation
- `resources/case study.png` — baseline failures
- Loiseau et al., *Adaptive Text Anonymization* — task-conditioned privacy–utility framing
- Meisenbacher et al., DP-MLM — honest DP scope
- Ye et al. / NAACL 2025 — federated HSD (contrast training vs release privacy)
- Companion docs: [`layer-01`](layer-01-cross-saliency-triage.md) through [`layer-04`](layer-04-sentence-level-em.md)

---

## 13. Summary

Layer 5 articulates **why TRIAGE-DP is a human-rights-conscious moderation infrastructure**, not merely a TO optimization:

- **Identity-agnostic** by architectural separation
- **Transparent** via dual-saliency audit logs
- **Local and accountable** processing
- **Honest** privacy claims
- **Impact-aware** limitations and follow-ups

Judges evaluating "Human Rights-Centered Innovation" and "Impact and Alignment" should find this layer in the research note and demo—not only in technical appendices.
