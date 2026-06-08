# Sensemaking, Product Requirements Document (v1, post-meeting)

Source of truth: the accepted PrivHSD concept summary, updated to the decisions taken since selection. Includes only what the proposal commits to, plus honest progress from it. Direction-dependent fields are marked TO SET. Reserved decisions are marked RESERVED.

Status legend, reset for the clean rebuild: nothing is BUILT, since the one-day prototype is being rebuilt from zero; only designs and invariants carry. [BUILD] buildable in the sixty-hour window; [ROADMAP] after the event. Priority: must, should, could, later.

## 1. Definition and problem
Sensemaking (short handle NoBS) is a privacy-preserving way to read online hate and manufactured discourse. It reads the conversation, not the person. The problem: hate-speech detection has become a re-identification and surveillance machine, and the open question is how to read harm without endangering the people in the data. It serves the New Democratic Pact for Europe's aim of countering disinformation while protecting free speech.

## 2. Users and jobs
Users named in the proposal: journalists, monitors, and institutions. Top jobs: see where a debate has polarised and turned hostile; estimate how much of a debate looks synthetically inflated; expose a manufactured consensus without surveilling anyone.
TO SET: the single user and job to serve first.

## 3. Goals and non-goals
Goals: an honest, author-agnostic reading of group-level discourse, multi-topic, runnable and inspectable by an institution, leading with discourse mapping and visualisation.
Non-goals, hard: no enforcement output of any kind; no identification or re-identification of any person; no social-media scraping; no scores or accuracy claims while the classifier is untrained.

## 4. Hard constraints and guarantees (public promises; must stay true)
- Author-agnostic by construction; author identity is never tracked, so there is no field to strip and nothing to leak. [BUILD, must]
- No finding about a group smaller than ten distinct items, enforced as a constructor invariant. [BUILD, must]
- Minors and private citizens excluded at ingestion, so no setting can reveal them. [BUILD, must]
- Outputs limited to three: add context, counter a narrative, none. No enforcement path. [BUILD, must]
- Machine-generated share shown as a figure with its interval, never a hard count, never a "we detect AI" capability. [BUILD heuristic, must]
- Severity is a four-dimension schema; derogation and threat calibratable, dehumanisation and incitement declared gaps shown without a score. [BUILD schema and base model; own calibration ROADMAP]
- Open-weight, self-hostable, no proprietary or black-box dependency in a default deployment; open cryptography on commodity hardware, no vendor as trust root. [BUILD, must]
- Data is licensed plus synthetic, never scraping. [BUILD, must]

## 5. Functional requirements
Engine
- Ingest content only; never bind an author to a record; emit per-item records with a salted opaque id and no author field. [BUILD, must]
- Cluster public discourse by topic and stance, never by account, multi-topic: sentence-transformers, UMAP, HDBSCAN, BERTopic. [BUILD, must]
- Map competing subnarratives at the group level, shown as the yellow channel. [BUILD, should] (section 6)
- Estimate the machine-generated share with its interval. [BUILD heuristic, must]
- Report a pattern only when several independent author-agnostic signals agree, the corroboration gate. [BUILD, should]

Guardrails
- Ingestion-time exclusion of minors and private citizens, before any model. [BUILD, must]
- k of ten floor. [BUILD, must]
- No central pool; protected group-level summaries compared over encrypted shares. [ROADMAP; a two-node encrypted proof is BUILD, could]
- Closed deliberation output set, no enforcement path. [BUILD, must]

Experience
- For any cluster, show how it reached its reading and which signals corroborated it. [BUILD, must]
- Show what it does not know: the two declared gaps and the estimate's interval. [BUILD, must]
- Stance and severity on separate colour-blind-safe scales; subnarratives in yellow as a distinct channel, checked for contrast against those scales. [BUILD, must]
- An open, reproducible build an institution can run, local-first. [BUILD packaging, should; full runnability ROADMAP]

Severity model
- Use the TurkuNLP Finnish toxicity model as the local base; map its threat output to threat and its insult and identity-attack outputs to derogation; keep dehumanisation and incitement as declared gaps. [BUILD, must]
- A small, locally runnable, Finnish-capable model refines classifier labels into a readable context line. RESERVED, provisional pick pending. [BUILD once locked]
- Fine-tune the team's own heads on the machine-translated Finnish toxicity data, validated on the native Finnish set. [ROADMAP]

## 6. Subnarrative and discourse-mapping layer
The proposal commits to mapping competing subnarratives. Progress path, drawing on the NDI monitoring model: a structured narrative taxonomy above the severity schema, plus additional author-agnostic, k-floor-safe cluster signals, platform diversity, temporal spike or concentration, source-type distribution, feeding the corroboration gate. Multi-topic is in scope; the subnarrative read renders in yellow. [BUILD a minimal within-cluster read, should; taxonomy at scale ROADMAP]

## 7. Strasbourg demo acceptance criteria
A judge must be able to:
- See an author-agnostic, multi-topic cluster map and open one topic's reading. [BUILD, must]
- See the polarisation read, the severity gradient with two dimensions marked as gaps, and the machine-generated estimate with its interval. [BUILD, must]
- See that minors, private citizens, and sub-floor groups never appear, and that there is no enforcement control. [BUILD, must]
- See the corroboration behind a reported pattern, and the yellow subnarratives within a topic. [BUILD, should]
- The showstopper: an election-period timeline. Exact content TO SET; data source under research.

## 8. Open questions and reserved decisions
- The local reads model, Finnish quality first, runs locally. RESERVED.
- Governance sign-off on the polarising-language threshold. RESERVED.
- The 2024 election-period data and the live source, Bluesky leading, synthetic fallback, scraping excluded. OPEN, research.
- Deliberation and counter-narrative outputs and their real-world impact. OPEN, planning; the build leads with mapping.
- Primary user, first use case, and the showstopper content. TO SET.
