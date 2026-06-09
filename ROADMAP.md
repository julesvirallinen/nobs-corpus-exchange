# Sensemaking, Roadmap (v1, post-meeting)

Anchored to the accepted concept's path: an architecture-first demonstration, then a calibration run and an independent privacy audit, leading to a first real-world test at the Finnish parliamentary election of 18 April 2027. Nothing here exceeds the proposal or its natural progression. No compute figures appear by design.

## Now: the clean sixty-hour Strasbourg build (17 to 19 June 2026)
Foundation, first
- Local-first setup: git plus local services. For the demo, a local virtual environment with pinned dependencies and a run script, not containers. A dev server only if a reason forces it.
- Define and freeze the seam schema, records with no author field. Commit fixtures on day one.
- Synthetic data is for fixtures and tests only, never a source; wire a small slice of compliant live data if feasible for the MVP.

Must finish
- Build the ingestion path that never tracks an author, with the ingestion guard that excludes minors and private citizens before any model, proven by negative tests that the protected flags never appear downstream.
- The multi-topic, author-agnostic analyst surface: cluster map, one topic's polarisation read, the severity gradient with two declared gaps, the machine-generated estimate with its interval, the closed deliberation set of three reframed poles, protect the debate, add context, counter a narrative, with sub-floor and excluded groups absent.
- Severity via the TurkuNLP Finnish toxicity model, mapped to threat and derogation, dehumanisation and incitement shown as gaps.
- The explainability view: the signals behind a reading, with the yellow subnarrative channel inside a topic.
- The showstopper: an election-period timeline. Strong candidate anchor, a meeting decision, the Finnish Ministry of Justice scenario across the 18 April 2027 vote. Exact content to set; depends on the data research below.

Should
- A minimal within-cluster subnarrative read.
- A small local model that refines classifier labels into a readable context line, once the model is picked.
- A signed, reproducible build with an open audit log.

Could
- A two-node encrypted-comparison proof of concept.
- A small slice of compliant live data, once the source is confirmed.

## Next: post-event hardening
- Fine-tune the team's own derogation and threat heads on the machine-translated Finnish toxicity data, validated on the native Finnish set; keep dehumanisation and incitement as declared gaps.
- Plan and prototype the deliberation and counter-narrative outputs, with a real account of how they change a debate.
- Commission an independent privacy audit.
- Mature the encrypted comparison from a proof of concept toward a working two-party flow.
- Package the open, reproducible build so an institution can run and verify it, containerising it at this stage rather than for the demo.
- Open the Faktabaari conversation as a possible first collaboration partner; direct contact in hand, not confirmed. Owner Liisi.

## Later: toward 18 April 2027 and beyond
- Full multi-institution comparison across borders, with no central record.
- The subnarrative taxonomy at scale, with the additional author-agnostic cluster signals, platform diversity, temporal spike, source type.
- Sustained platform-data access through the EU Digital Services Act, Article 40.
- Extend the verifiability toward post-quantum methods.
- First real-world test at the Finnish parliamentary election, 18 April 2027.

## Reserved decisions and the criteria for each
- The local reads model: Finnish quality first, must run locally, then cost. Poro 34B is out, it does not run locally. Owner: Arsham. RESERVED.
- Polarising-language threshold: governance sign-off, not editing the constant. Owner: Liisi. RESERVED.

## Open, needs research
- The 2024 election-period data and the live source. Suomi24 ends at 2023; scraping is out. Bluesky leads as a compliant public API; avoid an X aggregate service; synthetic fixtures are the fallback. Live data is real people, so the ingestion guard matters most here. Gates the showstopper.

## Open, needs planning
- Deliberation and counter-narrative, and how to make them genuinely change conversational debate. Wanted, and the destination; the build leads with mapping until the plan exists.
- The MVP and showstopper anchor, a meeting decision, not one person's call. Strong candidate: the Ministry of Justice election scenario, seen as the most high-impact and ambitious goal.
- The showstopper content and the primary user.

## Risks and honesty lines that must not slip under deadline pressure
- No scores or accuracy claims while the classifier is untrained; report only trained heads, with intervals.
- The machine-generated share stays a heuristic with its interval, never a hard count.
- Author identity is never tracked; no identity field is reintroduced for convenience.
- Minor and private-citizen exclusion stays at ingestion, never a display switch, and matters most for any live data.
- Deliberation-only output; no enforcement path is added, and the tool never publishes to any platform.
- "Open" stays open-weight and self-hostable, never FOSS-only or rooted in vendor hardware.
- Live data only from a compliant API, never scraping; synthetic data for testing only, never a source.
- Citations verified before any public claim.
