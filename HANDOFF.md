# Sensemaking — team handoff

This is the cover. It tells you what is in the package, where the project stands, what is decided, and what still needs a person. No compute figures appear here by design.

## What is in this package

- **README.md** — paste this whole file into a fresh assistant to spin up an instance that knows the entire project. It will open with a plain-language summary and goals tied to the New Democratic Pact for Europe, then the model and tech lock-ins, then the open items, and it will help you plan one thread at a time. This is the fastest way to onboard a teammate or get a working planning partner.
- **PRD.md** — the product requirements, revised to every decision since selection.
- **ROADMAP.md** — Now, Next, Later, with the clean sixty-hour build under Now.
- **HANDOFF.md** — this file.

## Where it stands, in five lines

We won the concept round of the Council of Europe Democracy Hackathon, PrivHSD challenge; the live build is 17 to 19 June in Strasbourg. Sensemaking reads online hate and manufactured discourse at the level of the conversation, never the person, and it does so for the New Democratic Pact's aim of countering disinformation while protecting free speech. We are rebuilding from a clean machine to current standards. The lead is discourse mapping and visualisation, multi-topic, with a timeline showstopper across an election period. The deliberation outputs are wanted but need their own plan; the election data and live source need research.

## Decision log

### Locked
- Name is **Sensemaking**; "Semantic Analysis Engine" dropped.
- Lead with **discourse mapping**, deliberation as the stance and the destination.
- **Never-track-author** architecture: author identity is never bound to a record, so there is nothing to strip and nothing to leak. An ingestion guard still excludes minors and private citizens before the model.
- **Multi-topic**, with subnarratives shown in **yellow** as their own channel. Check the yellow for contrast against the colour-blind-safe stance and severity scales.
- Severity base is the **TurkuNLP Finnish toxicity model**, mapping threat to threat and insult and identity-attack to derogation; dehumanisation and incitement stay declared gaps.
- **Local-first** development; git and local services; Docker Compose; a dev server only if a reason forces it.
- **Bridging is out of scope.**
- The full honesty floor: no enforcement, no identity field, no scraping, no scores while untrained, the machine-generated share stays a heuristic with its interval, "open" means open-weight and self-hostable.

### Reserved, criteria only
- The **local reads model** that turns classifier labels into readable context. Finnish quality first, must run locally, then cost. Poro 34B is out because it does not run locally. Shortlist: Poro 2 8B, a quantised Llama-3.1 8B, Viking 7B, a small Qwen.
- The **polarising-language threshold**, resolved by governance sign-off, not by editing a constant.

### Open, needs research
- **The 2024 election data and the live source.** Suomi24 ends at 2023, so it has no 2024 discourse, and scraping is off the table. Bluesky is the leading compliant candidate; avoid an X aggregate service; synthetic fixtures are the fallback. Live data is real people, so the ingestion guard matters most here. This gates the showstopper.

### Open, needs planning
- **Deliberation and counter-narrative impact.** Wanted and central to the vision, but the how, and how to make it genuinely change a debate, needs a dedicated planning track. The build leads with mapping; deliberation outputs are planned, not committed for the sixty hours.
- **The showstopper spec**, an election-period timeline whose exact content is being decided.
- **The primary user and first use case.**

## Team and lanes

Liisi Soroush, Democracy and Policy. Joonatan Myllys, Technologist. Arsham Soltani, Design and UX, and model, severity and data lead. Jules Uusinarkaus, Cybersecurity and dev-team lead. Proposed lanes: Arsham on model, severity, data; Jules on guardrails, security, infrastructure; Joonatan on engine and surface integration; Liisi on policy, citations, and the threshold sign-off.

## How to spin up an instance

Open README.md, copy the whole file, paste it into a fresh assistant. It primes itself and opens with the briefing. Use it to plan, to onboard, or to pressure-test a decision against the honesty floor.

## What needs a person now

1. Name an owner and a first session for the **deliberation-impact plan**.
2. Start the **2024-data and live-source research**; decide Bluesky or not, confirm the approved API.
3. Make a **provisional reads-model pick** so the local stack can be built.
4. Settle the **primary user** and lock the **showstopper** content.

The PRD and ROADMAP carry these forward as TO SET or RESERVED until they are answered. Nothing here invents a decision that was not made.
