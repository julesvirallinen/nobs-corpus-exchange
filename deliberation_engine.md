# The deliberation engine

A vivid technical walkthrough, written to plan how NoBS becomes useful and transformative for conversational debate. Plain language, honest about what is locked, what is buildable, and what is still a bet.

## What a debate looks like to the engine

To a person, a heated thread is a wall. Hundreds of voices, all shouting past each other, and somewhere inside it a real disagreement that nobody can see anymore. You scroll, you pick a side, you leave. The structure is invisible, so the debate cannot move.

To NoBS, the same thread is a shape. Not a list of people, because it never learns who anyone is. A landscape of arguments: where the camps are, where they harden into hostility, which claims are repeated by many independent voices and which are repeated by a suspiciously coordinated few. The engine's whole job is to turn the wall into a map, then hand the map to a human who can act on it with words rather than bans.

## How it reads

Everything starts by *not* knowing things. Content comes in, the author does not. There is no account, no handle, no identity field, because none is ever bound to a record. What survives ingestion is the text and an opaque item id. Minors and private citizens are removed at this boundary, before any model sees the data, so there is nothing to reveal later. This is the quiet foundation: the engine reads the conversation because it structurally cannot read the person.

Then the sense-making begins.

Each item becomes a vector, a point in a meaning-space where arguments that say similar things land near each other regardless of wording. That space is too high-dimensional to see, so it is compressed until the structure becomes legible, and then clustered by density. Out of the noise, camps appear: not because we told the system how many to expect, but because the arguments themselves clump. A topic model names them, so a cluster stops being "region 4" and becomes "the claim that the reform is a foreign plot." This is the discourse map, and it is the lead.

On top of the map sit the readings. Aggregation rolls everything up to the group, never the individual, and refuses to say anything about a group smaller than ten distinct items. The polarisation rule fires when two camps each hold a real share of the conversation and the temperature between them is high. A severity gradient grades the hostility, and here the engine is precise about its own limits: derogation and threat are calibratable, and now genuinely so, because a licensed Finnish signal exists for both. Dehumanisation and incitement are shown as declared gaps, marked but never scored, because faking them would be the one unforgivable thing. A separate estimate reports how much of the conversation looks machine-generated, always as a figure with its interval, never as a hard count and never as a claim to detect AI. And a corroboration gate holds everything back until several independent author-agnostic signals agree, so a single weak cue never becomes a headline.

The result is not a verdict. It is a read you can interrogate: here is the shape, here is how confident I am, and here is exactly what I do not know.

## The deliberation move

This is where most tools reach for a hammer. They detect something bad and they remove it, flag it, refer it. NoBS does not own a hammer. Its entire output space is three moves: add context, counter a narrative, or do nothing. More speech, or no action. There is no takedown path in the code, not as a setting, not as a feature behind a flag.

That constraint is not timidity, it is the product. A debate is not improved by deleting half of it; it is improved when the people in it can see what is happening and respond. So the engine's last step is to route a reading to the response that fits: a manufactured consensus gets context that exposes it, a hardening narrative gets a counter that a human can voice, an ordinary disagreement gets left alone. The engine proposes; a journalist, a moderator, an institution decides and speaks. The human is not a rubber stamp on an automated decision, because there is no automated decision to stamp.

## Why this could be transformative for conversational debate

The bet is simple. Most of what poisons online debate is not that bad arguments exist; it is that nobody can see the structure in time to respond well. NoBS attacks the invisibility.

It makes disagreement legible. A moderator looking at the map sees the actual fault lines, not a firehose, and can intervene where it matters instead of playing whack-a-mole on individual posts.

It exposes manufactured consensus. The thing that breaks a real debate is the impression that "everyone thinks X." When the machine-generated estimate and the corroboration gate show that the impression rests on a thin, coordinated layer, the spell breaks, and it breaks with a figure and an interval attached, not an accusation.

It scaffolds counter-speech instead of silence. Counter-speech is the slow, hard, correct answer to bad speech, and it usually loses because it is unscaffolded and exhausting. An engine that hands a deliberator the shape of the narrative and a place to answer it lowers the cost of the right move.

It builds a shared, verifiable picture. In an adversarial debate the first casualty is common ground about what is even being said. A read that every side can run, inspect, and reproduce gives them one honest thing to argue from.

And it is humble on purpose. It shows its gaps and its intervals. In a domain full of black boxes that claim certainty, a tool that says "I do not measure incitement, and here is my margin of error" is more trustworthy, not less, and trust is the only currency that lets an institution actually deploy it on real debate.

The unlock under all of this is the privacy. The reason institutions cannot use most discourse tools on live debate is that the tools surveil. NoBS does not. It never holds an identity, never reports a small group, excludes minors and private citizens at the door. That is what makes it usable on the conversations that matter, not just on sanitised samples.

## The lines that make it real

The transformative version still lives inside the locked floor, and that is the point. No enforcement output, ever. Group-level only, k of ten or nothing. No identity field anywhere. Two severity dimensions stay declared gaps until honest data exists. Open weights, self-hostable, open crypto on commodity hardware, no vendor as the trust root. The machine-generated share stays a heuristic.

A few things are deliberately *not* in this engine yet, and saying so is part of being honest about it. Bridging-based ranking, the technique that surfaces statements which win agreement across camps, is an adjacent and tempting idea, and it is out of scope unless the team re-decides it, because it was not in the accepted proposal. The trained derogation and threat heads are now feasible in the window thanks to the Finnish toxicity resources, but they are reportable only after validation on native Finnish text. The generative reads depend on the reserved open-weight model, which is chosen on Finnish quality first.

## Where the design bets are

These are the choices that decide whether the engine is merely correct or actually transformative, and they belong to the team. They are framed as questions, not answers.

The showstopper. The most memorable version is probably the manufactured-consensus reveal followed by the deliberation responses: show a thread that looks like a landslide, show that the landslide is thin and coordinated, then show the context move that pops it, all without naming a soul. That is a candidate, not a decision.

The within-cluster subnarrative read. Discourse mapping is the lead, so the demo has to show mapping, not just clustering: the named subnarratives inside a topic, with the author-agnostic signals that corroborate each. This is the difference between a pretty cluster plot and an instrument a journalist would trust.

The experience that lets a human act. The read is only transformative if a moderator can move from "here is the shape" to "here is what I will say" in one step. The interface, not the model, is where deliberation either happens or does not.

The threshold governance. The polarisation threshold is resolved by sign-off, not by editing a constant, because the moment a person can quietly tune what counts as "too hostile" is the moment the tool stops being neutral.

Build the engine that reads honestly and shows its work, keep the hammer out of its hands, and let humans do the deliberating. That is the whole bet.
