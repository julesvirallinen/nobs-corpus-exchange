"""Generate data/synthetic_dev.csv: 30-50 rows in the ID,Author,Text,HS schema.

Mild, fake content only. "Hate" rows (HS=1) use MILD insults and made-up
placeholder tokens registered in the test lexicon (configs/test.yaml) — never
real slurs. Covers the required edge cases: empty / very long Text, embedded
commas / double quotes / newlines, emoji, leetspeak + spaced obfuscation of
test-lexicon tokens, elongated words, non-ASCII, URLs and @mentions.

Run:  python scripts/make_synthetic.py
Deterministic — produces the same file every time.
"""

from __future__ import annotations

import csv
import os

OUT = os.path.join("data", "synthetic_dev.csv")

# (Author, Text, HS). Mild placeholders only.
ROWS = [
    # ── ordinary benign posts (HS=0) ───────────────────────────────────────
    ("alice", "I really enjoyed the documentary about deep sea creatures last night.", 0),
    ("bob", "Does anyone have a good recipe for sourdough bread that actually works?", 0),
    ("carol", "The bus was late again this morning but the weather was lovely.", 0),
    ("dave", "Just finished a great book about the history of cartography.", 0),
    ("erin", "My garden tomatoes are finally ripening and they taste amazing.", 0),
    ("frank", "Thinking about learning to play the cello this year, any tips?", 0),
    ("grace", "We hiked up the ridge and the view over the valley was incredible.", 0),
    ("heidi", "The new coffee place downtown has surprisingly good pastries.", 0),
    ("alice", "Spent the afternoon fixing my bike and now it rides like new.", 0),
    ("bob", "Anyone watching the meteor shower tonight? Skies look clear here.", 0),
    ("carol", "I think the city should add more bike lanes near the river.", 0),
    ("dave", "Learning a bit of pottery and my first bowl is hilariously lopsided.", 0),

    # ── mild "hate" posts (HS=1) using placeholders / mild insults ─────────
    ("grace", "Stop being such a dummy and read the instructions before you ask.", 1),
    ("frank", "What an absolute doofus, he parked across two spaces again.", 1),
    ("heidi", "Honestly that take is the work of a complete nitwit.", 1),
    ("alice", "You total zibber, you ruined the whole group project for everyone.", 1),
    ("bob", "Only a florbnax would believe that obvious scam, come on.", 1),
    ("carol", "Quit acting like a grumblefric and just apologise already.", 1),
    ("dave", "He is such a dummy, he never listens to anyone on the team.", 1),
    ("erin", "What a nitwit move, leaving the gate open with the dog outside.", 1),

    # ── edge cases ─────────────────────────────────────────────────────────
    ("alice", "", 0),  # empty Text
    ("carol", "Well, you know, that is, like, a whole lot of commas, right, yeah.", 0),  # commas
    ('dave', 'He said "you are a real dummy" and then just walked off.', 1),  # double quotes
    ("erin", "line one\nline two\nline three\nstill the same field", 0),  # newlines
    ("frank", "this is great \U0001F600\U0001F389 absolutely love it \U0001F525 best day", 0),  # emoji
    ("grace", "you total z1bb3r and d00fus, what were you even thinking", 1),  # leetspeak
    ("heidi", "that was sooo goooood and reallyyy niiice, loved it sooo much", 0),  # elongation
    ("alice", "café naïve résumé jalapeño über Москва Köln — non-ascii everywhere", 0),  # non-ascii
    ("bob", "check http://example.com/path?q=1 and @username thanks #cool stuff", 0),  # url+mention
    ("carol", "what an absolute g r u m b l e f r i c, spacing out the letters", 1),  # spaced obfuscation
    ('dave', 'hey @bob, this "florbn4x" post is sooo \U0001F525, see http://x.co now', 1),  # mixed
    ("erin", "n1tw1t behaviour again, honestly d u m m y energy all around", 1),  # leet + spaced
    ("frank", "Visit https://news.example.org, email me, or DM @frank_here ok?", 0),  # urls/mentions
]

# A very long benign Text (edge case).
_LONG = (
    "Okay so here is my very long and rambling review of the trip. "
    + "We started early and the road was quiet and pleasant. "
    * 40
    + "In the end it was a wonderful and uneventful weekend, ten out of ten."
)
ROWS.append(("grace", _LONG, 0))

# A couple more so several authors have both HS values and counts vary.
ROWS.extend([
    ("heidi", "You are being a dummy about this, please just double-check the math.", 1),
    ("bob", "The lecture on medieval bridges was unexpectedly fascinating today.", 0),
    ("frank", "Only a doofus leaves the milk out overnight, but here we are again.", 1),
    ("alice", "Planning a small picnic by the lake this weekend if it stays dry.", 0),
])


def main() -> None:
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["ID", "Author", "Text", "HS"])
        for i, (author, text, hs) in enumerate(ROWS, start=1):
            writer.writerow([f"r{i:03d}", author, text, hs])
    print(f"wrote {len(ROWS)} rows to {OUT}")


if __name__ == "__main__":
    main()
