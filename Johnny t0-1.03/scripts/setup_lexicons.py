"""Download a real, open hate/abuse lexicon to data/lexicons/hate_terms.txt.

The lexicon is NOT committed (the repo is public and lexicons contain slurs;
see .gitignore). Run once at setup:

    python scripts/setup_lexicons.py
    python scripts/setup_lexicons.py --url <your_lexicon_url>
    python scripts/setup_lexicons.py --from-file path/to/terms.txt

Default source: the open LDNOOBW English bad-words list (a profanity proxy).
The organiser may provide a curated hate lexicon — drop any newline-delimited
term list at data/lexicons/hate_terms.txt and spine mode will use it. One term
per line; blank lines and lines starting with '#' are ignored.

Sources / licences (documented, not bundled):
  * LDNOOBW "List of Dirty, Naughty, Obscene and Otherwise Bad Words" (CC-BY 4.0)
    https://github.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words
  * HurtLex EN (CC-BY-NC-SA) is another option for a richer hate lexicon.
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.request

DEST = os.path.join("data", "lexicons", "hate_terms.txt")
DEFAULT_URL = (
    "https://raw.githubusercontent.com/LDNOOBW/"
    "List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words/master/en"
)

# Terms common in IA-HSD / synthetic dev but absent from LDNOOBW.
IA_HSD_SUPPLEMENT = (
    "dummy", "doofus", "nitwit", "idiot", "moron", "stupid", "fool",
    "imbecile", "dimwit", "dumbass", "jackass", "bastard", "bitch",
)


def _write_terms(terms, dest=DEST):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    cleaned = []
    seen = set()
    for t in terms:
        t = t.strip()
        if not t or t.startswith("#"):
            continue
        low = t.lower()
        if low not in seen:
            seen.add(low)
            cleaned.append(low)
    with open(dest, "w", encoding="utf-8") as fh:
        fh.write("# Downloaded lexicon (NOT committed). One term per line.\n")
        for t in cleaned:
            fh.write(t + "\n")
    print(f"wrote {len(cleaned)} terms to {dest}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default=DEFAULT_URL)
    p.add_argument("--from-file", default=None,
                   help="use a local file instead of downloading")
    p.add_argument("--no-ia-hsd-supplement", action="store_true",
                   help="do not merge IA-HSD eval insult terms (dummy, idiot, …)")
    args = p.parse_args(argv)

    if args.from_file:
        with open(args.from_file, "r", encoding="utf-8") as fh:
            terms = fh.readlines()
    else:
        try:
            print(f"downloading lexicon from {args.url} ...")
            with urllib.request.urlopen(args.url, timeout=30) as resp:
                data = resp.read().decode("utf-8", errors="replace")
            terms = data.splitlines()
        except Exception as exc:
            print(
                f"ERROR: could not download lexicon: {exc}\n"
                f"Provide one manually at {DEST} (one term per line), or use "
                "--from-file / --url.",
                file=sys.stderr,
            )
            return 1

    if not args.no_ia_hsd_supplement:
        terms.extend(IA_HSD_SUPPLEMENT)
        print(f"merging {len(IA_HSD_SUPPLEMENT)} IA-HSD supplement terms")
    _write_terms(terms)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
