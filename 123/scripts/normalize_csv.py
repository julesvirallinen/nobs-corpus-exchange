#!/usr/bin/env python3
"""Normalize CSV columns to ID,Author,Text,HS for SPINE wrapper contract."""

from __future__ import annotations

import argparse
import csv
import sys

REQUIRED = ["ID", "Author", "Text", "HS"]
ALIASES = {
    "id": "ID",
    "author": "Author",
    "text": "Text",
    "hs": "HS",
}


def normalize(in_path: str, out_path: str) -> int:
    with open(in_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames:
            print("ERROR: empty CSV", file=sys.stderr)
            return 1
        lower = {c.lower(): c for c in reader.fieldnames}
        rows = []
        for row in reader:
            out = {}
            for req in REQUIRED:
                key = lower.get(req.lower())
                if not key or row.get(key) is None:
                    print(f"ERROR: missing column for {req}", file=sys.stderr)
                    return 1
                out[req] = row[key]
            rows.append(out)
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=REQUIRED)
        writer.writeheader()
        writer.writerows(rows)
    print(f"OK {len(rows)} rows -> {out_path}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_path", required=True)
    p.add_argument("--out", dest="out_path", required=True)
    return normalize(p.parse_args().in_path, p.parse_args().out_path)


if __name__ == "__main__":
    raise SystemExit(main())
