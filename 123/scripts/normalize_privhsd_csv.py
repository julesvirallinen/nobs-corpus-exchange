#!/usr/bin/env python3
"""Map lowercase reddit CSV -> PrivHSD contract (ID, Author, Text, HS)."""

from __future__ import annotations

import argparse
import csv
import sys


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_path", required=True)
    p.add_argument("--out", dest="out_path", required=True)
    args = p.parse_args()

    with open(args.in_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)

    out_rows = []
    for row in rows:
        out_rows.append({
            "ID": row.get("ID") or row.get("id", ""),
            "Author": row.get("Author") or row.get("author", ""),
            "Text": row.get("Text") or row.get("text", ""),
            "HS": row.get("HS") or row.get("hs", ""),
        })

    fieldnames = ["ID", "Author", "Text", "HS"]
    with open(args.out_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)

    print(f"OK {len(out_rows)} rows -> {args.out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
