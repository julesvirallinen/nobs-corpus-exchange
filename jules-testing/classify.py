#!/usr/bin/env python3
"""Finnish hate speech classifier — topic + category.

Usage:
    python classify.py --input data.csv --backend finbert
    python classify.py --input data.csv --backend poro --output results.csv
    python classify.py --input texts.txt --backend finbert

Input formats:
    CSV: must have a column named 'text'
    TXT: one text per line
"""

import argparse
import sys
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.progress import track

console = Console()


def load_input(path: str) -> list[str]:
    if path.endswith(".csv"):
        df = pd.read_csv(path)
        if "text" not in df.columns:
            console.print(f"[red]CSV must have a 'text' column. Found: {list(df.columns)}")
            sys.exit(1)
        return df["text"].dropna().tolist()
    else:
        with open(path, encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]


def make_backend(name: str, threshold: float = 0.6):
    if name == "finbert":
        from backends.finbert_backend import FinBERTBackend
        return FinBERTBackend(threshold=threshold)
    elif name == "poro":
        from backends.poro_backend import PoroBackend
        return PoroBackend()
    else:
        console.print(f"[red]Unknown backend: {name}")
        sys.exit(1)


def print_results_table(texts: list[str], results: list[dict]):
    table = Table(title="Hate Speech Classification Results", show_lines=True)
    table.add_column("Text", max_width=50, overflow="fold")
    table.add_column("Topic", style="cyan")
    table.add_column("Score", style="dim")
    table.add_column("Categories", style="yellow")

    for text, r in zip(texts, results):
        score = f"{r['topic_score']:.2f}" if r["topic_score"] is not None else "—"
        cats = ", ".join(r["categories"]) if r["categories"] else "none"
        table.add_row(text[:200], r["topic"], score, cats)

    console.print(table)


def save_results(texts: list[str], results: list[dict], path: str):
    rows = []
    for text, r in zip(texts, results):
        rows.append({
            "text": text,
            "topic": r["topic"],
            "topic_score": r["topic_score"],
            "categories": "|".join(r["categories"]),
            **{f"score_{k}": v for k, v in r.get("category_scores", {}).items()},
        })
    pd.DataFrame(rows).to_csv(path, index=False)
    console.print(f"[green]Results saved to {path}")


def main():
    parser = argparse.ArgumentParser(description="Finnish hate speech classifier")
    parser.add_argument("--input", required=True, help="Input CSV or TXT file")
    parser.add_argument(
        "--backend",
        choices=["finbert", "poro"],
        default="finbert",
        help="Model backend (default: finbert)",
    )
    parser.add_argument("--output", help="Save results to CSV file")
    parser.add_argument("--threshold", type=float, default=0.6, help="Category confidence threshold (default: 0.6)")
    args = parser.parse_args()

    texts = load_input(args.input)
    console.print(f"[bold]Loaded {len(texts)} texts[/bold] — backend: [cyan]{args.backend}[/cyan] — threshold: {args.threshold}")

    backend = make_backend(args.backend, threshold=args.threshold)

    results = []
    for text in track(texts, description="Classifying..."):
        results.append(backend.classify(text))

    print_results_table(texts, results)

    if args.output:
        save_results(texts, results, args.output)


if __name__ == "__main__":
    main()
