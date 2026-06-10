#!/usr/bin/env python3
"""Pull TurkuNLP/jigsaw_toxicity_pred_fi and export a sample CSV.

Usage:
    python load_dataset.py                          # 200 balanced samples -> dataset.csv
    python load_dataset.py --n 500 --out data.csv   # custom size and path
    python load_dataset.py --all                    # full train split (~160k rows)
"""

import argparse
import pandas as pd
from datasets import load_dataset
from rich.console import Console

console = Console()

LABEL_COLS = [
    "label_toxicity",
    "label_threat",
    "label_insult",
    "label_identity_attack",
    "label_obscene",
    "label_severe_toxicity",
]

THRESHOLD = 0.5


def hf_to_df(split) -> pd.DataFrame:
    df = split.to_pandas()
    # Binarize labels
    for col in LABEL_COLS:
        df[col] = (df[col] >= THRESHOLD).astype(int)
    df["any_toxic"] = df[LABEL_COLS].max(axis=1)
    return df[["text", "any_toxic"] + LABEL_COLS]


def balanced_sample(df: pd.DataFrame, n: int) -> pd.DataFrame:
    toxic = df[df["any_toxic"] == 1]
    clean = df[df["any_toxic"] == 0]
    half = n // 2
    s_toxic = toxic.sample(min(half, len(toxic)), random_state=42)
    s_clean = clean.sample(min(n - len(s_toxic), len(clean)), random_state=42)
    return pd.concat([s_toxic, s_clean]).sample(frac=1, random_state=42).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200, help="Number of samples (default: 200)")
    parser.add_argument("--out", default="dataset.csv", help="Output CSV path (default: dataset.csv)")
    parser.add_argument("--all", action="store_true", help="Export full train split")
    args = parser.parse_args()

    console.print("Loading [cyan]TurkuNLP/jigsaw_toxicity_pred_fi[/cyan]...")
    ds = load_dataset("TurkuNLP/jigsaw_toxicity_pred_fi")
    df = hf_to_df(ds["train"])

    console.print(f"  Total rows: {len(df):,} | Toxic: {df['any_toxic'].sum():,} ({df['any_toxic'].mean():.1%})")

    if args.all:
        out_df = df
    else:
        out_df = balanced_sample(df, args.n)
        console.print(f"  Sampled: {len(out_df)} rows (balanced toxic/clean)")

    out_df.to_csv(args.out, index=False)
    console.print(f"[green]Saved {len(out_df)} rows to {args.out}")
    console.print(f"\nRun classifier:\n  [bold].venv/bin/python classify.py --input {args.out} --backend finbert")


if __name__ == "__main__":
    main()
