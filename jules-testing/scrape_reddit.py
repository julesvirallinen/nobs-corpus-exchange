#!/usr/bin/env python3
"""Scrape Finnish Reddit content for hate speech analysis.

Usage:
    python scrape_reddit.py --client-id YOUR_ID --client-secret YOUR_SECRET
    python scrape_reddit.py --client-id ID --client-secret SECRET --limit 500 --out reddit_fi.csv
    python scrape_reddit.py --client-id ID --client-secret SECRET --subs Suomi Finland maahanmuutto
"""

import argparse
import pandas as pd
import praw
from rich.console import Console
from rich.progress import track

console = Console()

DEFAULT_SUBS = [
    "Suomi",         # main Finnish sub, mix of Finnish and English
    "Finland",       # English-language but Finnish topics
    "maahanmuutto",  # immigration — higher signal for hate speech
    "suomi24",       # mirror of Suomi24 discussions
]

# Sort options — "hot" for current, "new" for recent, "controversial" for heated posts
SORT_OPTIONS = ["controversial", "hot", "new", "top"]


def scrape_sub(reddit, sub_name: str, limit: int, sort: str) -> list[dict]:
    try:
        sub = reddit.subreddit(sub_name)
        getter = getattr(sub, sort)
        rows = []
        for post in getter(limit=limit):
            if post.selftext and post.selftext != "[removed]" and post.selftext != "[deleted]":
                rows.append({
                    "text": post.selftext[:1000],
                    "source": f"r/{sub_name}",
                    "type": "post",
                    "score": post.score,
                    "url": f"https://reddit.com{post.permalink}",
                })
            # Also grab top-level comments
            post.comments.replace_more(limit=0)
            for comment in post.comments[:10]:
                if comment.body and comment.body not in ("[removed]", "[deleted]"):
                    rows.append({
                        "text": comment.body[:1000],
                        "source": f"r/{sub_name}",
                        "type": "comment",
                        "score": comment.score,
                        "url": f"https://reddit.com{post.permalink}",
                    })
        return rows
    except Exception as e:
        console.print(f"[yellow]Warning: r/{sub_name} failed: {e}")
        return []


def main():
    parser = argparse.ArgumentParser(description="Scrape Finnish Reddit for hate speech analysis")
    parser.add_argument("--client-id", required=True)
    parser.add_argument("--client-secret", required=True)
    parser.add_argument("--subs", nargs="+", default=DEFAULT_SUBS, help="Subreddits to scrape")
    parser.add_argument("--limit", type=int, default=100, help="Posts per subreddit (default: 100)")
    parser.add_argument("--sort", choices=SORT_OPTIONS, default="controversial", help="Sort method (default: controversial)")
    parser.add_argument("--out", default="reddit_data.csv", help="Output CSV (default: reddit_data.csv)")
    args = parser.parse_args()

    reddit = praw.Reddit(
        client_id=args.client_id,
        client_secret=args.client_secret,
        user_agent="nobs-hate-speech-research/0.1",
    )

    console.print(f"Scraping [cyan]{args.subs}[/cyan] — {args.limit} posts each, sorted by [cyan]{args.sort}")

    all_rows = []
    for sub in track(args.subs, description="Scraping subreddits..."):
        rows = scrape_sub(reddit, sub, args.limit, args.sort)
        console.print(f"  r/{sub}: {len(rows)} items")
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows).drop_duplicates(subset=["text"])
    df.to_csv(args.out, index=False)
    console.print(f"\n[green]Saved {len(df)} unique texts to {args.out}")
    console.print(f"\nRun classifier:\n  [bold].venv/bin/python classify.py --input {args.out} --backend finbert")


if __name__ == "__main__":
    main()
