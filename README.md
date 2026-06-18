# NoBS · Corpus Exchange

Upload election posts. The pipeline anonymises them on your device before anything is shared.

## Steps

1. **Triage** — identify tokens that carry author identity
2. **Score** — RoBERTa flags hate-speech and assigns category + severity
3. **Rewrite** — Qwen generates paraphrases; the exponential mechanism selects the one that minimises hate signal under a differential-privacy budget
4. **Review** — human validator confirms labels, then publishes the anonymised corpus to the observatory

## Run

```bash
./run.sh
```

Opens at http://localhost:8000 · requires Python 3.10+, no other installs needed.
