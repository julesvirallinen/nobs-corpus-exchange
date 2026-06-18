# NoBS · Corpus Exchange


[CLICKABLE DEMO](https//136.114.94.183:8000) (pre-calculated LLM processing, live demo will be with real processing) 


https://github.com/user-attachments/assets/e04ba134-6ca2-488c-accc-bdb789c655dc



## Anonymization algorithm


1. **Triage** — identify tokens that carry author identity
2. **Score** — RoBERTa flags hate-speech and assigns category + severity
3. **Rewrite** — Qwen generates paraphrases; the exponential mechanism selects the one that minimises hate signal under a differential-privacy budget
4. **Review** — human validator confirms labels, then publishes the anonymised corpus to the observatory

## Run

```bash
./run.sh
```

Opens at http://localhost:8000 · requires Python 3.10+, no other installs needed.
