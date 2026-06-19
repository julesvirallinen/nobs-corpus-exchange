# NoBS · Corpus Exchange


clickable demo: http://136.114.94.183:8000 (pre-calculated LLM processing, live demo will be with real processing) 
[backup clickable demo](https://nobs-corpus-exchange.netlify.app/)


https://github.com/user-attachments/assets/e04ba134-6ca2-488c-accc-bdb789c655dc

NOTE: 2026-06-19T8:00:00 It seems that the main branch contains a couple broken docstrings. The server demo was deployed before the deadline, and is our final submission. https://github.com/julesvirallinen/nobs-corpus-exchange/tree/working has the working version, if anyone wants to run the algorithm locally, equal to what is deployed to the final submission. 



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
