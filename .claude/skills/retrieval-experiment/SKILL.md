---
name: retrieval-experiment
description: Run and record a retrieval experiment (a change to chunking, embeddings, keyword search or fusion) against the frozen baseline. Use when asked to try, compare or evaluate a retrieval change.
---

# Retrieval experiment

1. **State the hypothesis** and the queries it should affect (from `output/eval/baseline.json` or `docs/evaluation.md`).
2. **Make the change**, keeping settings as named constants with a one-line reason.
   - Chunking or embedding change: re-index with `python -m src.index_chunks`.
   - Search or fusion change: no re-index needed.
   - For quick comparisons of several fusion settings, cache each query's semantic and keyword results once and score the variants offline with the metric functions in `src/evaluation/evaluate.py`.
3. **Evaluate**: `python -m src.evaluation.evaluate --out output/eval/<expN_name>.json`
4. **Compare**: `python -m src.evaluation.report output/eval/<expN_name>.json --compare output/eval/baseline.json`
5. **Record** a row in the experiment log in `docs/evaluation.md` for every variant tried, kept or not, with hybrid R@5, R@10, MRR and a short note. State trade-offs (e.g. R@1 drops) and that the choice was made on the same query set.
6. **Guard rails:** never edit `data/queries.json` to move a metric; run `pytest` before finishing; don't commit.
