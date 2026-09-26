# Audio RAG: project context

Hybrid (keyword + semantic) search over two-speaker podcast clips. Audio → WhisperX transcript with word timings and speakers → speaker-turn chunks → bge-small embeddings → Postgres + pgvector. Queries merge Postgres full-text and pgvector results with weighted RRF. Write-up: SOLUTION.md; evaluation detail: docs/evaluation.md.

## Commands

Always run modules with `python -m` from the repo root (`.venv/bin/python`).

- `docker compose up -d`: Postgres (container `audio-search-pg`); `db/schema.sql` runs only on an empty volume
- `python -m src.database.connection`: check connection, extensions, row counts
- `python -m src.index_chunks`: re-chunk, embed and index saved transcripts (~12 s, no HF_TOKEN)
- `python -m src.pipeline`: transcribe (only files without a transcript) and index; `--force` re-transcribes (~12 min per clip on CPU)
- `python -m src.retrieval.pg_hybrid_search "query"`: search CLI
- `uvicorn src.api.app:app`: web UI at http://localhost:8000 and REST API (docs at /docs)
- `python -m src.evaluation.evaluate --out output/eval/<name>.json`, then `python -m src.evaluation.report <that file> --compare output/eval/baseline.json`
- `pytest`: unit tests, no database needed

## Conventions

- `.env` holds `HF_TOKEN`; never commit it or zip the folder (use `git archive`).
- Paths and `.env` loading live in `src/common.py`; don't rebuild paths per module.
- SQL uses named parameters only; never format user input into SQL.
- Schema changes: edit `db/schema.sql`, then `docker compose down -v && docker compose up -d && python -m src.index_chunks` (this deletes indexed data).
- Chunking or embedding changes: re-run `python -m src.index_chunks`, then evaluate.
- Retrieval changes are judged by `evaluate` against `output/eval/baseline.json`; report every variant tried, and don't add or edit labeled queries to move a metric.
- Labels in `data/queries.json` are time ranges (not chunk ids); validate with `python -m src.evaluation.validate_queries`.
- Don't commit unless asked; the author reviews and commits each step.
