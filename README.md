# Audio RAG: Hybrid Search over Speaker-Diarized Audio Transcripts

Submission for the G2 AI Hiring Hackathon, Problem Statement 1 ("Effective retrieval from audio transcripts"). Seven two-speaker podcast clips (8–10 minutes each) are transcribed and diarized with WhisperX, split into speaker-turn chunks, embedded locally with `BAAI/bge-small-en-v1.5`, and stored in Postgres + pgvector. A query runs semantic search (pgvector cosine distance) and keyword search (Postgres full-text search) and merges them with Reciprocal Rank Fusion. Each result shows the file, timestamp, speaker and the turn with matched words highlighted. Retrieval quality is measured as recall@k and MRR against 25 labeled queries.

The full write-up (design, rationale, results, limitations, use of a coding agent) is in [SOLUTION.md](SOLUTION.md).

## Layout

```
data/audio/            golden dataset (7 WAV clips); data/dataset.json lists their sources
data/queries.json      25 labeled evaluation queries
docs/evaluation.md     evaluation results, experiment log and known failure modes
db/schema.sql          tables and indexes (applied automatically by docker compose)
output/transcription/  diarized transcripts (included, so indexing needs no GPU)
output/chunks/         speaker-turn chunks
output/eval/           evaluation runs (baseline.json is the frozen baseline)
src/diarization/       WhisperX transcription + diarization
src/chunking/          speaker-turn chunking
src/embeddings/        local embeddings
src/database/          connection and writes
src/retrieval/         hybrid search + CLI
src/evaluation/        query validation and recall@k evaluation
src/index_chunks.py    transcripts -> chunks -> embeddings -> Postgres
src/pipeline.py        audio -> transcripts -> ... -> Postgres
tests/                 unit tests for the evaluation maths
```

## Setup

Requirements: Python 3.12 or 3.13 (tested on 3.13), Docker, and ffmpeg (`brew install ffmpeg`; needed only to transcribe audio).

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt       # a few minutes: torch + WhisperX
cp .env.example .env                  # defaults work; HF_TOKEN only needed to transcribe
docker compose up -d                  # Postgres + pgvector; schema from db/schema.sql
python -m src.database.connection     # should list both tables, 0 rows
```

If port 5432 is already in use, set another `DB_PORT` in `.env` before `docker compose up -d`.

**Hugging Face token (only to transcribe/diarize audio).** Indexing, search and evaluation work without it, because the transcripts are included. To transcribe:
1. Accept the terms of [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1) while logged in to Hugging Face.
2. Create a read token at <https://huggingface.co/settings/tokens> and set `HF_TOKEN=` in `.env`.

## Run

```bash
# Index the included transcripts (no GPU or HF_TOKEN needed, ~15 s)
python -m src.index_chunks

# Or run everything from the audio (~12 min per file on CPU; reuses existing transcripts)
python -m src.pipeline                # add --force to re-transcribe

# Search
python -m src.retrieval.pg_hybrid_search "insulin resistance in lean people"
python -m src.retrieval.pg_hybrid_search '"passive data collection"'   # exact phrase
python -m src.retrieval.pg_hybrid_search                               # interactive

# Evaluate
python -m src.evaluation.validate_queries   # check the labeled queries
python -m src.evaluation.evaluate           # recall@1/3/5/10 and MRR per method
python -m src.evaluation.report output/eval/results.json --compare output/eval/baseline.json
pytest                                      # unit tests (no database needed)
```

Individual stages can also be run on their own: `python -m src.diarization.diarize [audio]` and `python -m src.chunking.chunk [transcript ...]`.
