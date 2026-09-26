# Audio RAG: Hybrid Search over Speaker-Diarized Audio Transcripts

Submission for the G2 AI Hiring Hackathon, Problem Statement 1 ("Effective retrieval from audio transcripts"). Given a golden dataset of 5–6 two-speaker audio recordings (8–10 minutes each), this project transcribes and diarizes each file with WhisperX, embeds transcript segments locally with `BAAI/bge-small-en-v1.5`, and indexes them in Postgres + pgvector. Queries run a hybrid search that combines keyword matching (BM25) and semantic similarity, and return the matching file, timestamp, and speaker. Retrieval quality is evaluated with automated pytest tests that measure recall@k against a labeled query set.

## Setup

TODO

## Run

TODO
