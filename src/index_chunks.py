"""Chunk, embed and index saved transcripts into PostgreSQL.

Needs no GPU, WhisperX or HF_TOKEN: it starts from the transcripts in
output/transcription/. Re-run it after changing chunking or embedding.

Usage:
    python -m src.index_chunks                   # every file in output/transcription/
    python -m src.index_chunks output/transcription/health_01_transcription.json
"""

import json
import sys
import time
from pathlib import Path

from src.chunking.chunk import create_chunks, save_chunks
from src.common import transcription_paths
from src.database.connection import get_connection
from src.database.repository import count_chunks, replace_chunks, upsert_conversation
from src.embeddings.embed import create_embeddings


def index_transcription_file(conn, transcription_path):
    """Chunk, embed and store one transcript; return its chunk count (0 if skipped)."""
    start = time.perf_counter()
    with open(transcription_path, encoding="utf-8") as f:
        transcript = json.load(f)

    chunks = create_chunks(transcript)
    if not chunks:
        print(f"{Path(transcription_path).name}: skipped (no segments)")
        return 0
    save_chunks(transcript, chunks, transcription_path)
    create_embeddings(chunks)

    conversation_id = transcript["file"]
    speakers = sorted({chunk["speaker"] for chunk in chunks})
    with conn.transaction():
        upsert_conversation(
            conn,
            conversation_id,
            file_name=conversation_id,
            file_path=f"data/audio/{conversation_id}",
            duration_seconds=chunks[-1]["end"],
            speaker_count=len(speakers),
        )
        replace_chunks(conn, conversation_id, chunks)

    print(
        f"{conversation_id:<22} {len(chunks):>4} chunks  {len(speakers)} speakers  "
        f"{time.perf_counter() - start:.1f}s"
    )
    return len(chunks)


def index_files(paths):
    """Index each transcript, continuing past failures; return the failed paths."""
    failed, total_chunks = [], 0
    start = time.perf_counter()

    # autocommit=True so each file's conn.transaction() is a real commit.
    with get_connection(autocommit=True) as conn:
        for path in paths:
            try:
                total_chunks += index_transcription_file(conn, path)
            except Exception as e:
                failed.append(str(path))
                print(f"{Path(path).name}: ERROR {type(e).__name__}: {e}")
        conversations = conn.execute("SELECT count(*) FROM conversations").fetchone()[0]
        db_chunks = count_chunks(conn)

    print(
        f"\nIndexed {len(paths) - len(failed)}/{len(paths)} files, {total_chunks} chunks "
        f"in {time.perf_counter() - start:.1f}s. Database: {conversations} conversations, "
        f"{db_chunks} chunks."
    )
    return failed


def main():
    paths = transcription_paths(sys.argv[1:])
    if not paths:
        sys.exit("No transcripts found in output/transcription/")
    failed = index_files(paths)
    if failed:
        sys.exit(f"Failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
