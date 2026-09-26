"""Chunk, embed and index saved transcripts into PostgreSQL.

Runs everything after diarization on the transcripts already in
output/transcription/, so indexing needs no GPU, WhisperX or HF_TOKEN.
Use it after changing chunking or embedding, too.

Usage:
    python -m src.index_chunks                   # every file in output/transcription/
    python -m src.index_chunks output/transcription/health_01_transcription.json
"""

import json
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:  # also allow `python src/index_chunks.py`
    sys.path.insert(0, str(ROOT_DIR))

from src.chunking.chunk import create_chunks, save_chunks
from src.database.connection import get_connection
from src.database.repository import count_chunks, replace_chunks, upsert_conversation
from src.embeddings.embed import create_embeddings

TRANSCRIPTION_DIR = ROOT_DIR / "output" / "transcription"
AUDIO_DIR = ROOT_DIR / "data" / "audio"


def index_transcription_file(conn, transcription_path):
    """Index one transcript; return a result dict, or None if it was skipped."""
    start = time.perf_counter()
    transcription_path = Path(transcription_path)
    with open(transcription_path, encoding="utf-8") as f:
        transcript = json.load(f)

    segments = transcript.get("segments", [])
    if "file" not in transcript or not segments or "speaker" not in segments[0]:
        print(f"{transcription_path.name}: skipped (not a diarized transcript)")
        return None

    conversation_id = transcript["file"]
    chunks = create_chunks(transcript)
    if not chunks:
        print(f"{transcription_path.name}: skipped (no chunks)")
        return None
    save_chunks(transcript, chunks, transcription_path)

    create_embeddings(chunks)
    speakers = sorted({chunk["speaker"] for chunk in chunks})

    # Conversation + chunks are committed together or not at all.
    with conn.transaction():
        upsert_conversation(
            conn,
            conversation_id,
            file_name=conversation_id,
            file_path=str((AUDIO_DIR / conversation_id).relative_to(ROOT_DIR)),
            duration_seconds=chunks[-1]["end"],
            speaker_count=len(speakers),
        )
        replace_chunks(conn, conversation_id, chunks)

    result = {
        "conversation_id": conversation_id,
        "num_chunks": len(chunks),
        "speakers": speakers,
        "seconds": time.perf_counter() - start,
    }
    print(
        f"{conversation_id:<22} {result['num_chunks']:>4} chunks  "
        f"{len(speakers)} speakers ({', '.join(speakers)})  {result['seconds']:.1f}s"
    )
    return result


def main(paths):
    if not paths:
        paths = sorted(TRANSCRIPTION_DIR.glob("*_transcription.json"))
    if not paths:
        sys.exit(f"No transcription files found in {TRANSCRIPTION_DIR}")

    total_start = time.perf_counter()
    results, failed = [], []

    # autocommit=True so each conn.transaction() below is a real commit.
    with get_connection(autocommit=True) as conn:
        print(f"Indexing {len(paths)} transcript(s)\n")
        for path in paths:
            try:
                result = index_transcription_file(conn, path)
                if result:
                    results.append(result)
            except Exception as e:  # keep going; one bad file shouldn't stop the rest
                failed.append(str(path))
                print(f"{Path(path).name}: ERROR {type(e).__name__}: {e}")

        conversations = conn.execute("SELECT count(*) FROM conversations").fetchone()[0]
        db_chunks = count_chunks(conn)

    print(
        f"\nIndexed {len(results)}/{len(paths)} files, "
        f"{sum(r['num_chunks'] for r in results)} chunks in "
        f"{time.perf_counter() - total_start:.1f}s"
    )
    print(f"Database: {conversations} conversations, {db_chunks} chunks")
    if failed:
        sys.exit(f"Failed: {', '.join(failed)}")


if __name__ == "__main__":
    main(sys.argv[1:])
