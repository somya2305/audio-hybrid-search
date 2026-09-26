"""Writes and counts for conversations and transcript chunks.

These functions never commit; the caller wraps them in one transaction so a
conversation and its chunks are saved together or not at all.
"""

import numpy as np
from psycopg.types.json import Jsonb


def upsert_conversation(conn, conversation_id, file_name, file_path=None,
                        duration_seconds=None, speaker_count=2):
    """Insert a conversation, or update its details if it already exists."""
    conn.execute(
        """
        INSERT INTO conversations (conversation_id, file_name, file_path, duration_seconds, speaker_count)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (conversation_id) DO UPDATE SET
            file_name = EXCLUDED.file_name,
            file_path = EXCLUDED.file_path,
            duration_seconds = EXCLUDED.duration_seconds,
            speaker_count = EXCLUDED.speaker_count
        """,
        (conversation_id, file_name, file_path, duration_seconds, speaker_count),
    )


def replace_chunks(conn, conversation_id, chunks):
    """Replace a conversation's chunks; deleting first leaves no stale rows on re-runs."""
    with conn.cursor() as cur:
        cur.execute("DELETE FROM transcript_chunks WHERE conversation_id = %s", (conversation_id,))
        cur.executemany(
            """
            INSERT INTO transcript_chunks (
                chunk_id, conversation_id, chunk_index, speaker, start_time, end_time,
                text, word_count, embed_text, words, embedding
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            [
                (
                    chunk["chunk_id"], conversation_id, chunk["chunk_index"], chunk["speaker"],
                    chunk["start"], chunk["end"], chunk["text"], chunk["word_count"],
                    chunk["embed_text"], Jsonb(chunk["words"]),
                    np.asarray(chunk["embedding"], dtype=np.float32),
                )
                for chunk in chunks
            ],
        )


def count_chunks(conn, conversation_id=None):
    """Number of chunks, for one conversation or in total."""
    if conversation_id is None:
        return conn.execute("SELECT count(*) FROM transcript_chunks").fetchone()[0]
    return conn.execute(
        "SELECT count(*) FROM transcript_chunks WHERE conversation_id = %s", (conversation_id,)
    ).fetchone()[0]
