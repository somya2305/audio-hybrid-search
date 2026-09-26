"""Writes and counts for conversations and transcript chunks.

Every function takes an open connection and never commits, so the caller
controls the transaction: a conversation and its chunks are committed
together or not at all. Either wrap the calls in `with conn:` (commits on
success, rolls back on error), or use get_connection(autocommit=True) with
`with conn.transaction():`. On a default connection, conn.transaction()
after any earlier query only creates a savepoint and does not commit.
"""

import numpy as np
from psycopg.types.json import Jsonb


def upsert_conversation(
    conn,
    conversation_id,
    file_name,
    file_path=None,
    duration_seconds=None,
    speaker_count=2,
):
    """Insert a conversation, or update its details if it already exists."""
    conn.execute(
        """
        INSERT INTO conversations (
            conversation_id, file_name, file_path, duration_seconds, speaker_count
        )
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
    """Replace all chunks for a conversation with the given ones.

    Existing rows are deleted first so a re-run that produces fewer chunks
    doesn't leave stale rows behind.
    """
    with conn.cursor() as cur:
        cur.execute(
            "DELETE FROM transcript_chunks WHERE conversation_id = %s",
            (conversation_id,),
        )

        if not chunks:
            return

        cur.executemany(
            """
            INSERT INTO transcript_chunks (
                chunk_id, conversation_id, chunk_index, speaker,
                start_time, end_time, text, embed_text, words, embedding
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            [
                (
                    chunk.get("chunk_id")
                    or f"{conversation_id}_{chunk['chunk_index']:04d}",
                    conversation_id,
                    chunk["chunk_index"],
                    chunk.get("speaker", "UNKNOWN"),
                    chunk["start"],
                    chunk["end"],
                    chunk["text"],
                    chunk.get("embed_text"),
                    Jsonb(chunk.get("words", [])),
                    None
                    if chunk.get("embedding") is None
                    else np.asarray(chunk["embedding"], dtype=np.float32),
                )
                for chunk in chunks
            ],
        )


def count_chunks(conn, conversation_id=None):
    """Return the number of chunks, for one conversation or in total."""
    if conversation_id is None:
        return conn.execute("SELECT count(*) FROM transcript_chunks").fetchone()[0]
    return conn.execute(
        "SELECT count(*) FROM transcript_chunks WHERE conversation_id = %s",
        (conversation_id,),
    ).fetchone()[0]
