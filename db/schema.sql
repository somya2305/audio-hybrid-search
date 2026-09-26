-- Schema for hybrid (keyword + semantic) search over diarized transcripts.
-- Mounted as a docker init script: runs only when the data volume is empty.

CREATE EXTENSION IF NOT EXISTS vector;   -- embeddings + HNSW index
CREATE EXTENSION IF NOT EXISTS pg_trgm;  -- trigram index for fuzzy matching

CREATE TABLE IF NOT EXISTS conversations (
    id               BIGSERIAL PRIMARY KEY,
    conversation_id  TEXT NOT NULL UNIQUE,  -- audio file name, e.g. "health_01.wav"
    file_name        TEXT NOT NULL,
    file_path        TEXT,
    duration_seconds REAL,
    speaker_count    INTEGER DEFAULT 2,
    created_at       TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- One row per speaker turn.
CREATE TABLE IF NOT EXISTS transcript_chunks (
    id              BIGSERIAL PRIMARY KEY,
    chunk_id        TEXT NOT NULL UNIQUE,  -- "<file>_0001"
    conversation_id TEXT NOT NULL REFERENCES conversations(conversation_id) ON DELETE CASCADE,
    chunk_index     INTEGER NOT NULL,
    speaker         TEXT NOT NULL,
    start_time      REAL NOT NULL,
    end_time        REAL NOT NULL,
    text            TEXT NOT NULL,                     -- the turn itself: displayed + keyword-indexed
    word_count      INTEGER NOT NULL DEFAULT 0,        -- very short turns skip semantic search
    embed_text      TEXT,                              -- turn + neighbouring turns: what is embedded
    words           JSONB NOT NULL DEFAULT '[]'::jsonb, -- [{"word", "start", "end"}] for match timestamps
    embedding       VECTOR(384),                       -- BAAI/bge-small-en-v1.5
    created_at      TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    text_search     TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', text)) STORED,
    UNIQUE (conversation_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS idx_transcript_chunks_text_search
    ON transcript_chunks USING GIN (text_search);

CREATE INDEX IF NOT EXISTS idx_transcript_chunks_embedding_hnsw
    ON transcript_chunks USING hnsw (embedding vector_cosine_ops);

-- Also serves conversation_id lookups.
CREATE INDEX IF NOT EXISTS idx_transcript_chunks_timestamp
    ON transcript_chunks (conversation_id, start_time);

-- Fuzzy name matching (Whisper writes "Karpathy" as "Carpathy"); not used by search yet.
CREATE INDEX IF NOT EXISTS idx_transcript_chunks_text_trgm
    ON transcript_chunks USING GIN (text gin_trgm_ops);
