-- =========================================================
-- 1. Enable extensions
-- =========================================================
-- vector  : embeddings + HNSW index
-- pg_trgm : trigram similarity for fuzzy keyword matching

CREATE EXTENSION IF NOT EXISTS vector;

CREATE EXTENSION IF NOT EXISTS pg_trgm;


-- =========================================================
-- 2. Conversations / audio files
-- =========================================================

CREATE TABLE IF NOT EXISTS conversations (
    id BIGSERIAL PRIMARY KEY,

    conversation_id TEXT NOT NULL UNIQUE,

    file_name TEXT NOT NULL,

    file_path TEXT,

    duration_seconds REAL,

    speaker_count INTEGER DEFAULT 2,

    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);


-- =========================================================
-- 3. Transcript chunks (one row per speaker turn)
-- =========================================================
-- chunk_id   : "<file>_0001", the ID used by the chunk JSON and eval labels
-- text       : the turn's own words (displayed + keyword-indexed)
-- word_count : words in text; very short turns are left out of semantic search
-- embed_text : the turn plus its neighbouring turns (what is embedded)
-- words      : [{"word", "start", "end"}, ...] for exact match timestamps
-- embedding  : VECTOR(384) matches BAAI/bge-small-en-v1.5
-- =========================================================

CREATE TABLE IF NOT EXISTS transcript_chunks (
    id BIGSERIAL PRIMARY KEY,

    chunk_id TEXT NOT NULL UNIQUE,

    conversation_id TEXT NOT NULL
        REFERENCES conversations(conversation_id)
        ON DELETE CASCADE,

    chunk_index INTEGER NOT NULL,

    speaker TEXT NOT NULL,

    start_time REAL NOT NULL,

    end_time REAL NOT NULL,

    text TEXT NOT NULL,

    word_count INTEGER NOT NULL DEFAULT 0,

    embed_text TEXT,

    words JSONB NOT NULL DEFAULT '[]'::jsonb,

    embedding VECTOR(384),

    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,

    UNIQUE (conversation_id, chunk_index)
);


-- =========================================================
-- 4. PostgreSQL Full Text Search
-- =========================================================

ALTER TABLE transcript_chunks
ADD COLUMN IF NOT EXISTS text_search TSVECTOR
GENERATED ALWAYS AS (
    to_tsvector('english', text)
) STORED;


-- =========================================================
-- 5. Keyword / Full Text Search Index
-- =========================================================

CREATE INDEX IF NOT EXISTS idx_transcript_chunks_text_search
ON transcript_chunks
USING GIN (text_search);


-- =========================================================
-- 6. Vector Similarity Search Index
-- =========================================================

CREATE INDEX IF NOT EXISTS idx_transcript_chunks_embedding_hnsw
ON transcript_chunks
USING hnsw (embedding vector_cosine_ops);


-- =========================================================
-- 7. Timestamp Index
-- =========================================================
-- Also serves conversation_id lookups, so no separate index is needed.

CREATE INDEX IF NOT EXISTS idx_transcript_chunks_timestamp
ON transcript_chunks (conversation_id, start_time);


-- =========================================================
-- 8. Fuzzy Keyword Index (trigrams)
-- =========================================================
-- Catches names Whisper misspells ("Karpathy" -> "Carpathy"), which
-- full text search misses entirely. Used with word_similarity / <%.

CREATE INDEX IF NOT EXISTS idx_transcript_chunks_text_trgm
ON transcript_chunks
USING GIN (text gin_trgm_ops);

