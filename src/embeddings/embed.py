"""Local sentence embeddings (BAAI/bge-small-en-v1.5, 384-dim) for chunks and queries."""

import os
from functools import lru_cache

from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-small-en-v1.5"
BATCH_SIZE = 32

# BGE's recommended instruction for short queries searching longer passages.
# Only queries get it; chunks are embedded as-is.
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


@lru_cache(maxsize=1)
def get_embedding_model(model_name=MODEL_NAME):
    # The model is public, but sentence-transformers still sends HF_TOKEN (meant
    # for pyannote) despite token=False, and an expired token gets it rejected.
    hf_token = os.environ.pop("HF_TOKEN", None)
    try:
        return SentenceTransformer(model_name, token=False)
    finally:
        if hf_token is not None:
            os.environ["HF_TOKEN"] = hf_token


def create_embeddings(chunks, model_name=MODEL_NAME):
    """Add a normalized "embedding" to each chunk, computed from its embed_text."""
    if not chunks:
        return chunks
    embeddings = get_embedding_model(model_name).encode(
        [chunk.get("embed_text", chunk["text"]) for chunk in chunks],
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    for chunk, embedding in zip(chunks, embeddings):
        chunk["embedding"] = embedding.tolist()
    return chunks


def embed_query(query, model_name=MODEL_NAME):
    """Normalized embedding for a search query (with BGE's query instruction)."""
    return get_embedding_model(model_name).encode(QUERY_INSTRUCTION + query, normalize_embeddings=True)
