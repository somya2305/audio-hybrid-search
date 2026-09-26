"""Local sentence embeddings for chunks and queries."""

import os
from functools import lru_cache

from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-small-en-v1.5"  # 384-dim
BATCH_SIZE = 32


@lru_cache(maxsize=1)
def get_embedding_model(model_name=MODEL_NAME):
    print(f"Loading embedding model: {model_name}")
    # The model is public; don't send HF_TOKEN (set for pyannote), since a
    # token without access (or an expired one) makes the Hub reject even
    # public models. token=False alone isn't enough: sentence-transformers 5.x
    # still reads HF_TOKEN from the environment when loading the processor,
    # so it is hidden for the duration of the load.
    hf_token = os.environ.pop("HF_TOKEN", None)
    try:
        return SentenceTransformer(model_name, token=False)
    finally:
        if hf_token is not None:
            os.environ["HF_TOKEN"] = hf_token


def create_embeddings(chunks, model_name=MODEL_NAME):
    """Add an "embedding" vector to each chunk and return the chunks."""
    if not chunks:
        return chunks

    model = get_embedding_model(model_name)

    # Embed the turn with its neighbouring turns for context; the turn's own
    # text is what gets displayed and keyword-indexed.
    texts = [chunk.get("embed_text", chunk["text"]) for chunk in chunks]

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    for chunk, embedding in zip(chunks, embeddings):
        chunk["embedding"] = embedding.tolist()

    return chunks


def embed_query(query, model_name=MODEL_NAME):
    """Return the normalized embedding for a search query."""
    return get_embedding_model(model_name).encode(query, normalize_embeddings=True)
