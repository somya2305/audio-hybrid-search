"""Hybrid search over transcript chunks: pgvector semantic search plus
Postgres full-text keyword search, merged with Reciprocal Rank Fusion.

Usage:
    python -m src.retrieval.pg_hybrid_search "query"   # one query
    python -m src.retrieval.pg_hybrid_search           # interactive loop
"""

import sys

from pgvector import Vector

from src.database.connection import get_connection
from src.embeddings.embed import embed_query

# Candidates returned by each retriever before fusion.
TOP_K = 20

# Results returned after fusion.
TOP_K_FINAL = 5

# RRF damping constant (60 is the value from the original RRF paper): it keeps
# a single first-place rank from outweighing agreement between both lists.
RRF_K = 60

# Shared by both retrievers so their results have the same shape.
RESULT_COLUMNS = """
    c.id, c.chunk_id, c.conversation_id, c.chunk_index, c.speaker,
    c.start_time, c.end_time, c.text
"""

# Optional filter to one conversation; a NULL %(cid)s matches every row.
CONVERSATION_FILTER = "(%(cid)s::text IS NULL OR c.conversation_id = %(cid)s)"


def _row_to_result(row, rank):
    return {
        "id": row[0],
        "chunk_id": row[1],
        "conversation_id": row[2],
        "chunk_index": row[3],
        "speaker": row[4],
        "start_time": row[5],
        "end_time": row[6],
        "text": row[7],
        "score": float(row[8]),
        "rank": rank,
    }


def _run(sql, params):
    with get_connection() as conn:
        rows = conn.execute(sql, params).fetchall()
    return [_row_to_result(row, rank) for rank, row in enumerate(rows, start=1)]


def semantic_search(query, top_k=TOP_K, conversation_id=None):
    """Rank chunks by cosine similarity between the query and chunk embeddings."""
    return _run(
        f"""
        SELECT {RESULT_COLUMNS},
               1 - (c.embedding <=> %(embedding)s) AS score
        FROM transcript_chunks c
        WHERE {CONVERSATION_FILTER}
        ORDER BY c.embedding <=> %(embedding)s
        LIMIT %(top_k)s
        """,
        {
            "embedding": Vector(embed_query(query)),
            "cid": conversation_id,
            "top_k": top_k,
        },
    )


def keyword_search(query, top_k=TOP_K, conversation_id=None):
    """Rank chunks matching the query's words (stemmed, stopwords dropped) by ts_rank.

    plainto_tsquery ANDs the terms, so a chunk must contain every query word.
    """
    return _run(
        f"""
        SELECT {RESULT_COLUMNS},
               ts_rank(c.text_search, q.query) AS score
        FROM transcript_chunks c,
             plainto_tsquery('english', %(query)s) AS q(query)
        WHERE c.text_search @@ q.query
          AND {CONVERSATION_FILTER}
        ORDER BY score DESC, c.id
        LIMIT %(top_k)s
        """,
        {"query": query, "cid": conversation_id, "top_k": top_k},
    )


def reciprocal_rank_fusion(semantic, keyword, top_k=TOP_K_FINAL, rrf_k=RRF_K):
    """Merge two ranked lists: each chunk scores sum(1 / (rrf_k + rank)).

    Only ranks are used, so the retrievers' incomparable scores (cosine
    similarity vs ts_rank) never need normalising.
    """
    fused = {}
    for results in (semantic, keyword):
        for result in results:
            entry = fused.setdefault(result["id"], {**result, "rrf_score": 0.0})
            entry["rrf_score"] += 1.0 / (rrf_k + result["rank"])

    ranked = sorted(fused.values(), key=lambda r: r["rrf_score"], reverse=True)[:top_k]
    for rank, result in enumerate(ranked, start=1):
        result["rank"] = rank
    return ranked


def hybrid_search(query, conversation_id=None, top_k=TOP_K_FINAL):
    """Return (semantic_results, keyword_results, final_results) for the query."""
    semantic = semantic_search(query, conversation_id=conversation_id)
    keyword = keyword_search(query, conversation_id=conversation_id)
    return semantic, keyword, reciprocal_rank_fusion(semantic, keyword, top_k=top_k)


# ============================================================
# Output
# ============================================================

def format_time(seconds):
    """83.46 -> "01:23.5"."""
    # Round first so 59.96 becomes "01:00.0", not "00:60.0".
    minutes, secs = divmod(round(float(seconds), 1), 60)
    return f"{int(minutes):02d}:{secs:04.1f}"


def format_location(result):
    """"<file> · <start>–<end> · <speaker>"."""
    return (
        f"{result['conversation_id']} · "
        f"{format_time(result['start_time'])}–{format_time(result['end_time'])} · "
        f"{result['speaker']}"
    )


def _print_list(title, results, top_k=TOP_K_FINAL, score_key="score"):
    print(f"\n{title}")
    if not results:
        print("  (no results)")
    for result in results[:top_k]:
        print(f"  {result['rank']}. [{result[score_key]:.4f}] {format_location(result)}")


def print_results(query):
    semantic, keyword, final = hybrid_search(query)
    print(f'\n=== "{query}" ===')
    _print_list("Semantic", semantic)
    _print_list("Keyword", keyword)
    print("\nHybrid (RRF)")
    if not final:
        print("  (no results)")
    for result in final:
        print(f"  {result['rank']}. [{result['rrf_score']:.4f}] {format_location(result)}")
        print(f"     {result['text']}")


def main():
    if len(sys.argv) > 1:
        print_results(" ".join(sys.argv[1:]))
        return

    print("Hybrid search. Type a query, or 'exit' to quit.")
    while True:
        try:
            query = input("\nquery> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if query.lower() in {"exit", "quit"}:
            break
        if query:
            print_results(query)


if __name__ == "__main__":
    main()
