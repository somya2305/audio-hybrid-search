"""Hybrid search over transcript chunks: pgvector semantic search plus
Postgres full-text keyword search, merged with Reciprocal Rank Fusion.

Usage:
    python -m src.retrieval.pg_hybrid_search "query"   # one query
    python -m src.retrieval.pg_hybrid_search           # interactive loop
"""

import sys

from pgvector import Vector

from src.common import format_time
from src.database.connection import get_connection
from src.embeddings.embed import embed_query

TOP_K = 20  # candidates from each retriever before fusion
TOP_K_FINAL = 5  # results after fusion

# Turns shorter than this ("Look.", "What?") are left out of semantic search:
# their near-empty embeddings score moderately against any query.
MIN_SEMANTIC_WORDS = 4

# RRF damping constant from the original paper: agreement between both lists
# outweighs a single first place.
RRF_K = 60

# Keyword ranks count half as much as semantic ones in fusion: matching any
# query word lets chunks sharing only common words ("face", "other") into the
# keyword list, and at full weight they pushed real answers out of the top 5.
KEYWORD_WEIGHT = 0.5

# ts_rank length normalization: none. Turns are capped at ~80 words, and
# normalizing by length pushed one-word turns to the top.
TS_RANK_NORMALIZATION = 0

# Turns are short, so the snippet is the whole turn with matched words marked.
HEADLINE_OPTIONS = "StartSel=**, StopSel=**, HighlightAll=true"

# Any query word (stemmed, stopwords dropped). Ranking matches on any word
# rather than all of them keeps natural-language queries from missing.
ANY_WORD_TSQUERY = "replace(plainto_tsquery('english', %(query)s)::text, '&', '|')::tsquery"

# Both retrievers return these columns; wq.word_query marks matches in the
# snippet and finds the first matching word's timestamp.
RESULT_COLUMNS = """
    c.id, c.chunk_id, c.conversation_id, c.chunk_index, c.speaker,
    c.start_time, c.end_time, c.text,
    ts_headline('english', c.text, wq.word_query, %(headline_options)s) AS snippet,
    (
        SELECT (w->>'start')::real
        FROM jsonb_array_elements(c.words) AS w
        WHERE to_tsvector('english', w->>'word') @@ wq.word_query
        ORDER BY (w->>'start')::real
        LIMIT 1
    ) AS match_time
"""

# A NULL %(cid)s matches every conversation.
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
        "snippet": row[8],
        "match_time": row[9],  # start of the first word matching the query, if any
        "score": float(row[10]),
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
        FROM transcript_chunks c,
             (SELECT {ANY_WORD_TSQUERY}) AS wq(word_query)
        WHERE {CONVERSATION_FILTER}
          AND c.word_count >= %(min_words)s
        ORDER BY c.embedding <=> %(embedding)s
        LIMIT %(top_k)s
        """,
        {
            "query": query,
            "headline_options": HEADLINE_OPTIONS,
            "embedding": Vector(embed_query(query)),
            "min_words": MIN_SEMANTIC_WORDS,
            "cid": conversation_id,
            "top_k": top_k,
        },
    )


def keyword_search(query, top_k=TOP_K, conversation_id=None):
    """Rank chunks matching any query word by ts_rank; "quoted text" must match as a phrase."""
    match_query = (
        "websearch_to_tsquery('english', %(query)s)" if '"' in query else ANY_WORD_TSQUERY
    )
    return _run(
        f"""
        SELECT {RESULT_COLUMNS},
               ts_rank(c.text_search, q.query, %(norm)s) AS score
        FROM transcript_chunks c,
             (SELECT {match_query}) AS q(query),
             (SELECT {ANY_WORD_TSQUERY}) AS wq(word_query)
        WHERE c.text_search @@ q.query
          AND {CONVERSATION_FILTER}
        ORDER BY score DESC, c.id
        LIMIT %(top_k)s
        """,
        {
            "query": query,
            "headline_options": HEADLINE_OPTIONS,
            "norm": TS_RANK_NORMALIZATION,
            "cid": conversation_id,
            "top_k": top_k,
        },
    )


def reciprocal_rank_fusion(semantic, keyword, top_k=TOP_K_FINAL, rrf_k=RRF_K,
                           keyword_weight=KEYWORD_WEIGHT):
    """Merge two ranked lists: each chunk scores sum(weight / (rrf_k + rank)).

    Only ranks are used, so cosine similarity and ts_rank never need to be
    made comparable. A chunk in both lists keeps the version with a match_time.
    """
    fused = {}
    for results, weight in ((semantic, 1.0), (keyword, keyword_weight)):
        for result in results:
            entry = fused.get(result["id"])
            if entry is None:
                entry = fused[result["id"]] = {**result, "rrf_score": 0.0}
            elif entry.get("match_time") is None and result.get("match_time") is not None:
                entry = fused[result["id"]] = {**result, "rrf_score": entry["rrf_score"]}
            entry["rrf_score"] += weight / (rrf_k + result["rank"])

    ranked = sorted(fused.values(), key=lambda r: r["rrf_score"], reverse=True)[:top_k]
    for rank, result in enumerate(ranked, start=1):
        result["rank"] = rank
    return ranked


def hybrid_search(query, conversation_id=None, top_k=TOP_K_FINAL):
    """Return (semantic_results, keyword_results, final_results) for the query."""
    semantic = semantic_search(query, conversation_id=conversation_id)
    keyword = keyword_search(query, conversation_id=conversation_id)
    return semantic, keyword, reciprocal_rank_fusion(semantic, keyword, top_k=top_k)


def format_location(result):
    """"<file> · <start>–<end> · <speaker>[ · match at <time>]"."""
    location = (
        f"{result['conversation_id']} · "
        f"{format_time(result['start_time'])}–{format_time(result['end_time'])} · "
        f"{result['speaker']}"
    )
    if result.get("match_time") is not None:
        location += f" · match at {format_time(result['match_time'])}"
    return location


def print_results(query):
    semantic, keyword, final = hybrid_search(query)
    print(f'\n=== "{query}" ===')
    for title, results, score_key in (
        ("Semantic", semantic, "score"), ("Keyword", keyword, "score"), ("Hybrid (RRF)", final, "rrf_score")
    ):
        print(f"\n{title}")
        if not results:
            print("  (no results)")
        for result in results[:TOP_K_FINAL]:
            print(f"  {result['rank']}. [{result[score_key]:.4f}] {format_location(result)}")
            if title.startswith("Hybrid"):
                print(f"     {result['snippet']}")


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
