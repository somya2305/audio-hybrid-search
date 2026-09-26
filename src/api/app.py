"""HTTP API and web UI for searching the index and evaluating retrieval.

Usage:
    uvicorn src.api.app:app            # UI at http://localhost:8000, API docs at /docs
"""

import json
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src.common import AUDIO_DIR, EVAL_RESULTS_PATH, QUERIES_PATH
from src.database.connection import get_connection
from src.embeddings.embed import embed_query
from src.evaluation.evaluate import (
    K_VALUES, METHODS, SUCCESS_CRITERIA, evaluate, first_hit, matches, recall_at_k, reciprocal_rank,
)
from src.evaluation.validate_queries import QUERY_TYPES, load_transcripts, validate
from src.retrieval.pg_hybrid_search import format_location, hybrid_search

STATIC_DIR = Path(__file__).parent / "static"
EVAL_DIR = EVAL_RESULTS_PATH.parent
MAX_K = max(K_VALUES)


@asynccontextmanager
async def lifespan(app):
    embed_query("warm up")  # load the embedding model before the first request
    yield


app = FastAPI(
    title="Audio RAG",
    description="Hybrid (keyword + semantic) search over speaker-diarized podcast transcripts.",
    lifespan=lifespan,
)


class Span(BaseModel):
    file: str = Field(examples=["health_02.wav"])
    start: float = Field(examples=[409.7])
    end: float = Field(examples=[467.6])
    speaker: str | None = Field(default=None, examples=["SPEAKER_01"])


class CustomQuery(BaseModel):
    query: str = Field(examples=["why injected insulin behaves differently from the body's own"])
    type: str = Field(default="semantic", examples=["semantic"])
    relevant: list[Span] = Field(default_factory=list)


def _load_queries():
    with open(QUERIES_PATH, encoding="utf-8") as f:
        return json.load(f)["queries"]


def _public(result, relevant=()):
    """A search result as JSON, with whether it matches any labeled span."""
    return {
        "rank": result["rank"],
        "chunk_id": result["chunk_id"],
        "file": result["conversation_id"],
        "speaker": result["speaker"],
        "start": round(result["start_time"], 2),
        "end": round(result["end_time"], 2),
        "match_time": None if result["match_time"] is None else round(result["match_time"], 2),
        "score": round(result.get("rrf_score", result["score"]), 5),
        "location": format_location(result),
        "text": result["text"],
        "snippet": result["snippet"],
        "hit": any(matches(result, label) for label in relevant),
    }


def _evaluate_one(query, relevant):
    """Run one query through all three retrievers and score each against the labels."""
    start = time.perf_counter()
    semantic, keyword, hybrid = hybrid_search(query, top_k=MAX_K)
    rankings = {"keyword": keyword[:MAX_K], "semantic": semantic[:MAX_K], "hybrid": hybrid}
    methods = {}
    for method, results in rankings.items():
        hit = first_hit(results, relevant) if relevant else None
        methods[method] = {
            "recall": {f"@{k}": recall_at_k(results, relevant, k) for k in K_VALUES} if relevant else None,
            "reciprocal_rank": reciprocal_rank(results, relevant, MAX_K) if relevant else None,
            "first_hit_rank": hit[0] if hit else None,
            "results": [_public(r, relevant) for r in results],
        }
    return {"query": query, "relevant": relevant, "methods": methods,
            "elapsed_ms": round((time.perf_counter() - start) * 1000)}


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health", tags=["status"])
def health():
    """Database connection and what is indexed."""
    try:
        with get_connection() as conn:
            rows = conn.execute(
                """
                SELECT c.conversation_id, c.duration_seconds, c.speaker_count, count(t.id)
                FROM conversations c LEFT JOIN transcript_chunks t USING (conversation_id)
                GROUP BY 1, 2, 3 ORDER BY 1
                """
            ).fetchall()
    except Exception as e:
        raise HTTPException(503, f"Database unavailable: {e}")
    return {
        "status": "ok",
        "files": [
            {"file": f, "duration_s": round(d or 0, 1), "speakers": s, "chunks": n}
            for f, d, s, n in rows
        ],
        "total_chunks": sum(r[3] for r in rows),
    }


@app.get("/search", tags=["search"])
def search(
    q: str = Query(..., min_length=1, description='Search text; "quoted text" must match as a phrase'),
    k: int = Query(5, ge=1, le=20, description="Results per method"),
    file: str | None = Query(None, description="Only search this file, e.g. health_02.wav"),
):
    """Hybrid search, plus the keyword-only and semantic-only lists it was fused from."""
    start = time.perf_counter()
    semantic, keyword, hybrid = hybrid_search(q, conversation_id=file, top_k=k)
    return {
        "query": q,
        "hybrid": [_public(r) for r in hybrid],
        "keyword": [_public(r) for r in keyword[:k]],
        "semantic": [_public(r) for r in semantic[:k]],
        "elapsed_ms": round((time.perf_counter() - start) * 1000),
    }


@app.get("/queries", tags=["evaluation"])
def list_queries():
    """The labeled evaluation queries (data/queries.json)."""
    return _load_queries()


@app.get("/evaluate/query/{query_id}", tags=["evaluation"])
def evaluate_labeled_query(query_id: str):
    """Evaluate one labeled query: each method's top 10, which results hit a label, recall@k."""
    query = next((q for q in _load_queries() if q["id"] == query_id), None)
    if query is None:
        raise HTTPException(404, f"No query with id {query_id!r}")
    return {"id": query["id"], "type": query["type"], "note": query.get("note"),
            **_evaluate_one(query["query"], query["relevant"])}


@app.post("/evaluate/query", tags=["evaluation"])
def evaluate_custom_query(body: CustomQuery):
    """Check your own query: with answer span(s), results are scored against them;
    without, each method's results are returned unscored."""
    relevant = [span.model_dump() for span in body.relevant]
    if relevant:
        errors = validate([{"id": "custom", "query": body.query, "type": body.type, "relevant": relevant}],
                          load_transcripts())
        if errors:
            raise HTTPException(422, errors)
    query_type = body.type if relevant or body.type == "negative" else "unlabeled"
    return {"id": "custom", "type": query_type, **_evaluate_one(body.query, relevant)}


@app.post("/evaluate", tags=["evaluation"])
def evaluate_all():
    """Run every labeled query and return recall@k, MRR, by-type recall and the success criteria."""
    start = time.perf_counter()
    queries = _load_queries()
    errors = validate(queries, load_transcripts())
    if errors:
        raise HTTPException(422, errors)
    result = evaluate(queries)
    return {**result, "criteria": _criteria(result["metrics"]),
            "elapsed_ms": round((time.perf_counter() - start) * 1000)}


def _criteria(metrics):
    rows = [
        {"name": name, "target": target, "value": metrics.get(name),
         "passed": metrics.get(name) is not None and metrics[name] >= target}
        for name, target in SUCCESS_CRITERIA.items()
    ]
    best_single = max(metrics["keyword_recall@5"], metrics["semantic_recall@5"])
    rows.append({"name": "hybrid_recall@5 >= each retriever alone", "target": best_single,
                 "value": metrics["hybrid_recall@5"], "passed": metrics["hybrid_recall@5"] >= best_single})
    return rows


@app.get("/results", tags=["evaluation"])
def list_results():
    """Saved evaluation runs in output/eval/."""
    return sorted(p.stem for p in EVAL_DIR.glob("*.json"))


@app.get("/results/{name}", tags=["evaluation"])
def get_result(name: str):
    """A saved evaluation run (e.g. baseline) with its config, metrics and criteria."""
    if name not in list_results():
        raise HTTPException(404, f"No saved run {name!r}")
    with open(EVAL_DIR / f"{name}.json", encoding="utf-8") as f:
        run = json.load(f)
    return {"name": name, "config": run.get("config"), "metrics": run["metrics"],
            "criteria": _criteria(run["metrics"]), "queries": run["queries"]}


@app.get("/audio/{file}", tags=["audio"])
def audio(file: str):
    """Stream an audio clip (supports seeking, for playing from a timestamp)."""
    path = AUDIO_DIR / file
    if Path(file).name != file or path.suffix != ".wav" or not path.is_file():
        raise HTTPException(404, f"No audio file {file!r}")
    return FileResponse(path, media_type="audio/wav")


@app.get("/meta", tags=["status"])
def meta():
    """Constants the UI needs: k values, methods, query types."""
    return {"k_values": K_VALUES, "methods": METHODS, "query_types": QUERY_TYPES,
            "success_criteria": SUCCESS_CRITERIA}
