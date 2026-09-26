"""Recall@k against the live index and the labeled query set (data/queries.json).

Needs Postgres running with the transcripts indexed (python -m src.index_chunks);
skipped if the database is unreachable. Run with `pytest -s` to also print the report.
"""

import json

import pytest

from src.common import QUERIES_PATH
from src.evaluation.evaluate import SUCCESS_CRITERIA, evaluate, print_report
from src.evaluation.validate_queries import load_transcripts, validate


@pytest.fixture(scope="module")
def queries():
    with open(QUERIES_PATH, encoding="utf-8") as f:
        return json.load(f)["queries"]


@pytest.fixture(scope="module")
def indexed_files():
    from src.database.connection import get_connection

    try:
        with get_connection() as conn:
            return dict(conn.execute(
                "SELECT conversation_id, duration_seconds FROM conversations"
            ).fetchall())
    except Exception as e:
        pytest.skip(f"database not reachable: {e}")


@pytest.fixture(scope="module")
def evaluation(queries, indexed_files):
    result = evaluate(queries)
    print_report(result)  # shown with `pytest -s`
    return result


def test_query_set_is_valid(queries):
    assert validate(queries, load_transcripts()) == []


def test_labeled_files_are_indexed(queries, indexed_files):
    for query in queries:
        for label in query["relevant"]:
            assert label["file"] in indexed_files, f"{query['id']}: {label['file']} is not indexed"


@pytest.mark.parametrize("name", sorted(SUCCESS_CRITERIA))
def test_success_criterion(evaluation, name):
    value, target = evaluation["metrics"][name], SUCCESS_CRITERIA[name]
    assert value is not None and value >= target, f"{name} = {value:.3f}, target {target}"


def test_hybrid_beats_each_retriever_alone(evaluation):
    m = evaluation["metrics"]
    assert m["hybrid_recall@5"] >= m["keyword_recall@5"], "hybrid is worse than keyword-only"
    assert m["hybrid_recall@5"] >= m["semantic_recall@5"], "hybrid is worse than semantic-only"


def test_every_result_has_file_time_and_speaker(queries, indexed_files):
    from src.retrieval.pg_hybrid_search import hybrid_search

    for query in queries:
        for r in hybrid_search(query["query"], top_k=10)[2]:
            assert r["conversation_id"] and r["speaker"], (query["id"], r)
            assert r["start_time"] is not None and r["start_time"] < r["end_time"], (query["id"], r)
