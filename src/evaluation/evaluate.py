"""Retrieval evaluation against the labeled queries in data/queries.json.

Runs every query through hybrid_search and scores keyword-only, semantic-only
and hybrid (RRF) rankings separately: recall@k, MRR, recall by query type,
speaker accuracy, and top semantic scores for positive vs negative queries.

Usage:
    python -m src.evaluation.evaluate                                  # -> output/eval/results.json
    python -m src.evaluation.evaluate --out output/eval/<experiment>.json
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from src.common import EVAL_RESULTS_PATH, QUERIES_PATH, ROOT_DIR
from src.evaluation.validate_queries import load_transcripts, validate

K_VALUES = (1, 3, 5, 10)
METHODS = ("keyword", "semantic", "hybrid")

# A result must overlap the labeled range by this much (or half its duration,
# for very short turns). No "within N seconds" tolerance: turns sit back to
# back, so that would accept the adjacent turn, usually the other speaker.
MIN_OVERLAP_S = 0.5

SUCCESS_CRITERIA = {
    "hybrid_recall@5": 0.80,
    "hybrid_recall@10": 0.90,
    "speaker_accuracy": 0.80,
}


# ============================================================
# Matching and metrics (pure functions, no database)
# ============================================================

def matches(result, label, min_overlap=MIN_OVERLAP_S):
    """True if the result is from the label's file and overlaps its time range enough."""
    if result["conversation_id"] != label["file"]:
        return False
    overlap = min(result["end_time"], label["end"]) - max(result["start_time"], label["start"])
    result_duration = result["end_time"] - result["start_time"]
    return overlap > 0 and overlap >= min(min_overlap, result_duration / 2)


def recall_at_k(results, relevant, k):
    """Fraction of labels matched by at least one of the top k results."""
    top = results[:k]
    found = sum(1 for label in relevant if any(matches(r, label) for r in top))
    return found / len(relevant)


def first_hit(results, relevant):
    """(rank, result, label) for the first result matching any label, or None."""
    for rank, result in enumerate(results, start=1):
        for label in relevant:
            if matches(result, label):
                return rank, result, label
    return None


def reciprocal_rank(results, relevant, k):
    """1 / rank of the first hit within the top k, or 0."""
    hit = first_hit(results[:k], relevant)
    return 1.0 / hit[0] if hit else 0.0


def _mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


# ============================================================
# Running
# ============================================================

def _summary(result):
    return {
        "file": result["conversation_id"],
        "start": round(result["start_time"], 2),
        "end": round(result["end_time"], 2),
        "speaker": result["speaker"],
        "text": result["text"][:120],
    }


def evaluate_query(query, max_k=max(K_VALUES)):
    """Run one query and return its per-method metrics and details."""
    # Imported here so the pure metric functions (and tests) need no database.
    from src.retrieval.pg_hybrid_search import hybrid_search

    semantic, keyword, hybrid = hybrid_search(query["query"], top_k=max_k)
    rankings = {"keyword": keyword[:max_k], "semantic": semantic[:max_k], "hybrid": hybrid}
    relevant = query.get("relevant", [])

    detail = {
        "id": query["id"],
        "query": query["query"],
        "type": query["type"],
        "relevant": relevant,
        "top_semantic_score": semantic[0]["score"] if semantic else None,
        "top_hybrid": [_summary(r) for r in hybrid[:3]],
    }
    if not relevant:  # negative query
        return detail

    for method, results in rankings.items():
        detail[method] = {f"recall@{k}": recall_at_k(results, relevant, k) for k in K_VALUES}
        detail[method]["rr"] = reciprocal_rank(results, relevant, max_k)

    hit = first_hit(hybrid, relevant)
    detail["hybrid_hit_rank"] = hit[0] if hit else None
    if hit and hit[2].get("speaker"):
        detail["speaker_correct"] = hit[1]["speaker"] == hit[2]["speaker"]
    return detail


def aggregate(details):
    positives = [d for d in details if d["relevant"]]
    negatives = [d for d in details if not d["relevant"]]

    metrics = {
        "num_queries": len(details),
        "num_positive": len(positives),
        "num_negative": len(negatives),
    }
    for method in METHODS:
        for k in K_VALUES:
            metrics[f"{method}_recall@{k}"] = _mean(d[method][f"recall@{k}"] for d in positives)
        metrics[f"{method}_mrr"] = _mean(d[method]["rr"] for d in positives)

    metrics["hybrid_recall@5_by_type"] = {
        query_type: _mean(d["hybrid"]["recall@5"] for d in positives if d["type"] == query_type)
        for query_type in sorted({d["type"] for d in positives})
    }

    speaker_checks = [d["speaker_correct"] for d in positives if "speaker_correct" in d]
    metrics["speaker_accuracy"] = _mean(1.0 if ok else 0.0 for ok in speaker_checks)
    metrics["speaker_checked"] = len(speaker_checks)

    # A clear gap would let a similarity threshold reject off-topic queries.
    metrics["mean_top_semantic_score_positive"] = _mean(d["top_semantic_score"] for d in positives)
    metrics["mean_top_semantic_score_negative"] = _mean(d["top_semantic_score"] for d in negatives)
    return metrics


def evaluate(queries):
    """Run every query; return {"metrics": ..., "queries": [...]}."""
    details = [evaluate_query(query) for query in queries]
    return {"metrics": aggregate(details), "queries": details}


# ============================================================
# Report
# ============================================================

def _fmt(value):
    return "  -  " if value is None else f"{value:.3f}"


def print_report(evaluation):
    metrics = evaluation["metrics"]

    print("=" * 62)
    print(
        f"RETRIEVAL EVALUATION  ({metrics['num_positive']} labeled, "
        f"{metrics['num_negative']} negative queries)"
    )
    print("=" * 62)
    print(f"{'method':<10}" + "".join(f"{'R@' + str(k):>8}" for k in K_VALUES) + f"{'MRR':>8}")
    for method in METHODS:
        print(
            f"{method:<10}"
            + "".join(f"{_fmt(metrics[f'{method}_recall@{k}']):>8}" for k in K_VALUES)
            + f"{_fmt(metrics[f'{method}_mrr']):>8}"
        )

    print("\nHybrid recall@5 by query type:")
    for query_type, value in metrics["hybrid_recall@5_by_type"].items():
        count = sum(1 for d in evaluation["queries"] if d["type"] == query_type)
        print(f"  {query_type:<10} {_fmt(value)}  ({count} queries)")

    print(
        f"\nSpeaker accuracy (first hybrid hit in top {max(K_VALUES)}): "
        f"{_fmt(metrics['speaker_accuracy'])} over {metrics['speaker_checked']} queries"
    )
    print(
        f"Mean top semantic score: positive {_fmt(metrics['mean_top_semantic_score_positive'])}"
        f" | negative {_fmt(metrics['mean_top_semantic_score_negative'])}"
    )

    misses = [
        d for d in evaluation["queries"]
        if d["relevant"] and (d["hybrid_hit_rank"] is None or d["hybrid_hit_rank"] > 5)
    ]
    print(f"\nMissed by hybrid top 5 ({len(misses)}):")
    for d in misses:
        rank = f"rank {d['hybrid_hit_rank']}" if d["hybrid_hit_rank"] else "not in top 10"
        top = d["top_hybrid"][0] if d["top_hybrid"] else None
        got = f"; #1 was {top['file']} {top['start']:.0f}s {top['speaker']}" if top else ""
        print(f"  {d['id']} [{d['type']}] {d['query']}  ({rank}{got})")

    print("\nSuccess criteria:")
    for name, target in SUCCESS_CRITERIA.items():
        value = metrics.get(name)
        status = "PASS" if value is not None and value >= target else "FAIL"
        print(f"  {name:<20} {_fmt(value)}  (target {target:.2f})  {status}")


def run_config():
    """The settings and code version a run used, so saved results stay comparable."""
    from src.chunking import chunk
    from src.embeddings import embed
    from src.retrieval import pg_hybrid_search as search

    def git(*args):
        return subprocess.run(["git", *args], cwd=ROOT_DIR, capture_output=True, text=True).stdout.strip()

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git("rev-parse", "--short", "HEAD"),
        "uncommitted_changes": bool(git("status", "--porcelain", "src")),
        "embedding_model": embed.MODEL_NAME,
        "query_instruction": embed.QUERY_INSTRUCTION,
        "max_turn_words": chunk.MAX_TURN_WORDS,
        "context_turns": chunk.CONTEXT_TURNS,
        "short_turn_words": chunk.SHORT_TURN_WORDS,
        "top_k_per_retriever": search.TOP_K,
        "rrf_k": search.RRF_K,
        "keyword_weight": search.KEYWORD_WEIGHT,
        "min_semantic_words": search.MIN_SEMANTIC_WORDS,
        "ts_rank_normalization": search.TS_RANK_NORMALIZATION,
    }


def save_results(evaluation, path=EVAL_RESULTS_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(evaluation, f, indent=2, ensure_ascii=False)
    return path


def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval against data/queries.json.")
    parser.add_argument("--out", type=Path, default=EVAL_RESULTS_PATH, help="Where to save the results JSON")
    out = parser.parse_args().out.resolve()

    with open(QUERIES_PATH, encoding="utf-8") as f:
        queries = json.load(f)["queries"]

    errors = validate(queries, load_transcripts())
    if errors:
        print("Query set is invalid:")
        for error in errors:
            print(f"  {error}")
        sys.exit(1)

    evaluation = {"config": run_config(), **evaluate(queries)}
    print_report(evaluation)
    print(f"\nPer-query details saved to {save_results(evaluation, out).relative_to(ROOT_DIR)}")


if __name__ == "__main__":
    main()
