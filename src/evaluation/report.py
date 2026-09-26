"""Markdown tables from a saved evaluation run, optionally compared to another.

Usage:
    python -m src.evaluation.report output/eval/baseline.json
    python -m src.evaluation.report output/eval/new.json --compare output/eval/baseline.json
"""

import argparse
import json

from src.evaluation.evaluate import K_VALUES, METHODS, SUCCESS_CRITERIA

QUERY_TYPES = ("keyword", "phrase", "semantic", "speaker")
LABELS = {"keyword": "Keyword (text)", "semantic": "Semantic", "hybrid": "Hybrid (RRF)"}


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _positives(run):
    return [q for q in run["queries"] if q["relevant"]]


def _mean(values):
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _cell(value, base=None, digits=3):
    if base is None or round(value - base, digits) == 0:
        return f"{value:.{digits}f}"
    return f"{value:.{digits}f} ({value - base:+.{digits}f})"


def overall_table(run, base=None):
    m, b = run["metrics"], (base or {}).get("metrics")
    lines = [
        "| Method | " + " | ".join(f"R@{k}" for k in K_VALUES) + " | MRR |",
        "|---" * (len(K_VALUES) + 2) + "|",
    ]
    for method in METHODS:
        keys = [f"{method}_recall@{k}" for k in K_VALUES] + [f"{method}_mrr"]
        cells = [_cell(m[key], b[key] if b else None) for key in keys]
        lines.append(f"| {LABELS[method]} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _type_recall(run, query_type, method, k):
    queries = [q for q in _positives(run) if query_type == "all" or q["type"] == query_type]
    return _mean(q[method][f"recall@{k}"] for q in queries), len(queries)


def by_type_table(run, base=None):
    header = [f"{method[:3].title()} R@{k}" for method in METHODS for k in K_VALUES]
    lines = ["| Type (n) | " + " | ".join(header) + " |", "|---" * (len(header) + 1) + "|"]
    for query_type in (*QUERY_TYPES, "all"):
        cells, count = [], 0
        for method in METHODS:
            for k in K_VALUES:
                value, count = _type_recall(run, query_type, method, k)
                base_value = _type_recall(base, query_type, method, k)[0] if base else None
                cells.append(_cell(value, base_value, digits=2))
        name = f"**{query_type} ({count})**" if query_type == "all" else f"{query_type} ({count})"
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def per_query_table(run):
    header = [f"{method[:3].title()} R@{k}" for method in METHODS for k in K_VALUES]
    lines = [
        "| id | type | query | " + " | ".join(header) + " | Hybrid rank |",
        "|---" * (len(header) + 4) + "|",
    ]
    for q in _positives(run):
        cells = [f"{q[method][f'recall@{k}']:.2f}" for method in METHODS for k in K_VALUES]
        query = q["query"] if len(q["query"]) <= 45 else q["query"][:44] + "…"
        lines.append(
            f"| {q['id']} | {q['type']} | {query} | " + " | ".join(cells)
            + f" | {q['hybrid_hit_rank'] or '>10'} |"
        )
    return "\n".join(lines)


def criteria_table(run):
    m = run["metrics"]
    rows = [(name, m.get(name), target) for name, target in SUCCESS_CRITERIA.items()]
    best_single = max(m["keyword_recall@5"], m["semantic_recall@5"])
    lines = ["| Criterion | Target | Result | Status |", "|---|---|---|---|"]
    for name, value, target in rows:
        status = "PASS" if value is not None and value >= target else "FAIL"
        lines.append(f"| {name} | ≥ {target:.2f} | {value:.3f} | {status} |")
    status = "PASS" if m["hybrid_recall@5"] >= best_single else "FAIL"
    lines.append(
        f"| hybrid recall@5 ≥ each retriever alone | holds | {m['hybrid_recall@5']:.3f} vs "
        f"{m['keyword_recall@5']:.3f} / {m['semantic_recall@5']:.3f} | {status} |"
    )
    return "\n".join(lines)


def rank_changes(run, base):
    """Queries whose hybrid hit rank changed between two runs."""
    before = {q["id"]: q.get("hybrid_hit_rank") for q in _positives(base)}
    lines = []
    for q in _positives(run):
        old, new = before.get(q["id"]), q.get("hybrid_hit_rank")
        if old != new:
            better = (new or 99) < (old or 99)
            lines.append(
                f"- {q['id']} [{q['type']}] {q['query']}: rank {old or '>10'} → {new or '>10'} "
                f"({'better' if better else 'worse'})"
            )
    return "\n".join(lines) or "- (no hybrid rank changes)"


def main():
    parser = argparse.ArgumentParser(description="Markdown tables from an evaluation run.")
    parser.add_argument("results", help="Results JSON from src.evaluation.evaluate")
    parser.add_argument("--compare", help="Baseline results JSON to show changes against")
    args = parser.parse_args()

    run = load(args.results)
    base = load(args.compare) if args.compare else None

    print("### Overall\n\n" + overall_table(run, base))
    print("\n### Recall by query type\n\n" + by_type_table(run, base))
    print("\n### Success criteria\n\n" + criteria_table(run))
    if base:
        print("\n### Hybrid rank changes vs baseline\n\n" + rank_changes(run, base))
    else:
        print("\n### Per query\n\n" + per_query_table(run))


if __name__ == "__main__":
    main()
