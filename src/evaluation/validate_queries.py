"""Check the labeled evaluation queries in data/queries.json.

Usage:
    python -m src.evaluation.validate_queries [queries.json]

Exits non-zero if any query is invalid. Also prints a review table and
warns about semantic queries that reuse words from the spans they label.
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

from src.common import QUERIES_PATH, format_time, transcription_paths

QUERY_TYPES = ("keyword", "phrase", "semantic", "speaker", "negative")

# Spans come from segment timestamps; allow for rounding at the file's end.
DURATION_TOLERANCE = 0.5

# Words too common to count as "reusing the speaker's wording".
STOPWORDS = set(
    "the a an and or but if of to in on at for with about from by as is are was were be "
    "been being it its this that these those what which who whom why how when where do "
    "does did have has had can could will would should may might must just very so than "
    "then there their they them he she his her you your we our i my me not no all any "
    "some more most much many into out up down over under after before again people "
    "thing things like get got make made".split()
)


def load_transcripts():
    """{file: {"duration", "speakers", "segments"}} for every transcript."""
    transcripts = {}
    for path in transcription_paths():
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        segments = data.get("segments", [])
        transcripts[data["file"]] = {
            "duration": max((s["end"] for s in segments), default=0.0),
            "speakers": {s.get("speaker") for s in segments},
            "segments": segments,
        }
    return transcripts


def validate(queries, transcripts):
    """Return a list of error strings (empty if every query is valid)."""
    errors = []
    ids = Counter(q.get("id") for q in queries)
    for qid, count in ids.items():
        if count > 1:
            errors.append(f"{qid}: id used {count} times")

    for q in queries:
        qid = q.get("id", "<no id>")
        if not str(q.get("query", "")).strip():
            errors.append(f"{qid}: empty query")
        if q.get("type") not in QUERY_TYPES:
            errors.append(f"{qid}: unknown type {q.get('type')!r}")

        relevant = q.get("relevant", [])
        if q.get("type") == "negative" and relevant:
            errors.append(f"{qid}: negative query has {len(relevant)} labels")
        if q.get("type") != "negative" and not relevant:
            errors.append(f"{qid}: no relevant spans")

        for i, span in enumerate(relevant, start=1):
            where = f"{qid} span {i}"
            transcript = transcripts.get(span.get("file"))
            if transcript is None:
                errors.append(f"{where}: no transcript for {span.get('file')!r}")
                continue
            start, end = span.get("start"), span.get("end")
            if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
                errors.append(f"{where}: start/end must be numbers")
                continue
            if not start < end:
                errors.append(f"{where}: start {start} is not before end {end}")
            if start < 0 or end > transcript["duration"] + DURATION_TOLERANCE:
                errors.append(
                    f"{where}: {start}-{end}s is outside the file's "
                    f"0-{transcript['duration']:.1f}s"
                )
            if span.get("speaker") and span["speaker"] not in transcript["speakers"]:
                errors.append(f"{where}: speaker {span['speaker']!r} not in {span['file']}")

    return errors


def _content_words(text):
    words = re.findall(r"[a-z0-9]+", text.lower())
    # A 5-letter prefix is a crude stem: "resistant" ~ "resistance".
    return {w[:5] for w in words if len(w) > 3 and w not in STOPWORDS}


def semantic_overlap_warnings(queries, transcripts):
    """Semantic queries sharing content words with their spans (these would
    let keyword search find the answer, so they no longer test embeddings)."""
    warnings = []
    for q in queries:
        if q.get("type") != "semantic":
            continue
        span_text = " ".join(
            s["text"]
            for span in q.get("relevant", [])
            for s in transcripts.get(span["file"], {}).get("segments", [])
            if s["end"] > span["start"] and s["start"] < span["end"]
        )
        shared = sorted(_content_words(q["query"]) & _content_words(span_text))
        if shared:
            warnings.append(f"{q['id']}: shares {', '.join(shared)} with its spans")
    return warnings


def print_table(queries):
    print(f"{'id':<4} | {'type':<8} | {'rev':<3} | {'file':<20} | {'start–end':<15} | query")
    print("-" * 110)
    for q in queries:
        spans = q.get("relevant") or [None]
        for i, span in enumerate(spans):
            first = i == 0
            location = (
                f"{span['file']:<20} | {format_time(span['start'])}–{format_time(span['end'])}"
                if span else f"{'(none)':<20} | {'':<15}"
            )
            print(
                f"{q['id'] if first else '':<4} | {q['type'] if first else '':<8} | "
                f"{('yes' if q.get('reviewed') else 'no') if first else '':<3} | "
                f"{location} | {q['query'] if first else '  (also)'}"
            )


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else QUERIES_PATH
    with open(path, encoding="utf-8") as f:
        queries = json.load(f)["queries"]
    transcripts = load_transcripts()

    print_table(queries)

    by_type = Counter(q["type"] for q in queries)
    by_file = Counter(span["file"] for q in queries for span in {s["file"]: s for s in q.get("relevant", [])}.values())
    reviewed = sum(1 for q in queries if q.get("reviewed"))
    print(f"\n{len(queries)} queries, {reviewed} reviewed")
    print("By type: " + ", ".join(f"{t} {by_type.get(t, 0)}" for t in QUERY_TYPES))
    print("By file (queries with a span in it):")
    for file in sorted(transcripts):
        print(f"  {file:<22} {by_file.get(file, 0)}")

    warnings = semantic_overlap_warnings(queries, transcripts)
    if warnings:
        print("\nWarnings (semantic queries reusing words from their spans):")
        for warning in warnings:
            print(f"  {warning}")

    errors = validate(queries, transcripts)
    if errors:
        print(f"\n{len(errors)} error(s):")
        for error in errors:
            print(f"  {error}")
        sys.exit(1)
    print("\nAll queries valid.")


if __name__ == "__main__":
    main()
