"""Unit tests for the evaluation maths and query validation (no database)."""

import pytest

from src.evaluation.evaluate import first_hit, matches, recall_at_k, reciprocal_rank
from src.evaluation.validate_queries import validate

FILE = "a.wav"
LABEL = {"file": FILE, "start": 10.0, "end": 20.0, "speaker": "SPEAKER_01"}


def result(start, end, file=FILE, speaker="SPEAKER_01"):
    return {"conversation_id": file, "start_time": start, "end_time": end, "speaker": speaker}


# ------------------------------------------------------------
# matches
# ------------------------------------------------------------

def test_overlapping_result_matches():
    assert matches(result(15.0, 25.0), LABEL)


@pytest.mark.parametrize("start, end", [(5.0, 10.0), (20.0, 25.0)])
def test_adjacent_turn_does_not_match(start, end):
    # Turns sit back to back: the turn ending exactly at the label's start
    # (or starting at its end) is a different turn, usually the other speaker.
    assert not matches(result(start, end), LABEL)


def test_barely_touching_result_does_not_match():
    # 0.2s overlap on a 10s turn: under the 0.5s minimum.
    assert not matches(result(19.8, 29.8), LABEL)
    assert not matches(result(0.2, 10.2), LABEL)


def test_short_turn_inside_label_matches():
    # A 0.3s turn can't overlap by 0.5s; half its duration is enough.
    assert matches(result(12.0, 12.3), LABEL)


def test_wrong_file_does_not_match():
    assert not matches(result(10.0, 20.0, file="b.wav"), LABEL)


# ------------------------------------------------------------
# recall, first hit, reciprocal rank
# ------------------------------------------------------------

LABEL_2 = {"file": FILE, "start": 100.0, "end": 110.0, "speaker": "SPEAKER_00"}


def test_recall_counts_each_label_once():
    results = [result(10.0, 15.0), result(15.0, 20.0)]  # both hit LABEL
    assert recall_at_k(results, [LABEL], k=2) == 1.0
    assert recall_at_k(results, [LABEL, LABEL_2], k=2) == 0.5


def test_recall_respects_k():
    results = [result(50.0, 60.0), result(70.0, 80.0), result(12.0, 18.0)]
    assert recall_at_k(results, [LABEL], k=2) == 0.0
    assert recall_at_k(results, [LABEL], k=3) == 1.0


def test_first_hit_returns_rank_result_and_label():
    results = [result(50.0, 60.0), result(101.0, 105.0)]
    rank, hit, label = first_hit(results, [LABEL, LABEL_2])
    assert (rank, hit, label) == (2, results[1], LABEL_2)
    assert first_hit([result(50.0, 60.0)], [LABEL]) is None


def test_reciprocal_rank():
    results = [result(50.0, 60.0), result(70.0, 80.0), result(12.0, 18.0)]
    assert reciprocal_rank(results, [LABEL], k=10) == pytest.approx(1 / 3)
    assert reciprocal_rank(results, [LABEL], k=2) == 0.0


# ------------------------------------------------------------
# query validation
# ------------------------------------------------------------

TRANSCRIPTS = {FILE: {"duration": 600.0, "speakers": {"SPEAKER_00", "SPEAKER_01"}, "segments": []}}


def query(qid="q01", qtype="keyword", relevant=None):
    return {"id": qid, "query": "some query", "type": qtype,
            "relevant": [dict(LABEL)] if relevant is None else relevant}


def errors_for(queries):
    return validate(queries, TRANSCRIPTS)


def test_valid_queries_have_no_errors():
    assert errors_for([query("q01"), query("q02", "negative", relevant=[])]) == []


def test_validator_flags_duplicate_id():
    assert any("used 2 times" in e for e in errors_for([query("q01"), query("q01")]))


def test_validator_flags_unknown_type():
    assert any("unknown type" in e for e in errors_for([query(qtype="fuzzy")]))


def test_validator_flags_negative_with_labels():
    assert any("negative query has" in e for e in errors_for([query(qtype="negative")]))


def test_validator_flags_positive_without_labels():
    assert any("no relevant spans" in e for e in errors_for([query(relevant=[])]))


@pytest.mark.parametrize("start, end", [(20.0, 10.0), (15.0, 15.0)])
def test_validator_flags_start_not_before_end(start, end):
    label = {**LABEL, "start": start, "end": end}
    assert any("is not before end" in e for e in errors_for([query(relevant=[label])]))
