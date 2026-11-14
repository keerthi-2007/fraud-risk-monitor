"""Tests for Prometheus feedback monitoring."""

import json

import pytest

from backend.feedback_monitor import (
    FEEDBACK_CONFUSION,
    FEEDBACK_DUPLICATES,
    FEEDBACK_RECORDS,
    refresh_feedback_metrics,
)


def make_record(predicted, reported, model, row):
    return {
        "predicted_decision": predicted,
        "corrected_label": reported,
        "model_run_id": model,
        "source_file": "test_transactions.csv",
        "row_number": row,
    }


def write_records(path, records):
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records)
    )


@pytest.fixture(autouse=True)
def reset_metrics():
    FEEDBACK_CONFUSION.clear()
    FEEDBACK_DUPLICATES.clear()
    FEEDBACK_RECORDS.clear()
    yield
    FEEDBACK_CONFUSION.clear()
    FEEDBACK_DUPLICATES.clear()
    FEEDBACK_RECORDS.clear()


def test_feedback_metrics_exported(tmp_path):
    path = tmp_path / "feedback.jsonl"

    write_records(path, [
        make_record("review", "fraud", "model-a", 1),
        make_record("normal", "legitimate", "model-a", 2),
        make_record("normal", "fraud", "model-a", 3),
    ])

    result = refresh_feedback_metrics(path)

    assert result["model-a"]["reported_feedback_count"] == 3

    assert FEEDBACK_RECORDS.labels(
        model_run_id="model-a"
    )._value.get() == 3

    assert FEEDBACK_CONFUSION.labels(
        model_run_id="model-a",
        category="tp",
    )._value.get() == 1

    assert FEEDBACK_CONFUSION.labels(
        model_run_id="model-a",
        category="tn",
    )._value.get() == 1

    assert FEEDBACK_CONFUSION.labels(
        model_run_id="model-a",
        category="fn",
    )._value.get() == 1


def test_model_versions_and_duplicates(tmp_path):
    path = tmp_path / "feedback.jsonl"

    write_records(path, [
        make_record("review", "fraud", "model-a", 1),
        make_record("review", "fraud", "model-a", 1),
        make_record("normal", "legitimate", "model-b", 2),
    ])

    refresh_feedback_metrics(path)

    assert FEEDBACK_RECORDS.labels(
        model_run_id="model-a"
    )._value.get() == 1

    assert FEEDBACK_RECORDS.labels(
        model_run_id="model-b"
    )._value.get() == 1

    assert FEEDBACK_DUPLICATES.labels(
        model_run_id="model-a"
    )._value.get() == 1


def test_invalid_feedback_preserves_previous_metrics(tmp_path):
    path = tmp_path / "feedback.jsonl"

    write_records(path, [
        make_record("review", "fraud", "model-a", 1),
    ])

    refresh_feedback_metrics(path)

    before = FEEDBACK_RECORDS.labels(
        model_run_id="model-a"
    )._value.get()

    path.write_text("invalid json\n")

    with pytest.raises(ValueError, match="line 1"):
        refresh_feedback_metrics(path)

    after = FEEDBACK_RECORDS.labels(
        model_run_id="model-a"
    )._value.get()

    assert before == 1
    assert after == before
