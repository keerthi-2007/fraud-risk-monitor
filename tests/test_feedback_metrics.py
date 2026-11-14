"""Tests for feedback-based model performance summaries."""

import json

import pytest

from backend.feedback_metrics import (
    load_feedback_summary,
    summarise_feedback,
)


def record(
    predicted,
    reported,
    model="model-a",
    row=None,
):
    return {
        "predicted_decision": predicted,
        "corrected_label": reported,
        "model_run_id": model,
        "source_file": "transactions.csv",
        "row_number": row,
    }


def test_confusion_matrix_counts():
    records = [
        record("review", "fraud", row=1),
        record("review", "legitimate", row=2),
        record("normal", "legitimate", row=3),
        record("normal", "fraud", row=4),
    ]

    result = summarise_feedback(records)["model-a"]

    assert result["true_positive"] == 1
    assert result["false_positive"] == 1
    assert result["true_negative"] == 1
    assert result["false_negative"] == 1
    assert result["reported_feedback_count"] == 4
    assert result["metrics_available"] is False
    assert result["reported_precision"] is None
    assert result["reported_recall"] is None


def test_model_versions_are_separated():
    records = [
        record("normal", "legitimate", "model-a", 1),
        record("review", "fraud", "model-b", 2),
        record("normal", "fraud", None, 3),
    ]

    result = summarise_feedback(records)

    assert result["model-a"]["true_negative"] == 1
    assert result["model-b"]["true_positive"] == 1
    assert result["unknown"]["false_negative"] == 1


def test_duplicate_source_reference_excluded():
    records = [
        record("review", "fraud", row=1),
        record("review", "fraud", row=1),
        record("normal", "legitimate", row=2),
    ]

    result = summarise_feedback(records)["model-a"]

    assert result["reported_feedback_count"] == 2
    assert result["duplicate_references_excluded"] == 1


def test_precision_and_recall_from_sufficient_feedback():
    records = (
        [record("review", "fraud", row=i) for i in range(1, 11)]
        + [record("review", "legitimate", row=i) for i in range(11, 16)]
        + [record("normal", "fraud", row=i) for i in range(16, 21)]
        + [record("normal", "legitimate", row=i) for i in range(21, 31)]
    )

    result = summarise_feedback(records)["model-a"]

    assert result["reported_feedback_count"] == 30
    assert result["metrics_available"] is True
    assert result["reported_precision"] == pytest.approx(10 / 15)
    assert result["reported_recall"] == pytest.approx(10 / 15)
    assert result["label_source"] == "user_reported_unverified"


def test_no_predicted_positives_suppresses_rates():
    records = [
        record("normal", "fraud", row=i)
        for i in range(1, 31)
    ]

    result = summarise_feedback(records)["model-a"]

    assert result["metrics_available"] is False
    assert result["reported_precision"] is None
    assert result["reported_recall"] is None


def test_invalid_predicted_decision_rejected():
    records = [record("unknown", "fraud", row=1)]

    with pytest.raises(ValueError, match="Invalid predicted decision"):
        summarise_feedback(records)


def test_empty_feedback_file(tmp_path):
    path = tmp_path / "feedback.jsonl"

    assert load_feedback_summary(path) == {}


def test_feedback_file_loaded(tmp_path):
    path = tmp_path / "feedback.jsonl"
    path.write_text(
        json.dumps(record("normal", "legitimate", row=1)) + "\n"
    )

    result = load_feedback_summary(path)

    assert result["model-a"]["true_negative"] == 1


def test_invalid_json_rejected(tmp_path):
    path = tmp_path / "feedback.jsonl"
    path.write_text("not valid json\n")

    with pytest.raises(ValueError, match="line 1"):
        load_feedback_summary(path)
