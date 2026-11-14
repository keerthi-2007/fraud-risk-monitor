"""Regression tests for feedback metrics exposed by the API."""

import json
from unittest.mock import patch

from backend import feedback_store
from backend.main import app
from tests.test_api import make_client


def test_metrics_export_feedback_counts(monkeypatch, tmp_path):
    path = tmp_path / "feedback.jsonl"

    record = {
        "predicted_decision": "normal",
        "corrected_label": "legitimate",
        "model_run_id": "test-model",
        "source_file": "test.csv",
        "row_number": 1,
    }

    path.write_text(json.dumps(record) + "\n")

    monkeypatch.setattr(feedback_store, "FEEDBACK_FILE", path)

    response = make_client().get("/metrics")

    assert response.status_code == 200
    assert (
        'fraud_reported_feedback_records_count'
        '{model_run_id="test-model"} 1.0'
    ) in response.text

    assert (
        'fraud_reported_feedback_confusion_count'
        '{category="tn",model_run_id="test-model"} 1.0'
    ) in response.text


def test_invalid_feedback_does_not_break_metrics(
    monkeypatch, tmp_path, caplog
):
    path = tmp_path / "feedback.jsonl"
    path.write_text("invalid json\n")

    monkeypatch.setattr(feedback_store, "FEEDBACK_FILE", path)

    response = make_client().get("/metrics")

    assert response.status_code == 200
    assert "fraud_api_requests_total" in response.text
    assert "Feedback monitoring refresh failed" in caplog.text


def test_feedback_refresh_io_error_is_handled(caplog):
    with patch(
        "backend.main.refresh_feedback_metrics",
        side_effect=OSError("Feedback storage unavailable"),
    ):
        response = make_client().get("/metrics")

    assert response.status_code == 200
    assert "Feedback monitoring refresh failed" in caplog.text
