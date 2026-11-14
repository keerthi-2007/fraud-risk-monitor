import json

from fastapi.testclient import TestClient

from backend.main import app
from backend import feedback_store


def feedback_payload():
    return {
        "features": {
            **{f"V{i}": 0.0 for i in range(1, 29)},
            "Amount": 50.0,
        },
        "risk_score": 0.10,
        "decision_threshold": 0.87292890258894895,
        "predicted_decision": "normal",
        "corrected_label": "legitimate",
        "model_run_id": "test-model",
    }


def test_feedback_accepts_valid_token(monkeypatch, tmp_path):
    monkeypatch.setenv("FEEDBACK_TOKEN", "test-feedback-token")
    feedback_file = tmp_path / "feedback.jsonl"

    monkeypatch.setattr(feedback_store, "FEEDBACK_DIR", tmp_path)
    monkeypatch.setattr(feedback_store, "FEEDBACK_FILE", feedback_file)

    response = TestClient(app).post(
        "/feedback",
        json=feedback_payload(),
        headers={"X-Feedback-Token": "test-feedback-token"},
    )

    assert response.status_code == 201
    assert response.json()["status"] == "stored"

    records = feedback_file.read_text().splitlines()
    assert len(records) == 1
    assert json.loads(records[0])["corrected_label"] == "legitimate"


def test_feedback_rejects_missing_token(monkeypatch, tmp_path):
    monkeypatch.setenv("FEEDBACK_TOKEN", "test-feedback-token")
    feedback_file = tmp_path / "feedback.jsonl"

    monkeypatch.setattr(feedback_store, "FEEDBACK_DIR", tmp_path)
    monkeypatch.setattr(feedback_store, "FEEDBACK_FILE", feedback_file)

    response = TestClient(app).post(
        "/feedback",
        json=feedback_payload(),
    )

    assert response.status_code == 403
    assert not feedback_file.exists()


def test_feedback_rejects_wrong_token(monkeypatch, tmp_path):
    monkeypatch.setenv("FEEDBACK_TOKEN", "test-feedback-token")
    feedback_file = tmp_path / "feedback.jsonl"

    monkeypatch.setattr(feedback_store, "FEEDBACK_DIR", tmp_path)
    monkeypatch.setattr(feedback_store, "FEEDBACK_FILE", feedback_file)

    response = TestClient(app).post(
        "/feedback",
        json=feedback_payload(),
        headers={"X-Feedback-Token": "wrong-token"},
    )

    assert response.status_code == 403
    assert not feedback_file.exists()


def test_feedback_rejects_unconfigured_token(monkeypatch, tmp_path):
    monkeypatch.delenv("FEEDBACK_TOKEN", raising=False)

    feedback_file = tmp_path / "feedback.jsonl"
    monkeypatch.setattr(feedback_store, "FEEDBACK_DIR", tmp_path)
    monkeypatch.setattr(feedback_store, "FEEDBACK_FILE", feedback_file)

    response = TestClient(app).post(
        "/feedback",
        json=feedback_payload(),
        headers={"X-Feedback-Token": "any-token"},
    )

    assert response.status_code == 403
    assert not feedback_file.exists()
