"""Tests for the isolated retraining monitoring API."""

import json

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend import retraining_status as status


def make_client(tmp_path, monkeypatch):
    monkeypatch.setenv("MONITORING_TOKEN", "test-secret")
    monkeypatch.setattr(
        status,
        "SNAPSHOT_PATH",
        tmp_path / "retraining_monitor.json",
    )
    app = FastAPI()
    app.include_router(status.router)
    return TestClient(app)


def sample():
    return {
        "schema_version": 1,
        "checked_at": "2026-11-18T12:00:00+00:00",
        "feedback_received": 7,
        "not_approved": 7,
        "invalid": 0,
        "eligible": 0,
        "eligible_fraud": 0,
        "eligible_legitimate": 0,
        "required_feedback": 20,
        "required_per_class": 3,
        "threshold_met": False,
    }


def test_rejects_missing_or_wrong_token(tmp_path, monkeypatch):
    client = make_client(tmp_path, monkeypatch)

    assert client.get(
        "/monitoring/retraining-status"
    ).status_code == 403

    assert client.get(
        "/monitoring/retraining-status",
        headers={"X-Monitoring-Token": "wrong"},
    ).status_code == 403


def test_missing_snapshot_returns_503(tmp_path, monkeypatch):
    client = make_client(tmp_path, monkeypatch)

    response = client.get(
        "/monitoring/retraining-status",
        headers={"X-Monitoring-Token": "test-secret"},
    )
    assert response.status_code == 503


def test_valid_status_and_no_private_data(tmp_path, monkeypatch):
    client = make_client(tmp_path, monkeypatch)

    snapshot = sample()
    snapshot["transaction_features"] = {"V1": 999}
    status.SNAPSHOT_PATH.write_text(json.dumps(snapshot))

    response = client.get(
        "/monitoring/retraining-status",
        headers={"X-Monitoring-Token": "test-secret"},
    )

    assert response.status_code == 200
    assert response.json()["feedback_received"] == 7
    assert response.json()["eligible"] == 0
    assert "transaction_features" not in response.json()


def test_inconsistent_status_rejected(tmp_path, monkeypatch):
    client = make_client(tmp_path, monkeypatch)

    snapshot = sample()
    snapshot["threshold_met"] = True

    status.SNAPSHOT_PATH.write_text(json.dumps(snapshot))

    response = client.get(
        "/monitoring/retraining-status",
        headers={"X-Monitoring-Token": "test-secret"},
    )

    assert response.status_code == 503
