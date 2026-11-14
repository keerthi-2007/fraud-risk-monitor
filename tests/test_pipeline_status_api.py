"""Tests for the secured pipeline status endpoint."""

import json

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend import pipeline_status


def make_client():
    app = FastAPI()
    app.include_router(pipeline_status.router)
    return TestClient(app)


def valid_snapshot():
    return {
        "schema_version": 1,
        "generated_at": "2026-11-14T15:44:29+00:00",
        "overall_up_to_date": True,
        "stages": [
            {
                "name": "prepare_data",
                "recorded_in_lock": True,
                "up_to_date": True,
            }
        ],
        "spark_audit": {"rows": 284807},
    }


def test_valid_snapshot(monkeypatch, tmp_path):
    path = tmp_path / "pipeline_status.json"
    path.write_text(json.dumps(valid_snapshot()))

    monkeypatch.setattr(pipeline_status, "SNAPSHOT_PATH", path)
    monkeypatch.setenv("MONITORING_TOKEN", "test-secret")

    # The route function's default path is bound at definition time,
    # so patch the loader for this isolated API test.
    original_loader = pipeline_status.load_pipeline_snapshot
    monkeypatch.setattr(
        pipeline_status,
        "load_pipeline_snapshot",
        lambda: original_loader(path)
    )

    response = make_client().get(
        "/monitoring/pipeline-status",
        headers={"X-Monitoring-Token": "test-secret"},
    )

    assert response.status_code == 200
    assert response.json()["overall_up_to_date"] is True
    assert response.json()["spark_audit"]["rows"] == 284807


def test_missing_token_rejected(monkeypatch):
    monkeypatch.setenv("MONITORING_TOKEN", "test-secret")

    response = make_client().get("/monitoring/pipeline-status")

    assert response.status_code == 403


def test_wrong_token_rejected(monkeypatch):
    monkeypatch.setenv("MONITORING_TOKEN", "test-secret")

    response = make_client().get(
        "/monitoring/pipeline-status",
        headers={"X-Monitoring-Token": "wrong"},
    )

    assert response.status_code == 403


def test_missing_configuration_rejected(monkeypatch):
    monkeypatch.delenv("MONITORING_TOKEN", raising=False)

    response = make_client().get(
        "/monitoring/pipeline-status",
        headers={"X-Monitoring-Token": "some-token"},
    )

    assert response.status_code == 403


def test_missing_snapshot_returns_503(tmp_path):
    from fastapi import HTTPException

    missing = tmp_path / "missing.json"

    try:
        pipeline_status.load_pipeline_snapshot(missing)
    except HTTPException as exc:
        assert exc.status_code == 503
    else:
        raise AssertionError("Missing snapshot was accepted.")


def test_invalid_json_returns_503(tmp_path):
    from fastapi import HTTPException

    path = tmp_path / "invalid.json"
    path.write_text("invalid json")

    try:
        pipeline_status.load_pipeline_snapshot(path)
    except HTTPException as exc:
        assert exc.status_code == 503
    else:
        raise AssertionError("Invalid JSON was accepted.")


def test_invalid_structure_returns_503(tmp_path):
    from fastapi import HTTPException

    path = tmp_path / "invalid_structure.json"
    path.write_text(json.dumps({"schema_version": 1}))

    try:
        pipeline_status.load_pipeline_snapshot(path)
    except HTTPException as exc:
        assert exc.status_code == 503
    else:
        raise AssertionError("Invalid snapshot structure was accepted.")
