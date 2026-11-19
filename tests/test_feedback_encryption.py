"""Encryption-at-rest tests for stored feedback."""

import json

import pytest
from cryptography.fernet import Fernet

from backend import feedback_crypto, feedback_store
from tests.test_api import make_client


def payload():
    return {
        "features": {**{f"V{i}": float(i) for i in range(1, 29)}, "Amount": 42.5},
        "risk_score": 0.9,
        "decision_threshold": 0.87,
        "predicted_decision": "review",
        "corrected_label": "fraud",
    }


@pytest.fixture
def key(monkeypatch):
    value = Fernet.generate_key().decode()
    monkeypatch.setenv(feedback_crypto.KEY_ENV, value)
    return value


@pytest.fixture
def store(monkeypatch, tmp_path):
    path = tmp_path / "feedback.jsonl"
    monkeypatch.setattr(feedback_store, "FEEDBACK_DIR", tmp_path)
    monkeypatch.setattr(feedback_store, "FEEDBACK_FILE", path)
    monkeypatch.setenv("FEEDBACK_TOKEN", "t")
    return path


def test_feedback_features_are_encrypted_on_disk(key, store):
    response = make_client().post(
        "/feedback", json=payload(), headers={"X-Feedback-Token": "t"}
    )
    assert response.status_code == 201

    raw = store.read_text()
    record = json.loads(raw)
    assert "features" not in record
    assert record["encryption"] == "fernet-v1"
    assert "42.5" not in raw
    assert feedback_crypto.record_features(record)["Amount"] == 42.5


def test_plaintext_when_no_key(store):
    make_client().post("/feedback", json=payload(), headers={"X-Feedback-Token": "t"})
    record = json.loads(store.read_text())
    assert record["features"]["Amount"] == 42.5


def test_missing_key_refuses_to_read(key, monkeypatch):
    record = feedback_crypto.protect_features({"Amount": 1.0})
    monkeypatch.delenv(feedback_crypto.KEY_ENV)
    with pytest.raises(feedback_crypto.FeedbackKeyMissing):
        feedback_crypto.record_features(record)


def test_tampered_token_rejected(key):
    record = feedback_crypto.protect_features({"Amount": 1.0})
    record["features_encrypted"] = record["features_encrypted"][:-4] + "AAAA"
    with pytest.raises(ValueError):
        feedback_crypto.record_features(record)


def test_wrong_key_rejected(key, monkeypatch):
    record = feedback_crypto.protect_features({"Amount": 1.0})
    monkeypatch.setenv(feedback_crypto.KEY_ENV, Fernet.generate_key().decode())
    with pytest.raises(ValueError):
        feedback_crypto.record_features(record)


def test_key_read_from_env_file(monkeypatch, tmp_path):
    value = Fernet.generate_key().decode()
    env_file = tmp_path / ".env"
    env_file.write_text(f"OTHER=1\n{feedback_crypto.KEY_ENV}={value}\n")
    monkeypatch.setattr(feedback_crypto, "ENV_FILE", env_file)
    assert feedback_crypto.encryption_enabled()
