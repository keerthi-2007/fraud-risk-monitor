"""Isolated tests for automatic model promotion."""

from unittest.mock import Mock

import pytest

from scripts import auto_promotion as promotion


MODEL_ID = "a" * 32
EVALUATION_ID = "b" * 32
THRESHOLD = 0.75
TARGET_URI = f"runs:/{MODEL_ID}/model"


def setup_candidate(monkeypatch):
    monkeypatch.setattr(
        promotion,
        "verify_candidate",
        lambda: (MODEL_ID, EVALUATION_ID, THRESHOLD),
    )


def test_no_candidate(monkeypatch):
    monkeypatch.setattr(promotion, "verify_candidate", lambda: None)
    health = Mock()
    monkeypatch.setattr(promotion, "health", health)

    assert promotion.process_promotion(deploy=True) == "no_candidate"
    health.assert_not_called()


def test_verified_candidate_dry_run(monkeypatch):
    setup_candidate(monkeypatch)

    monkeypatch.setattr(
        promotion,
        "health",
        lambda: {"status": "ok", "model_run_id": "original-model"},
    )

    deploy = Mock()
    monkeypatch.setattr(promotion.subprocess, "run", deploy)

    assert promotion.process_promotion(deploy=False) == "ready"
    deploy.assert_not_called()


def test_already_deployed(monkeypatch):
    setup_candidate(monkeypatch)

    monkeypatch.setattr(
        promotion,
        "health",
        lambda: {"status": "ok", "model_run_id": MODEL_ID},
    )
    monkeypatch.setattr(
        promotion, "saved_model_uri", lambda: TARGET_URI
    )

    deploy = Mock()
    monkeypatch.setattr(promotion.subprocess, "run", deploy)

    assert promotion.process_promotion(deploy=True) == "already_deployed"
    deploy.assert_not_called()


def test_successful_promotion(monkeypatch):
    setup_candidate(monkeypatch)

    health = Mock(side_effect=[
        {"status": "ok", "model_run_id": "original-model"},
        {"status": "ok", "model_run_id": MODEL_ID},
    ])

    monkeypatch.setattr(promotion, "health", health)
    monkeypatch.setattr(
        promotion, "saved_model_uri", lambda: TARGET_URI
    )

    deploy = Mock()
    monkeypatch.setattr(promotion.subprocess, "run", deploy)

    assert promotion.process_promotion(deploy=True) == "deployed"
    deploy.assert_called_once()
    assert "--deploy" in deploy.call_args.args[0]
    assert deploy.call_args.kwargs["check"] is True


def test_unhealthy_backend_blocks_promotion(monkeypatch):
    setup_candidate(monkeypatch)

    monkeypatch.setattr(
        promotion,
        "health",
        lambda: {"status": "error", "model_run_id": "original-model"},
    )

    deploy = Mock()
    monkeypatch.setattr(promotion.subprocess, "run", deploy)

    with pytest.raises(RuntimeError, match="unhealthy"):
        promotion.process_promotion(deploy=True)

    deploy.assert_not_called()


def test_configuration_mismatch_rejected(monkeypatch):
    setup_candidate(monkeypatch)

    monkeypatch.setattr(
        promotion,
        "health",
        lambda: {"status": "ok", "model_run_id": MODEL_ID},
    )
    monkeypatch.setattr(
        promotion, "saved_model_uri", lambda: "wrong-model"
    )

    with pytest.raises(RuntimeError, match="disagree"):
        promotion.process_promotion(deploy=True)
