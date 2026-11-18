"""Test deployment and rollback without touching real Docker services."""

import sys

import pytest

from scripts import deploy_candidate as deployer


PREVIOUS = {
    "MODEL_URI": "models:/FraudRiskClassifier/1",
    "EVALUATION_RUN_ID": "original-evaluation",
    "DECISION_THRESHOLD": "0.87292890258894895",
}

CANDIDATE = ("new-model-run", "new-evaluation-run", 0.75)


def test_no_candidate_blocks_deployment(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["deploy_candidate.py", "--deploy"])
    monkeypatch.setattr(deployer, "verify_candidate", lambda: None)

    def forbidden(*args, **kwargs):
        raise AssertionError("Docker must not be accessed")

    monkeypatch.setattr(deployer, "deploy", forbidden)
    monkeypatch.setattr(deployer, "current_configuration", forbidden)

    deployer.main()


def test_default_mode_does_not_deploy(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["deploy_candidate.py"])
    monkeypatch.setattr(deployer, "verify_candidate", lambda: CANDIDATE)

    def forbidden(*args, **kwargs):
        raise AssertionError("Deployment must not happen")

    monkeypatch.setattr(deployer, "deploy", forbidden)
    monkeypatch.setattr(deployer, "current_configuration", forbidden)

    deployer.main()


def setup_deployment(monkeypatch, deploy_calls):
    monkeypatch.setattr(sys, "argv", ["deploy_candidate.py", "--deploy"])
    monkeypatch.setattr(deployer, "verify_candidate", lambda: CANDIDATE)
    monkeypatch.setattr(deployer, "save_active_model", lambda *args: None)
    monkeypatch.setattr(
        deployer, "current_configuration", lambda: PREVIOUS.copy()
    )
    monkeypatch.setattr(
        deployer,
        "health",
        lambda: {"status": "ok", "model_run_id": "original-model"},
    )
    monkeypatch.setattr(
        deployer,
        "deploy",
        lambda *args: deploy_calls.append(args),
    )


def test_successful_deployment(monkeypatch, tmp_path):
    calls = []
    setup_deployment(monkeypatch, calls)
    (tmp_path / ".env").write_text("MONITORING_TOKEN=test-token\n")
    monkeypatch.setattr(deployer, "ROOT", tmp_path)

    verified = []
    monkeypatch.setattr(
        deployer, "wait_for_model", lambda model_id: verified.append(model_id)
    )

    saved = []
    monkeypatch.setattr(
        deployer, "save_active_model", lambda *args: saved.append(args)
    )

    deployer.main()

    assert calls == [
        ("runs:/new-model-run/model", "new-evaluation-run", 0.75)
    ]
    assert verified == ["new-model-run"]
    assert saved == [
        (
            deployer.ROOT / ".env",
            "runs:/new-model-run/model",
            "new-evaluation-run",
            0.75,
        )
    ]


def test_failed_health_check_triggers_rollback(monkeypatch, tmp_path):
    calls = []
    setup_deployment(monkeypatch, calls)
    (tmp_path / ".env").write_text("MONITORING_TOKEN=test-token\n")
    monkeypatch.setattr(deployer, "ROOT", tmp_path)

    def verify_health(model_id):
        if model_id == "new-model-run":
            raise RuntimeError("Simulated health-check failure")
        assert model_id == "original-model"

    monkeypatch.setattr(deployer, "wait_for_model", verify_health)

    with pytest.raises(RuntimeError, match="Simulated health-check failure"):
        deployer.main()

    assert calls == [
        ("runs:/new-model-run/model", "new-evaluation-run", 0.75),
        (
            "models:/FraudRiskClassifier/1",
            "original-evaluation",
            "0.87292890258894895",
        ),
    ]


def test_failed_deployment_preserves_settings(monkeypatch, tmp_path):
    calls = []
    setup_deployment(monkeypatch, calls)

    env = tmp_path / ".env"
    original = (
        "MONITORING_TOKEN=test-token\n"
        "ACTIVE_MODEL_URI=models:/FraudRiskClassifier/1\n"
    )
    env.write_text(original)

    monkeypatch.setattr(deployer, "ROOT", tmp_path)

    saved = []
    monkeypatch.setattr(
        deployer,
        "save_active_model",
        lambda *args: saved.append(args),
    )

    def check_health(model_id):
        if model_id == "new-model-run":
            raise RuntimeError("Simulated deployment failure")
        assert model_id == "original-model"

    monkeypatch.setattr(deployer, "wait_for_model", check_health)

    with pytest.raises(RuntimeError, match="Simulated deployment failure"):
        deployer.main()

    assert len(calls) == 2
    assert calls[1][0] == "models:/FraudRiskClassifier/1"
    assert saved == []
    assert env.read_text() == original
