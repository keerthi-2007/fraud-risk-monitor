"""Failure-injection tests for atomic deployment rollback."""

import os
import stat
import sys

import pytest

from scripts import deploy_candidate as deployer


ORIGINAL_RUN = "27da3507bddb413bab9cd0a12aeb092a"
ORIGINAL_EVAL = "008277161614417d8da7e13727013676"
CANDIDATE_RUN = "a" * 32
CANDIDATE_EVAL = "b" * 32


@pytest.mark.parametrize("rollback_fails", [False, True])
def test_original_env_restored_after_save_failure(
    tmp_path, monkeypatch, rollback_fails
):
    env_path = tmp_path / ".env"

    original = (
        "MONITORING_TOKEN=synthetic-test-value\n"
        "ACTIVE_MODEL_URI=models:/FraudRiskClassifier/1\n"
        f"ACTIVE_EVALUATION_RUN_ID={ORIGINAL_EVAL}\n"
        "ACTIVE_DECISION_THRESHOLD=0.87292890258894895\n"
    ).encode("utf-8")

    env_path.write_bytes(original)
    os.chmod(env_path, 0o600)

    monkeypatch.setattr(deployer, "ROOT", tmp_path)
    monkeypatch.setattr(
        sys, "argv", ["deploy_candidate.py", "--deploy"]
    )

    monkeypatch.setattr(
        deployer,
        "verify_candidate",
        lambda: (CANDIDATE_RUN, CANDIDATE_EVAL, 0.8),
    )

    previous = {
        "MODEL_URI": "models:/FraudRiskClassifier/1",
        "EVALUATION_RUN_ID": ORIGINAL_EVAL,
        "DECISION_THRESHOLD": "0.87292890258894895",
    }

    monkeypatch.setattr(
        deployer,
        "current_configuration",
        lambda: previous.copy(),
    )

    monkeypatch.setattr(
        deployer,
        "health",
        lambda: {
            "status": "ok",
            "model_run_id": ORIGINAL_RUN,
        },
    )

    deployment_calls = []

    def fake_deploy(model_uri, evaluation_id, threshold):
        deployment_calls.append(
            (model_uri, evaluation_id, threshold)
        )

        if rollback_fails and len(deployment_calls) == 2:
            raise RuntimeError("SIMULATED_ROLLBACK_FAILURE")

    monkeypatch.setattr(
        deployer, "deploy", fake_deploy
    )

    health_checks = []

    monkeypatch.setattr(
        deployer,
        "wait_for_model",
        lambda run_id: health_checks.append(run_id),
    )

    # Actually write to the temporary .env, then inject a failure.
    original_save = deployer.save_active_model

    def save_then_fail(path, model_uri, evaluation_id, threshold):
        original_save(
            path, model_uri, evaluation_id, threshold
        )

        assert path.read_bytes() != original

        raise RuntimeError("SIMULATED_POST_SAVE_FAILURE")

    monkeypatch.setattr(
        deployer, "save_active_model", save_then_fail
    )

    expected_error = (
        "SIMULATED_ROLLBACK_FAILURE"
        if rollback_fails
        else "SIMULATED_POST_SAVE_FAILURE"
    )

    with pytest.raises(RuntimeError, match=expected_error):
        deployer.main()

    # The exact original bytes and permissions must be restored.
    assert env_path.read_bytes() == original

    assert stat.S_IMODE(
        env_path.stat().st_mode
    ) == 0o600

    # Both the candidate attempt and rollback must be requested.
    assert deployment_calls == [
        (
            f"runs:/{CANDIDATE_RUN}/model",
            CANDIDATE_EVAL,
            0.8,
        ),
        (
            "models:/FraudRiskClassifier/1",
            ORIGINAL_EVAL,
            "0.87292890258894895",
        ),
    ]

    assert health_checks[0] == CANDIDATE_RUN

    if not rollback_fails:
        assert health_checks[-1] == ORIGINAL_RUN
