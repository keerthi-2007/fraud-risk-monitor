"""Verify opt-in deployment orchestration without changing Docker."""

from unittest.mock import Mock

import pytest

from scripts import retraining_cycle as cycle


@pytest.mark.parametrize(
    "setting,expected_deploy",
    [
        (None, False),
        ("0", False),
        ("1", True),
    ],
)
def test_auto_deploy_switch(monkeypatch, setting, expected_deploy):
    if setting is None:
        monkeypatch.delenv("FRAUD_AUTO_DEPLOY", raising=False)
    else:
        monkeypatch.setenv("FRAUD_AUTO_DEPLOY", setting)

    monkeypatch.setattr(
        cycle,
        "run_once",
        Mock(return_value="trained_candidate"),
    )

    calls = []
    monkeypatch.setattr(
        cycle.subprocess,
        "run",
        lambda command, **kwargs: calls.append(
            (command, kwargs)
        ),
    )

    result = cycle.process_cycle()

    assert result == "trained_candidate"
    assert len(calls) == 2

    assert calls[0][0] == [
        cycle.sys.executable,
        "-m",
        "scripts.prepare_candidate_evaluation",
    ]

    expected_command = [
        cycle.sys.executable,
        "-m",
        "scripts.auto_promotion",
    ]

    if expected_deploy:
        expected_command.append("--deploy")

    assert calls[1][0] == expected_command
    assert all(options["check"] is True for _, options in calls)


def test_no_feedback_never_deploys(monkeypatch):
    monkeypatch.setenv("FRAUD_AUTO_DEPLOY", "1")
    monkeypatch.setattr(
        cycle,
        "run_once",
        Mock(return_value="pending_feedback"),
    )

    command = Mock()
    monkeypatch.setattr(cycle.subprocess, "run", command)

    assert cycle.process_cycle() == "pending_feedback"
    command.assert_not_called()
