"""Isolated tests for automatic retraining orchestration."""

from unittest.mock import Mock, call

import pytest

from scripts import retraining_cycle as cycle


@pytest.mark.parametrize(
    "result,dry_run,expected_calls",
    [
        ("pending_feedback", False, 0),
        ("trained_candidate", False, 2),
        ("already_processed", False, 2),
        ("eligible_dry_run", True, 0),
    ],
)
def test_retraining_orchestration(
    monkeypatch, result, dry_run, expected_calls
):
    retrain = Mock(return_value=result)
    evaluate = Mock()

    monkeypatch.setattr(cycle, "run_once", retrain)
    monkeypatch.setattr(cycle.subprocess, "run", evaluate)

    actual = cycle.process_cycle(
        min_feedback=20,
        min_fraud=3,
        dry_run=dry_run,
    )

    assert actual == result

    retrain.assert_called_once_with(
        min_feedback=20,
        min_fraud=3,
        dry_run=dry_run,
    )

    assert evaluate.call_count == expected_calls

    if expected_calls:
        assert evaluate.call_args_list == [
            call(
                [cycle.sys.executable, "-m",
                 "scripts.prepare_candidate_evaluation"],
                cwd=cycle.ROOT,
                check=True,
            ),
            call(
                [cycle.sys.executable, "-m",
                 "scripts.auto_promotion"],
                cwd=cycle.ROOT,
                check=True,
            ),
        ]


def test_evaluation_failure_is_not_hidden(monkeypatch):
    monkeypatch.setattr(
        cycle,
        "run_once",
        Mock(return_value="trained_candidate"),
    )

    monkeypatch.setattr(
        cycle.subprocess,
        "run",
        Mock(side_effect=RuntimeError("Evaluation failed")),
    )

    with pytest.raises(RuntimeError, match="Evaluation failed"):
        cycle.process_cycle()
