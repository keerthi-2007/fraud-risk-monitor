"""Tests for real DVC stage execution logging."""

import json
import subprocess
from unittest.mock import patch

import pytest

from scripts.run_pipeline_stage import run_stage


def read_history(path):
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def test_successful_execution(tmp_path):
    history = tmp_path / "history.jsonl"

    result = subprocess.CompletedProcess(
        args=["dvc"],
        returncode=0,
        stdout="Running stage 'spark_audit':\nDone.\n",
        stderr="",
    )

    with patch(
        "scripts.run_pipeline_stage.subprocess.run",
        return_value=result,
    ) as mock_run:
        record = run_stage(
            "spark_audit",
            force=True,
            history_path=history,
        )

    assert record["status"] == "succeeded"
    assert record["execution_confirmed"] is True
    assert record["returncode"] == 0
    assert record["duration_seconds"] >= 0

    assert mock_run.call_args.args[0] == [
        "dvc",
        "repro",
        "--single-item",
        "--force",
        "--no-run-cache",
        "spark_audit",
    ]

    assert read_history(history)[0]["status"] == "succeeded"


def test_skipped_execution(tmp_path):
    history = tmp_path / "history.jsonl"

    result = subprocess.CompletedProcess(
        args=["dvc"],
        returncode=0,
        stdout="Stage 'spark_audit' didn't change, skipping\n",
        stderr="",
    )

    with patch(
        "scripts.run_pipeline_stage.subprocess.run",
        return_value=result,
    ):
        record = run_stage(
            "spark_audit",
            history_path=history,
        )

    assert record["status"] == "skipped"
    assert record["execution_confirmed"] is False
    assert read_history(history)[0]["status"] == "skipped"


def test_failed_execution(tmp_path):
    history = tmp_path / "history.jsonl"

    result = subprocess.CompletedProcess(
        args=["dvc"],
        returncode=1,
        stdout="Running stage 'spark_audit':\n",
        stderr="Spark audit failed",
    )

    with patch(
        "scripts.run_pipeline_stage.subprocess.run",
        return_value=result,
    ):
        record = run_stage(
            "spark_audit",
            history_path=history,
        )

    assert record["status"] == "failed"
    assert record["returncode"] == 1
    assert record["execution_confirmed"] is False
    assert read_history(history)[0]["status"] == "failed"


def test_missing_dvc_executable(tmp_path):
    history = tmp_path / "history.jsonl"

    with patch(
        "scripts.run_pipeline_stage.subprocess.run",
        side_effect=FileNotFoundError("dvc not found"),
    ):
        record = run_stage(
            "spark_audit",
            history_path=history,
        )

    assert record["status"] == "failed"
    assert record["returncode"] == -1
    assert len(read_history(history)) == 1


def test_unsupported_stage_never_executes(tmp_path):
    history = tmp_path / "history.jsonl"

    with patch(
        "scripts.run_pipeline_stage.subprocess.run"
    ) as mock_run:
        with pytest.raises(ValueError, match="Unsupported DVC stage"):
            run_stage(
                "train_baselines",
                force=True,
                history_path=history,
            )

    mock_run.assert_not_called()
    assert not history.exists()


def test_execution_history_appends(tmp_path):
    history = tmp_path / "history.jsonl"

    result = subprocess.CompletedProcess(
        args=["dvc"],
        returncode=0,
        stdout="Stage unchanged, skipping\n",
        stderr="",
    )

    with patch(
        "scripts.run_pipeline_stage.subprocess.run",
        return_value=result,
    ):
        run_stage("spark_audit", history_path=history)
        run_stage("spark_audit", history_path=history)

    records = read_history(history)

    assert len(records) == 2
    assert records[0]["execution_id"] != records[1]["execution_id"]
    assert all(record["status"] == "skipped" for record in records)
