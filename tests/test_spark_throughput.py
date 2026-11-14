"""Tests for Spark throughput recorded by the DVC stage runner."""

import json
import subprocess
from unittest.mock import patch

import pytest

import scripts.run_pipeline_stage as runner


@pytest.fixture
def audit_root(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "ROOT", tmp_path)

    audit_path = tmp_path / "data/processed/spark_audit.json"
    audit_path.parent.mkdir(parents=True)
    audit_path.write_text(json.dumps({"rows": 284807}))

    return tmp_path


def run_mocked(root, stdout, returncode=0):
    result = subprocess.CompletedProcess(
        args=["dvc"],
        returncode=returncode,
        stdout=stdout,
        stderr="",
    )

    with (
        patch.object(runner.subprocess, "run", return_value=result),
        patch.object(runner, "perf_counter", side_effect=[100.0, 110.0]),
    ):
        return runner.run_stage(
            "spark_audit",
            force=True,
            history_path=root / "history.jsonl",
        )


def test_successful_spark_throughput(audit_root):
    record = run_mocked(
        audit_root,
        "Running stage 'spark_audit':\nCompleted.\n",
    )

    assert record["status"] == "succeeded"
    assert record["duration_seconds"] == 10
    assert record["rows_processed"] == 284807
    assert record["throughput_rows_per_second"] == pytest.approx(
        28480.70
    )


def test_skipped_stage_has_no_throughput(audit_root):
    record = run_mocked(
        audit_root,
        "Stage 'spark_audit' didn't change, skipping\n",
    )

    assert record["status"] == "skipped"
    assert record["rows_processed"] is None
    assert record["throughput_rows_per_second"] is None


def test_failed_stage_has_no_throughput(audit_root):
    record = run_mocked(
        audit_root,
        "Running stage 'spark_audit':\n",
        returncode=1,
    )

    assert record["status"] == "failed"
    assert record["rows_processed"] is None
    assert record["throughput_rows_per_second"] is None


def test_missing_audit_file_has_no_throughput(audit_root):
    (
        audit_root / "data/processed/spark_audit.json"
    ).unlink()

    record = run_mocked(
        audit_root,
        "Running stage 'spark_audit':\nCompleted.\n",
    )

    assert record["status"] == "succeeded"
    assert record["rows_processed"] is None
    assert record["throughput_rows_per_second"] is None


@pytest.mark.parametrize("rows", [0, -1, "284807", True])
def test_invalid_row_count_has_no_throughput(audit_root, rows):
    audit_path = audit_root / "data/processed/spark_audit.json"
    audit_path.write_text(json.dumps({"rows": rows}))

    record = run_mocked(
        audit_root,
        "Running stage 'spark_audit':\nCompleted.\n",
    )

    assert record["rows_processed"] is None
    assert record["throughput_rows_per_second"] is None
