"""Tests for recorded execution history in pipeline snapshots."""

import json

import pytest

from tests.test_pipeline_status import (
    pipeline_root,
    collect_with_status,
)


def write_history(root, records):
    path = root / "monitoring/pipeline_executions.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records)
    )


def execution(stage="spark_audit", status="succeeded", **kwargs):
    return {
        "stage": stage,
        "status": status,
        "started_at": "2026-11-14T16:06:31+00:00",
        "finished_at": "2026-11-14T16:06:47+00:00",
        "duration_seconds": 15.4,
        "rows_processed": None,
        "throughput_rows_per_second": None,
        **kwargs,
    }


def test_no_history_available(pipeline_root):
    result = collect_with_status(pipeline_root, {})

    assert result["execution_history_available"] is False
    assert result["throughput_measured"] is False
    assert result["recent_executions"] == []


def test_successful_execution_history(pipeline_root):
    record = execution()
    record["rows_processed"] = 284807
    record["throughput_rows_per_second"] = 18495.64

    write_history(pipeline_root, [record])
    result = collect_with_status(pipeline_root, {})

    assert result["execution_history_available"] is True
    assert result["throughput_measured"] is True
    assert len(result["recent_executions"]) == 1
    assert result["recent_executions"][0]["status"] == "succeeded"
    assert result["recent_executions"][0]["rows_processed"] == 284807


def test_failed_execution_history(pipeline_root):
    write_history(pipeline_root, [
        execution(status="failed"),
    ])

    result = collect_with_status(pipeline_root, {})

    assert result["execution_history_available"] is True
    assert result["throughput_measured"] is False
    assert result["recent_executions"][0]["status"] == "failed"


def test_skipped_execution_history(pipeline_root):
    write_history(pipeline_root, [
        execution(status="skipped"),
    ])

    result = collect_with_status(pipeline_root, {})

    assert result["recent_executions"][0]["status"] == "skipped"
    assert result["throughput_measured"] is False


def test_only_latest_twenty_executions(pipeline_root):
    records = []

    for index in range(25):
        record = execution()
        record["duration_seconds"] = float(index)
        records.append(record)

    write_history(pipeline_root, records)
    result = collect_with_status(pipeline_root, {})

    assert len(result["recent_executions"]) == 20
    assert result["recent_executions"][0]["duration_seconds"] == 5.0
    assert result["recent_executions"][-1]["duration_seconds"] == 24.0


def test_invalid_stage_rejected(pipeline_root):
    write_history(pipeline_root, [
        execution(stage="train_baselines"),
    ])

    with pytest.raises(ValueError, match="Unexpected pipeline"):
        collect_with_status(pipeline_root, {})


def test_invalid_status_rejected(pipeline_root):
    write_history(pipeline_root, [
        execution(status="running"),
    ])

    with pytest.raises(ValueError, match="Unexpected pipeline"):
        collect_with_status(pipeline_root, {})


def test_invalid_json_rejected(pipeline_root):
    path = pipeline_root / "monitoring/pipeline_executions.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("invalid json\n")

    with pytest.raises(json.JSONDecodeError):
        collect_with_status(pipeline_root, {})
