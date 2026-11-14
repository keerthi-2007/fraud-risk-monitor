"""Tests for read-only DVC pipeline status snapshots."""

import json
import subprocess
from unittest.mock import patch

import pytest
import yaml

from scripts.build_pipeline_status import collect_pipeline_status


@pytest.fixture
def pipeline_root(tmp_path):
    stages = {
        name: {
            "cmd": f"python scripts/{name}.py",
            "deps": [f"scripts/{name}.py"],
        }
        for name in (
            "prepare_data",
            "create_splits",
            "spark_audit",
            "build_drift_baseline",
        )
    }

    (tmp_path / "dvc.yaml").write_text(
        yaml.safe_dump({"stages": stages})
    )

    (tmp_path / "dvc.lock").write_text(
        yaml.safe_dump({
            "schema": "2.0",
            "stages": {
                name: {"cmd": definition["cmd"]}
                for name, definition in stages.items()
            },
        })
    )

    audit = tmp_path / "data/processed/spark_audit.json"
    audit.parent.mkdir(parents=True)
    audit.write_text(json.dumps({
        "spark_version": "4.3.0",
        "rows": 284807,
        "fraud_rows": 492,
        "missing_cells": 0,
    }))

    return tmp_path


def collect_with_status(root, status):
    with patch(
        "scripts.build_pipeline_status.subprocess.run",
        return_value=subprocess.CompletedProcess(
            args=["dvc", "status", "--json"],
            returncode=0,
            stdout=json.dumps(status),
            stderr="",
        ),
    ) as run:
        result = collect_pipeline_status(root)

    run.assert_called_once()
    assert run.call_args.kwargs["cwd"] == root
    assert run.call_args.kwargs["check"] is True

    return result


def test_all_stages_up_to_date(pipeline_root):
    result = collect_with_status(pipeline_root, {})

    assert result["overall_up_to_date"] is True
    assert len(result["stages"]) == 4
    assert all(stage["up_to_date"] for stage in result["stages"])
    assert result["spark_audit"]["rows"] == 284807
    assert result["throughput_measured"] is False
    assert result["execution_history_available"] is False


def test_modified_stage(pipeline_root):
    result = collect_with_status(
        pipeline_root,
        {"spark_audit": {"changed deps": {}}},
    )

    stages = {
        stage["name"]: stage
        for stage in result["stages"]
    }

    assert result["overall_up_to_date"] is False
    assert stages["spark_audit"]["up_to_date"] is False
    assert stages["prepare_data"]["up_to_date"] is True


def test_missing_lock_entry(pipeline_root):
    lock_path = pipeline_root / "dvc.lock"
    lock = yaml.safe_load(lock_path.read_text())
    del lock["stages"]["build_drift_baseline"]
    lock_path.write_text(yaml.safe_dump(lock))

    result = collect_with_status(pipeline_root, {})

    assert result["overall_up_to_date"] is False
    stages = {
        stage["name"]: stage
        for stage in result["stages"]
    }

    assert stages["build_drift_baseline"]["recorded_in_lock"] is False
    assert stages["build_drift_baseline"]["up_to_date"] is False


def test_missing_spark_audit(pipeline_root):
    path = pipeline_root / "data/processed/spark_audit.json"
    path.unlink()

    result = collect_with_status(pipeline_root, {})

    assert result["spark_audit"] is None


def test_invalid_dvc_json(pipeline_root):
    with patch(
        "scripts.build_pipeline_status.subprocess.run",
        return_value=subprocess.CompletedProcess(
            args=["dvc", "status", "--json"],
            returncode=0,
            stdout="[]",
            stderr="",
        ),
    ):
        with pytest.raises(ValueError, match="Unexpected DVC status"):
            collect_pipeline_status(pipeline_root)


def test_dvc_command_failure(pipeline_root):
    with patch(
        "scripts.build_pipeline_status.subprocess.run",
        side_effect=subprocess.CalledProcessError(
            1, ["dvc", "status", "--json"]
        ),
    ):
        with pytest.raises(subprocess.CalledProcessError):
            collect_pipeline_status(pipeline_root)
