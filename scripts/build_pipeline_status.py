"""Generate a read-only status snapshot for the AIOps pipeline UI."""

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "monitoring" / "pipeline_status.json"


def collect_pipeline_status(root=ROOT):
    """Collect current DVC and Spark metadata without executing stages."""
    root = Path(root)

    pipeline = yaml.safe_load(
        (root / "dvc.yaml").read_text(encoding="utf-8")
    )
    lock = yaml.safe_load(
        (root / "dvc.lock").read_text(encoding="utf-8")
    )

    result = subprocess.run(
        ["dvc", "status", "--json"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )

    dvc_status = json.loads(result.stdout)

    if not isinstance(dvc_status, dict):
        raise ValueError("Unexpected DVC status response.")

    defined_stages = pipeline.get("stages", {})
    locked_stages = lock.get("stages", {})

    stages = []

    for name, definition in defined_stages.items():
        stages.append({
            "name": name,
            "command": definition.get("cmd"),
            "dependencies": definition.get("deps", []),
            "recorded_in_lock": name in locked_stages,
            "up_to_date": (
                name in locked_stages
                and name not in dvc_status
            ),
        })

    spark_path = root / "data/processed/spark_audit.json"
    spark_audit = None

    if spark_path.exists():
        report = json.loads(
            spark_path.read_text(encoding="utf-8")
        )

        spark_audit = {
            "spark_version": report.get("spark_version"),
            "rows": report.get("rows"),
            "fraud_rows": report.get("fraud_rows"),
            "missing_cells": report.get("missing_cells"),
        }

    # Include only verified fields from recorded DVC executions.
    history_path = root / "monitoring/pipeline_executions.jsonl"
    executions = []

    if history_path.exists():
        with history_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue

                record = json.loads(line)

                if not isinstance(record, dict):
                    raise ValueError("Invalid pipeline execution record.")

                stage_name = record.get("stage")
                stage_status = record.get("status")

                if (
                    stage_name not in defined_stages
                    or stage_status not in {
                        "succeeded", "failed", "skipped"
                    }
                ):
                    raise ValueError(
                        "Unexpected pipeline execution stage or status."
                    )

                executions.append({
                    "stage": stage_name,
                    "status": stage_status,
                    "started_at": record.get("started_at"),
                    "finished_at": record.get("finished_at"),
                    "duration_seconds": record.get("duration_seconds"),
                    "rows_processed": record.get("rows_processed"),
                    "throughput_rows_per_second": record.get(
                        "throughput_rows_per_second"
                    ),
                })

    executions = executions[-20:]

    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pipeline_tool": "DVC",
        "overall_up_to_date": (
            not dvc_status
            and all(stage["up_to_date"] for stage in stages)
        ),
        "dvc_status": dvc_status,
        "stages": stages,
        "spark_audit": spark_audit,
        "training_and_evaluation": {
            "orchestration": "MLproject / controlled manual execution",
            "included_in_dvc": False,
        },
        "execution_history_available": bool(executions),
        "throughput_measured": any(
            item["status"] == "succeeded"
            and isinstance(item["throughput_rows_per_second"], (int, float))
            and not isinstance(item["throughput_rows_per_second"], bool)
            and item["throughput_rows_per_second"] > 0
            for item in executions
        ),
        "recent_executions": executions,
    }


def main():
    """Collect DVC pipeline status and write the snapshot JSON."""
    status = collect_pipeline_status()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(status, indent=2) + "\n",
        encoding="utf-8",
    )

    print("Pipeline status saved:", OUTPUT_PATH)
    print("DVC stages:", len(status["stages"]))
    print("DVC up to date:", status["overall_up_to_date"])
    print("Spark audited rows:", status["spark_audit"]["rows"]
          if status["spark_audit"] else "Unavailable")


if __name__ == "__main__":
    main()
