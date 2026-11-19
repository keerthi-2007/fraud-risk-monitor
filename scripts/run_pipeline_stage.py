"""Execute an allowlisted DVC stage and record its actual result."""

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
HISTORY_PATH = ROOT / "monitoring" / "pipeline_executions.jsonl"

ALLOWED_STAGES = {
    "prepare_data",
    "create_splits",
    "spark_audit",
    "spark_transform",
    "build_drift_baseline",
}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def run_stage(stage, *, force=False, history_path=HISTORY_PATH):
    if stage not in ALLOWED_STAGES:
        raise ValueError(f"Unsupported DVC stage: {stage}")

    command = ["dvc", "repro", "--single-item"]

    if force:
        command.extend(["--force", "--no-run-cache"])

    command.append(stage)

    started_at = utc_now()
    started = perf_counter()

    try:
        result = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        returncode = result.returncode
        stdout = result.stdout
        stderr = result.stderr
    except OSError as exc:
        returncode = -1
        stdout = ""
        stderr = str(exc)

    duration_seconds = perf_counter() - started

    # A successful no-op DVC command is not proof of stage execution.
    execution_confirmed = (
        returncode == 0
        and f"Running stage '{stage}'" in stdout
    )

    if returncode != 0:
        status = "failed"
    elif execution_confirmed:
        status = "succeeded"
    else:
        status = "skipped"

    record = {
        "execution_id": str(uuid4()),
        "stage": stage,
        "started_at": started_at,
        "finished_at": utc_now(),
        "duration_seconds": round(duration_seconds, 6),
        "status": status,
        "returncode": returncode,
        "execution_confirmed": execution_confirmed,
        "command": command,
    }

    # Measure end-to-end throughput only for a confirmed,
    # successful Spark audit execution.
    record["rows_processed"] = None
    record["throughput_rows_per_second"] = None

    if stage == "spark_audit" and status == "succeeded":
        audit_path = ROOT / "data/processed/spark_audit.json"

        try:
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            rows = audit.get("rows")

            if (
                isinstance(rows, int)
                and not isinstance(rows, bool)
                and rows > 0
                and duration_seconds > 0
            ):
                record["rows_processed"] = rows
                record["throughput_rows_per_second"] = round(
                    rows / duration_seconds, 2
                )
        except (OSError, ValueError):
            # Missing audit metadata must not be reported
            # as an invented throughput value.
            pass

    history_path = Path(history_path)
    history_path.parent.mkdir(parents=True, exist_ok=True)

    with history_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")

    print("Stage:", stage)
    print("Status:", status)
    print("Duration:", round(duration_seconds, 3), "seconds")
    print("Exit code:", returncode)
    print("History:", history_path)

    if returncode != 0:
        print(stderr[-2000:], file=sys.stderr)

    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=sorted(ALLOWED_STAGES))
    parser.add_argument(
        "--force",
        action="store_true",
        help="Execute the stage even if DVC considers it up to date.",
    )
    args = parser.parse_args()

    record = run_stage(args.stage, force=args.force)

    if record["status"] == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
