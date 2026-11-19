"""Read-only checks before considering a retrained model for promotion."""

import json
import math
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "data/feedback/auto_retrain_state.json"


def main():
    """Print whether the latest retraining candidate passed its quality gate."""
    if not STATE.is_file():
        print("NO_CANDIDATE: No completed retraining run yet.")
        return

    state = json.loads(STATE.read_text(encoding="utf-8"))

    if state.get("quality_gate_passed") is not True:
        print("BLOCKED: Candidate did not pass the quality gate.")
        return

    run_id = state.get("run_id")
    signature = state.get("feedback_signature")
    threshold = state.get("candidate_threshold")

    if not run_id or not signature:
        raise ValueError("Candidate state is incomplete.")

    if (
        not isinstance(threshold, (int, float))
        or not math.isfinite(threshold)
        or not 0 <= threshold <= 1
    ):
        raise ValueError("Invalid candidate decision threshold.")

    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    run = MlflowClient().get_run(run_id)

    if run.info.status != "FINISHED":
        raise ValueError("Candidate MLflow run is not finished.")

    required_tags = {
        "quality_gate_passed": "true",
        "deployment": "candidate_only",
        "test_split_labels_used": "false",
        "feedback_signature": signature,
    }

    for key, expected in required_tags.items():
        if run.data.tags.get(key) != expected:
            raise ValueError(f"Candidate verification failed: {key}")

    print("CANDIDATE_VALIDATED")
    print("MLflow run:", run_id)
    print("Decision threshold:", threshold)
    print("Quality gate: PASSED")
    print("Deployment: NOT PERFORMED")
    print("Independent label verification still required.")


if __name__ == "__main__":
    main()
