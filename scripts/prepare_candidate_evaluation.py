"""Validate a retrained candidate and create an MLflow evaluation record.

Uses the validation split only. Never accesses final test labels.
Does not deploy or restart the backend.
"""

import json
import math
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient

from scripts.auto_retrain import FEATURES, choose_threshold

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "data/feedback/auto_retrain_state.json"
OUTPUT = ROOT / "data/feedback/promotion_evaluation.json"
VALIDATION = ROOT / "data/processed/splits/validation.csv"


def main():
    """Log an MLflow evaluation run for a candidate that passed the quality gate."""
    if not STATE.exists():
        print("NO_CANDIDATE")
        return

    state = json.loads(STATE.read_text())

    if state.get("quality_gate_passed") is not True:
        print("BLOCKED: Quality gate not passed")
        return

    run_id = state["run_id"]
    signature = state["feedback_signature"]
    threshold = float(state["candidate_threshold"])

    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Invalid candidate threshold")

    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    client = MlflowClient()
    source = client.get_run(run_id)

    if source.info.status != "FINISHED":
        raise ValueError("Candidate training run incomplete")

    required_tags = {
        "quality_gate_passed": "true",
        "deployment": "candidate_only",
        "test_split_labels_used": "false",
        "feedback_signature": signature,
    }
    for key, expected in required_tags.items():
        if source.data.tags.get(key) != expected:
            raise ValueError(f"Candidate verification failed: {key}")

    if not math.isclose(
        float(source.data.params["candidate_threshold"]),
        threshold, abs_tol=1e-12
    ):
        raise ValueError("Candidate threshold mismatch")

    baseline_id = source.data.tags["source_model_run_id"]

    if OUTPUT.exists():
        previous = json.loads(OUTPUT.read_text())
        if previous.get("candidate_run_id") == run_id:
            evaluation = client.get_run(previous["evaluation_run_id"])
            if evaluation.info.status == "FINISHED":
                print("ALREADY_EVALUATED:", previous["evaluation_run_id"])
                return

    validation = pd.read_csv(VALIDATION)
    X = validation[FEATURES]
    y = validation["Class"].to_numpy(dtype=int)

    candidate = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
    baseline = mlflow.sklearn.load_model(f"runs:/{baseline_id}/model")

    selected_threshold, candidate_metrics = choose_threshold(
        y, candidate.predict_proba(X)[:, 1]
    )
    _, baseline_metrics = choose_threshold(
        y, baseline.predict_proba(X)[:, 1]
    )

    if not math.isclose(selected_threshold, threshold, abs_tol=1e-12):
        raise ValueError("Recomputed threshold mismatch")

    passed = (
        candidate_metrics["recall"] >= 0.85
        and candidate_metrics["average_precision"]
            >= baseline_metrics["average_precision"] - 0.01
        and candidate_metrics["alerts"]
            <= int(baseline_metrics["alerts"] * 1.15)
    )

    if not passed:
        print("BLOCKED: Independent validation gate failed")
        return

    mlflow.set_experiment("fraud-promotion-evaluations")

    with mlflow.start_run(run_name="candidate_validation_approval") as run:
        mlflow.set_tags({
            "evaluation_stage": "validation_only",
            "approval_gate_passed": "true",
            "feedback_signature": signature,
        })
        mlflow.log_params({
            "source_model_run_id": run_id,
            "baseline_model_run_id": baseline_id,
            "decision_threshold": format(threshold, ".17g"),
        })
        mlflow.log_metrics({
            f"validation_{key}": float(value)
            for key, value in candidate_metrics.items()
        })
        evaluation_run_id = run.info.run_id

    result = {
        "candidate_run_id": run_id,
        "evaluation_run_id": evaluation_run_id,
        "decision_threshold": threshold,
        "feedback_signature": signature,
        "validation_passed": True,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n")

    print("CANDIDATE_EVALUATION_APPROVED")
    print("Candidate:", run_id)
    print("Evaluation:", evaluation_run_id)


if __name__ == "__main__":
    main()
