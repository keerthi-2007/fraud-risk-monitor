"""Verify a retrained candidate before any deployment is attempted.

This version is read-only. Deployment is deliberately not enabled yet.
"""

import json
import math
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "data/feedback/auto_retrain_state.json"
EVALUATION = ROOT / "data/feedback/promotion_evaluation.json"


def verify_candidate():
    if not STATE.exists():
        print("NO_CANDIDATE: Retraining has not produced a candidate.")
        return None

    if not EVALUATION.exists():
        print("NOT_READY: Candidate evaluation record is missing.")
        return None

    state = json.loads(STATE.read_text(encoding="utf-8"))
    approval = json.loads(EVALUATION.read_text(encoding="utf-8"))

    candidate_id = state.get("run_id")
    evaluation_id = approval.get("evaluation_run_id")
    threshold = state.get("candidate_threshold")
    signature = state.get("feedback_signature")

    if (
        not candidate_id
        or not evaluation_id
        or not signature
        or state.get("quality_gate_passed") is not True
        or approval.get("validation_passed") is not True
    ):
        raise ValueError("Candidate has not passed the approval gates.")

    if (
        approval.get("candidate_run_id") != candidate_id
        or approval.get("feedback_signature") != signature
    ):
        raise ValueError("Candidate and evaluation records do not match.")

    if (
        not isinstance(threshold, (int, float))
        or not math.isfinite(threshold)
        or not 0 <= threshold <= 1
    ):
        raise ValueError("Invalid candidate threshold.")

    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    client = MlflowClient()

    candidate = client.get_run(candidate_id)
    evaluation = client.get_run(evaluation_id)

    if candidate.info.status != "FINISHED":
        raise ValueError("Candidate training is incomplete.")

    if evaluation.info.status != "FINISHED":
        raise ValueError("Candidate evaluation is incomplete.")

    if candidate.data.tags.get("quality_gate_passed") != "true":
        raise ValueError("MLflow candidate quality gate failed.")

    if candidate.data.tags.get("feedback_signature") != signature:
        raise ValueError("MLflow candidate feedback signature mismatch.")

    if candidate.data.tags.get("test_split_labels_used") != "false":
        raise ValueError("Candidate test-set isolation check failed.")

    if evaluation.data.tags.get("approval_gate_passed") != "true":
        raise ValueError("MLflow evaluation approval is missing.")

    if evaluation.data.tags.get("evaluation_stage") != "validation_only":
        raise ValueError("Evaluation must use the validation split only.")

    if evaluation.data.tags.get("feedback_signature") != signature:
        raise ValueError("MLflow evaluation feedback signature mismatch.")

    if evaluation.data.params.get("source_model_run_id") != candidate_id:
        raise ValueError("Evaluation references a different model.")

    approved_threshold = float(
        evaluation.data.params["decision_threshold"]
    )

    if not math.isclose(
        approved_threshold, threshold, rel_tol=0, abs_tol=1e-12
    ):
        raise ValueError("Evaluation decision threshold mismatch.")

    print("CANDIDATE_VERIFIED")
    print("Model URI:", f"runs:/{candidate_id}/model")
    print("Evaluation run:", evaluation_id)
    print("Decision threshold:", threshold)
    print("Deployment performed: NO")

    return candidate_id, evaluation_id, threshold


if __name__ == "__main__":
    verify_candidate()
