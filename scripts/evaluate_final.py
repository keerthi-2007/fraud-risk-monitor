"""One-off final evaluation of the approved model on the held-out test split.

Kept outside the DVC DAG so the test split is never used for model selection.
Set MODEL_RUN_ID to evaluate a different MLflow run.
"""

from hashlib import sha256
from io import BytesIO
from pathlib import Path
import os
import subprocess

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
VALIDATION_PATH = ROOT / "data" / "processed" / "splits" / "validation.csv"
TEST_PATH = ROOT / "data" / "processed" / "splits" / "test.csv"
REPORT_PATH = ROOT / "data" / "reports" / "final_test_evaluation.md"

MODEL_RUN_ID = os.getenv("MODEL_RUN_ID", "27da3507bddb413bab9cd0a12aeb092a")
RECALL_TARGET = 0.85

mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")


def load_xy(frame):
    """Split a frame into features and labels, checking both classes are present."""
    if "Class" not in frame.columns:
        raise ValueError("Dataset must contain a 'Class' column.")
    y = frame["Class"].to_numpy(dtype=int)
    X = frame.drop(columns=["Class", "feature_group_id"], errors="ignore")
    if set(np.unique(y)) != {0, 1}:
        raise ValueError(f"Expected labels 0 and 1; found {sorted(np.unique(y))}")
    return X, y


def evaluate(y, scores, threshold):
    """Return confusion-matrix counts and metrics at a threshold."""
    predicted = scores >= threshold
    tn, fp, fn, tp = confusion_matrix(
        y, predicted, labels=[0, 1]
    ).ravel()
    alerts = tp + fp
    precision = tp / alerts if alerts else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    return {
        "average_precision": float(average_precision_score(y, scores)),
        "roc_auc": float(roc_auc_score(y, scores)),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "alert_rate": alerts / len(y),
        "alerts": alerts,
        "fraud_caught": tp,
        "false_positives": fp,
        "fraud_missed": fn,
        "true_negatives": tn,
    }


if not VALIDATION_PATH.exists() or not TEST_PATH.exists():
    raise FileNotFoundError("A validation or test split is missing.")

client = MlflowClient()
source_run = client.get_run(MODEL_RUN_ID)
model = mlflow.sklearn.load_model(f"runs:/{MODEL_RUN_ID}/model")

validation = pd.read_csv(VALIDATION_PATH)
X_validation, y_validation = load_xy(validation)
validation_scores = model.predict_proba(X_validation)[:, 1]

precision, recall, thresholds = precision_recall_curve(
    y_validation, validation_scores
)
eligible = np.flatnonzero(recall[:-1] >= RECALL_TARGET)
if len(eligible) == 0:
    raise RuntimeError("The selected validation recall target is unreachable.")

threshold = float(thresholds[eligible[-1]])
validation_metrics = evaluate(y_validation, validation_scores, threshold)

print(
    f"Validation selection: threshold={threshold:.17g}, "
    f"recall={validation_metrics['recall']:.3%}, "
    f"alerts={validation_metrics['alerts']:,}, "
    f"fraud_caught={validation_metrics['fraud_caught']}"
)

if validation_metrics["recall"] + 1e-12 < RECALL_TARGET:
    raise RuntimeError(
        "Selected threshold did not meet the validation recall target. "
        "The test split was not read."
    )

# Read and score the test split only after the validation selection check passes.
test_bytes = TEST_PATH.read_bytes()
test_sha256 = sha256(test_bytes).hexdigest()
test = pd.read_csv(BytesIO(test_bytes))
X_test, y_test = load_xy(test)

if list(X_test.columns) != list(X_validation.columns):
    raise ValueError("Validation and test feature columns do not match.")

test_scores = model.predict_proba(X_test)[:, 1]
test_metrics = evaluate(y_test, test_scores, threshold)

git_commit = subprocess.run(
    ["git", "rev-parse", "HEAD"],
    cwd=ROOT,
    capture_output=True,
    text=True,
    check=True,
).stdout.strip()

with mlflow.start_run(
    experiment_id=source_run.info.experiment_id,
    run_name="final_test_logistic_recall85",
) as run:
    evaluation_run_id = run.info.run_id
    mlflow.set_tags(
        {
            "evaluation_stage": "final_test",
            "source_model_run_id": MODEL_RUN_ID,
            "git_commit": git_commit,
        }
    )
    mlflow.log_params(
        {
            "selection_rule": "highest threshold with validation recall >= 0.85",
            "recall_target": RECALL_TARGET,
            "decision_threshold": format(threshold, ".17g"),
            "source_model_run_id": MODEL_RUN_ID,
            "test_csv_sha256": test_sha256,
        }
    )
    mlflow.log_metrics(
        {
            f"validation_{name}": float(value)
            for name, value in validation_metrics.items()
        }
    )
    mlflow.log_metrics(
        {
            f"test_{name}": float(value)
            for name, value in test_metrics.items()
        }
    )

    report = f"""# Final test evaluation

- Model: Logistic Regression
- Source MLflow model run: `{MODEL_RUN_ID}`
- Evaluation MLflow run: `{evaluation_run_id}`
- Git commit: `{git_commit}`
- Threshold rule: highest threshold with validation recall at least 85%
- Decision threshold: `{format(threshold, ".17g")}`
- Validation rows: {len(y_validation):,}
- Validation fraud caught: {validation_metrics["fraud_caught"]} of {int(y_validation.sum())}
- Validation alerts: {validation_metrics["alerts"]:,}
- Test rows: {len(y_test):,}
- Test fraud cases: {int(y_test.sum())}
- Test CSV SHA-256: `{test_sha256}`

## Test metrics

- Average Precision: {test_metrics["average_precision"]:.6f}
- ROC-AUC: {test_metrics["roc_auc"]:.6f}
- Precision at selected threshold: {test_metrics["precision"]:.4%}
- Recall at selected threshold: {test_metrics["recall"]:.4%}
- F1 at selected threshold: {test_metrics["f1"]:.6f}
- Alert rate: {test_metrics["alert_rate"]:.4%}
- Fraud caught: {test_metrics["fraud_caught"]}
- False positives: {test_metrics["false_positives"]}
- Fraud missed: {test_metrics["fraud_missed"]}
- True negatives: {test_metrics["true_negatives"]}
"""

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    mlflow.log_artifact(str(REPORT_PATH), artifact_path="reports")

print(
    f"Validation selection: threshold={threshold:.17g}, "
    f"recall={validation_metrics['recall']:.3%}, "
    f"alerts={validation_metrics['alerts']:,}, "
    f"fraud_caught={validation_metrics['fraud_caught']}"
)

if validation_metrics["recall"] + 1e-12 < RECALL_TARGET:
    raise RuntimeError(
        "Selected threshold did not meet the validation recall target. "
        "The test split was not read."
    )
