"""Compare candidate decision thresholds on validation scores (recall vs alert volume)."""

from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

ROOT = Path(__file__).resolve().parents[1]
VALIDATION_PATH = ROOT / "data" / "processed" / "splits" / "validation.csv"

RUNS = {
    "Logistic Regression": "27da3507bddb413bab9cd0a12aeb092a",
    "HistGradientBoosting": "89200109020045859f1247d2aca2f3fd",
}
RECALL_TARGETS = (0.80, 0.85, 0.90, 0.95, 0.99)

mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")

if not VALIDATION_PATH.exists():
    raise FileNotFoundError(f"Validation split not found: {VALIDATION_PATH}")

validation = pd.read_csv(VALIDATION_PATH)
if "Class" not in validation.columns:
    raise ValueError("Validation data must contain a 'Class' column.")

y = validation["Class"].to_numpy(dtype=int)
X = validation.drop(columns=["Class", "feature_group_id"], errors="ignore")

if set(np.unique(y)) != {0, 1}:
    raise ValueError(f"Expected labels 0 and 1; found {sorted(np.unique(y))}")

print(f"Validation rows: {len(y):,}")
print(f"Fraud rows: {int(y.sum()):,}")
print("The test split is not loaded or used.")

def measure_threshold(y_true, scores, threshold):
    """Return recall, precision and alert counts for one threshold."""
    predicted = scores >= threshold
    true_positive = int(np.sum(predicted & (y_true == 1)))
    false_positive = int(np.sum(predicted & (y_true == 0)))
    false_negative = int(np.sum(~predicted & (y_true == 1)))
    alerts = true_positive + false_positive

    precision = true_positive / alerts if alerts else 0.0
    recall = true_positive / (true_positive + false_negative)

    return {
        "threshold": float(threshold),
        "precision": precision,
        "recall": recall,
        "alerts": alerts,
        "alert_rate": alerts / len(y_true),
        "fraud_caught": true_positive,
        "false_positives": false_positive,
        "fraud_missed": false_negative,
    }

for model_name, run_id in RUNS.items():
    model_uri = f"runs:/{run_id}/model"
    model = mlflow.sklearn.load_model(model_uri)
    scores = model.predict_proba(X)[:, 1]

    print(f"\n{model_name} (run {run_id})")
    print(
        "criterion       threshold  precision  recall  alerts  "
        "alert_rate  fraud_caught  false_positives  fraud_missed"
    )

    reference = measure_threshold(y, scores, 0.5)
    rows = [("fixed 0.50", reference)]

    precision, recall, thresholds = precision_recall_curve(y, scores)
    candidates = [
        measure_threshold(y, scores, threshold)
        for threshold, _, _ in zip(thresholds, precision[1:], recall[1:])
    ]

    for target in RECALL_TARGETS:
        eligible = [
            row for row in candidates
            if row["recall"] + 1e-12 >= target
        ]
        if eligible:
            best = max(eligible, key=lambda row: row["threshold"])
            rows.append((f"recall >= {target:.0%}", best))
        else:
            print(f"Recall target {target:.0%} could not be reached.")

    for criterion, row in rows:
        print(
            f"{criterion:<15} "
            f"{row['threshold']:>16.12f}  "
            f"{row['precision']:>9.3%}  "
            f"{row['recall']:>6.1%}  "
            f"{row['alerts']:>6,d}  "
            f"{row['alert_rate']:>9.3%}  "
            f"{row['fraud_caught']:>12,d}  "
            f"{row['false_positives']:>15,d}  "
            f"{row['fraud_missed']:>12,d}"
        )
