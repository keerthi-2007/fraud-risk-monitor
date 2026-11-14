from hashlib import sha256
from pathlib import Path
import subprocess

import mlflow
import mlflow.sklearn
import pandas as pd
import sklearn
from mlflow.tracking import MlflowClient
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = ROOT / "data" / "processed" / "splits"
TRAIN_PATH = SPLIT_DIR / "train.csv"
VALIDATION_PATH = SPLIT_DIR / "validation.csv"
SPLIT_MANIFEST = SPLIT_DIR / "split_manifest.json"
EXPERIMENT_NAME = "fraud-detection-baselines"
THRESHOLD = 0.5

FEATURE_COLUMNS = [f"V{i}" for i in range(1, 29)] + ["Amount"]
EXPECTED_COLUMNS = FEATURE_COLUMNS + ["Class"]

def file_sha256(path):
    digest = sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def load_split(path):
    frame = pd.read_csv(path)
    if list(frame.columns) != EXPECTED_COLUMNS:
        raise ValueError(f"Unexpected columns in {path.name}: {list(frame.columns)}")
    if frame.isna().any().any():
        raise ValueError(f"Missing values found in {path.name}")
    return frame[FEATURE_COLUMNS], frame["Class"].astype("int8")

train_x, train_y = load_split(TRAIN_PATH)
validation_x, validation_y = load_split(VALIDATION_PATH)

if set(train_y.unique()) != {0, 1} or set(validation_y.unique()) != {0, 1}:
    raise ValueError("Both splits must contain legitimate and fraud examples.")

commit_hash = subprocess.run(
    ["git", "rev-parse", "HEAD"],
    cwd=ROOT,
    check=True,
    capture_output=True,
    text=True,
).stdout.strip()

tracking_uri = f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"
artifact_root = (ROOT / "mlartifacts").as_uri()

mlflow.set_tracking_uri(tracking_uri)
client = MlflowClient(tracking_uri=tracking_uri)

if client.get_experiment_by_name(EXPERIMENT_NAME) is None:
    client.create_experiment(
        EXPERIMENT_NAME,
        artifact_location=artifact_root,
    )
mlflow.set_experiment(EXPERIMENT_NAME)

models = [
    (
        "logistic_regression",
        Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=2000,
                        random_state=42,
                    ),
                ),
            ]
        ),
        {
            "scaler": "StandardScaler",
            "class_weight": "balanced",
            "max_iter": 2000,
        },
    ),
    (
        "hist_gradient_boosting",
        HistGradientBoostingClassifier(
            class_weight="balanced",
            learning_rate=0.1,
            max_iter=150,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=42,
        ),
        {
            "class_weight": "balanced",
            "learning_rate": 0.1,
            "max_iter": 150,
            "max_leaf_nodes": 31,
            "l2_regularization": 1.0,
        },
    ),
]

print(f"Git commit: {commit_hash}")
print(f"Training rows: {len(train_x):,}; validation rows: {len(validation_x):,}")
print("The test split is not loaded or used.")

for model_name, model, parameters in models:
    print(f"Training {model_name}...")
    model.fit(train_x, train_y)

    fraud_scores = model.predict_proba(validation_x)[:, 1]
    predictions_at_threshold = (fraud_scores >= THRESHOLD).astype("int8")

    metrics = {
        "validation_average_precision": float(
            average_precision_score(validation_y, fraud_scores)
        ),
        "validation_roc_auc": float(roc_auc_score(validation_y, fraud_scores)),
        "validation_precision_at_0_5": float(
            precision_score(validation_y, predictions_at_threshold, zero_division=0)
        ),
        "validation_recall_at_0_5": float(
            recall_score(validation_y, predictions_at_threshold, zero_division=0)
        ),
        "validation_f1_at_0_5": float(
            f1_score(validation_y, predictions_at_threshold, zero_division=0)
        ),
        "validation_alert_rate_at_0_5": float(predictions_at_threshold.mean()),
    }

    with mlflow.start_run(run_name=model_name) as run:
        mlflow.log_params(
            {
                "model": model_name,
                "n_features": len(FEATURE_COLUMNS),
                "threshold_for_reference_metrics": THRESHOLD,
                "git_commit": commit_hash,
                "train_split_sha256": file_sha256(TRAIN_PATH),
                "validation_split_sha256": file_sha256(VALIDATION_PATH),
                "scikit_learn_version": sklearn.__version__,
                **parameters,
            }
        )
        mlflow.set_tags(
            {
                "test_split_used": "false",
                "evaluation_split": "validation",
                "data_split_manifest": SPLIT_MANIFEST.name,
            }
        )
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(SPLIT_MANIFEST), artifact_path="data")
        mlflow.log_dict(metrics, "validation_metrics.json")
        mlflow.sklearn.log_model(model, name="model", skops_trusted_types=["sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor"],)

        print(f"MLflow run ID: {run.info.run_id}")

    for metric_name, value in metrics.items():
        print(f"  {metric_name}: {value:.6f}")

print("Baseline runs logged to local MLflow.")
