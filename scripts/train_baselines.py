"""Train baseline fraud classifiers and track them in MLflow.

Hyperparameters come from ``params.yaml`` so DVC can version them and
invalidate the ``train_baselines`` stage when they change. Each run logs the
Git commit, split checksums, hyperparameters, validation metrics, the model
artifact and permutation feature importance, so every experiment is
reproducible from a Git commit plus an MLflow run ID.
"""

from hashlib import sha256
import json
import logging
from pathlib import Path
import subprocess

import mlflow
import mlflow.sklearn
import pandas as pd
import sklearn
import yaml
from mlflow.tracking import MlflowClient
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = ROOT / "data" / "processed" / "splits"
TRAIN_PATH = SPLIT_DIR / "train.csv"
VALIDATION_PATH = SPLIT_DIR / "validation.csv"
SPLIT_MANIFEST = SPLIT_DIR / "split_manifest.json"
PARAMS_PATH = ROOT / "params.yaml"
METRICS_DIR = ROOT / "metrics"
METRICS_PATH = METRICS_DIR / "train_metrics.json"
IMPORTANCE_PATH = METRICS_DIR / "feature_importance.json"
EXPERIMENT_NAME = "fraud-detection-baselines"

logger = logging.getLogger("train_baselines")

FEATURE_COLUMNS = [f"V{i}" for i in range(1, 29)] + ["Amount"]
EXPECTED_COLUMNS = FEATURE_COLUMNS + ["Class"]

def file_sha256(path):
    """SHA-256 of a file, read in 1 MiB chunks."""
    digest = sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def load_split(path):
    """Load a split and check its columns and missing values."""
    frame = pd.read_csv(path)
    if list(frame.columns) != EXPECTED_COLUMNS:
        raise ValueError(f"Unexpected columns in {path.name}: {list(frame.columns)}")
    if frame.isna().any().any():
        raise ValueError(f"Missing values found in {path.name}")
    return frame[FEATURE_COLUMNS], frame["Class"].astype("int8")


def load_params(path=PARAMS_PATH):
    """Read the ``train`` section of params.yaml."""
    params = yaml.safe_load(path.read_text(encoding="utf-8"))
    return params["train"]


def build_models(params):
    """Build the candidate estimators from versioned hyperparameters."""
    lr = params["logistic_regression"]
    hgb = params["hist_gradient_boosting"]
    return [
        (
            "logistic_regression",
            Pipeline(
                [
                    ("scale", StandardScaler()),
                    (
                        "classifier",
                        LogisticRegression(
                            class_weight=lr["class_weight"],
                            max_iter=lr["max_iter"],
                            random_state=lr["random_state"],
                        ),
                    ),
                ]
            ),
            {
                "scaler": "StandardScaler",
                "class_weight": lr["class_weight"],
                "max_iter": lr["max_iter"],
            },
        ),
        (
            "hist_gradient_boosting",
            HistGradientBoostingClassifier(
                class_weight=hgb["class_weight"],
                learning_rate=hgb["learning_rate"],
                max_iter=hgb["max_iter"],
                max_leaf_nodes=hgb["max_leaf_nodes"],
                l2_regularization=hgb["l2_regularization"],
                random_state=hgb["random_state"],
            ),
            {
                "class_weight": hgb["class_weight"],
                "learning_rate": hgb["learning_rate"],
                "max_iter": hgb["max_iter"],
                "max_leaf_nodes": hgb["max_leaf_nodes"],
                "l2_regularization": hgb["l2_regularization"],
            },
        ),
    ]


def validation_metrics(y_true, fraud_scores, threshold):
    """Threshold-free ranking metrics plus reference metrics at ``threshold``."""
    predicted = (fraud_scores >= threshold).astype("int8")
    return {
        "validation_average_precision": float(
            average_precision_score(y_true, fraud_scores)
        ),
        "validation_roc_auc": float(roc_auc_score(y_true, fraud_scores)),
        "validation_precision_at_0_5": float(
            precision_score(y_true, predicted, zero_division=0)
        ),
        "validation_recall_at_0_5": float(
            recall_score(y_true, predicted, zero_division=0)
        ),
        "validation_f1_at_0_5": float(
            f1_score(y_true, predicted, zero_division=0)
        ),
        "validation_alert_rate_at_0_5": float(predicted.mean()),
    }


def compute_feature_importance(model, x, y, settings):
    """Permutation importance (drop in average precision) per feature.

    A stratified sample keeps the computation fast while preserving the
    rare fraud class. Returns features sorted by descending importance.
    """
    sample_rows = int(settings["sample_rows"])
    if len(x) > sample_rows:
        x, _, y, _ = train_test_split(
            x,
            y,
            train_size=sample_rows,
            stratify=y,
            random_state=settings["random_state"],
        )

    result = permutation_importance(
        model,
        x,
        y,
        scoring="average_precision",
        n_repeats=int(settings["n_repeats"]),
        random_state=settings["random_state"],
        n_jobs=1,
    )
    ranking = [
        {
            "feature": name,
            "importance_mean": float(mean),
            "importance_std": float(std),
        }
        for name, mean, std in zip(
            x.columns, result.importances_mean, result.importances_std
        )
    ]
    return sorted(ranking, key=lambda row: row["importance_mean"], reverse=True)


def git_commit_hash():
    """Return the current Git commit, logged with each MLflow run."""
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main():
    """Train the baseline models and log runs, metrics and feature importance to MLflow."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    params = load_params()
    threshold = float(params["reference_threshold"])

    train_x, train_y = load_split(TRAIN_PATH)
    validation_x, validation_y = load_split(VALIDATION_PATH)

    if set(train_y.unique()) != {0, 1} or set(validation_y.unique()) != {0, 1}:
        raise ValueError("Both splits must contain legitimate and fraud examples.")

    commit_hash = git_commit_hash()

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

    logger.info("Git commit: %s", commit_hash)
    logger.info(
        "Training rows: %s; validation rows: %s",
        f"{len(train_x):,}",
        f"{len(validation_x):,}",
    )
    logger.info("The test split is not loaded or used.")

    summary = {"git_commit": commit_hash, "models": {}}
    importance_summary = {}

    for model_name, model, parameters in build_models(params):
        logger.info("Training %s...", model_name)
        model.fit(train_x, train_y)

        fraud_scores = model.predict_proba(validation_x)[:, 1]
        metrics = validation_metrics(validation_y, fraud_scores, threshold)

        logger.info("Computing permutation feature importance for %s...", model_name)
        importance = compute_feature_importance(
            model, validation_x, validation_y, params["feature_importance"]
        )

        with mlflow.start_run(run_name=model_name) as run:
            mlflow.log_params(
                {
                    "model": model_name,
                    "n_features": len(FEATURE_COLUMNS),
                    "threshold_for_reference_metrics": threshold,
                    "git_commit": commit_hash,
                    "train_split_sha256": file_sha256(TRAIN_PATH),
                    "validation_split_sha256": file_sha256(VALIDATION_PATH),
                    "params_yaml_sha256": file_sha256(PARAMS_PATH),
                    "scikit_learn_version": sklearn.__version__,
                    **parameters,
                }
            )
            mlflow.set_tags(
                {
                    "test_split_used": "false",
                    "evaluation_split": "validation",
                    "data_split_manifest": SPLIT_MANIFEST.name,
                    "top_feature": importance[0]["feature"],
                }
            )
            mlflow.log_metrics(metrics)
            # Track feature importance beyond autolog: one metric per feature
            # plus the full ranking as a JSON artifact.
            mlflow.log_metrics(
                {
                    f"perm_importance_{row['feature']}": row["importance_mean"]
                    for row in importance
                }
            )
            mlflow.log_dict(importance, "feature_importance.json")
            mlflow.log_artifact(str(SPLIT_MANIFEST), artifact_path="data")
            mlflow.log_artifact(str(PARAMS_PATH), artifact_path="config")
            mlflow.log_dict(metrics, "validation_metrics.json")
            mlflow.sklearn.log_model(
                model,
                name="model",
                skops_trusted_types=[
                    "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor"
                ],
            )
            run_id = run.info.run_id
            logger.info("MLflow run ID: %s", run_id)

        for metric_name, value in metrics.items():
            logger.info("  %s: %.6f", metric_name, value)

        summary["models"][model_name] = {"mlflow_run_id": run_id, **metrics}
        importance_summary[model_name] = importance[:10]

    # DVC-tracked outputs: `dvc metrics show` / `dvc metrics diff`.
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    IMPORTANCE_PATH.write_text(
        json.dumps(importance_summary, indent=2) + "\n", encoding="utf-8"
    )
    logger.info("Baseline runs logged to local MLflow.")


if __name__ == "__main__":
    main()
