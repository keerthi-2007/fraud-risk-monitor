"""Integration test using synthetic fixtures and isolated MLflow storage.

No real feedback, production MLflow database, or Docker deployment.
"""

import json
import math

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.tracking import fluent
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from scripts import auto_retrain as retrain
from scripts import prepare_candidate_evaluation as evaluation
from scripts import promotion_controller as promotion
from scripts.approve_feedback import record_hash


def test_feedback_to_verified_candidate(tmp_path, monkeypatch):
    # Never use the actual project directory for test data.
    assert tmp_path.resolve() != retrain.ROOT.resolve()

    splits = tmp_path / "data/processed/splits"
    feedback_dir = tmp_path / "data/feedback"
    artifacts = tmp_path / "mlflow-artifacts"

    splits.mkdir(parents=True)
    feedback_dir.mkdir(parents=True)
    artifacts.mkdir()

    def make_row(index, label, phase):
        features = {
            name: 0.0 for name in retrain.FEATURES
        }
        features["V1"] = (
            (8.0 if label else -8.0)
            + index * 0.0001
            + phase * 0.002
        )
        features["V2"] = (
            (index % 7 - 3) * 0.05
            + phase * 0.001
        )
        features["Amount"] = (
            25.0 + index % 17 + phase * 0.0001
        )
        return features

    def create_split(name, count, phase):
        rows = [
            {
                **make_row(i, i % 2, phase),
                "Class": i % 2,
            }
            for i in range(count)
        ]
        frame = pd.DataFrame(rows)
        frame.to_csv(splits / f"{name}.csv", index=False)
        return frame

    train = create_split("train", 80, 0)
    create_split("validation", 40, 1)
    create_split("test", 20, 2)

    # Synthetic feedback is confined to this temporary directory.
    records = []
    for index in range(20):
        label = 1 if index < 6 else 0
        records.append({
            "feedback_id": f"synthetic-test-{index}",
            "corrected_label": (
                "fraud" if label else "legitimate"
            ),
            "features": make_row(index, label, 3),
            "source_file": "synthetic-test-fixture",
        })

    feedback_path = feedback_dir / "feedback.jsonl"
    feedback_path.write_text(
        "".join(json.dumps(r) + "\n" for r in records),
        encoding="utf-8",
    )

    # Check 1: Unapproved records cannot trigger retraining.
    result = retrain.run_once(
        root=tmp_path,
        min_feedback=20,
        min_fraud=3,
        dry_run=True,
    )
    assert result == "pending_feedback"

    # Create test-only approvals. These are NOT real evidence.
    approvals = [
        {
            "feedback_id": record["feedback_id"],
            "record_sha256": record_hash(record),
            "approved_label": record["corrected_label"],
            "verified_by": "synthetic-test-only",
            "evidence_reference": "synthetic-fixture-only",
        }
        for record in records
    ]

    (feedback_dir / "approved_feedback.jsonl").write_text(
        "".join(json.dumps(a) + "\n" for a in approvals),
        encoding="utf-8",
    )

    # Check 2: Approved fixtures meet retraining requirements.
    result = retrain.run_once(
        root=tmp_path,
        min_feedback=20,
        min_fraud=3,
        dry_run=True,
    )
    assert result == "eligible_dry_run"

    # Set up an isolated MLflow backend AND artifact locations.
    previous_uri = mlflow.get_tracking_uri()
    monkeypatch.setattr(
        fluent, "_active_experiment_id", None
    )

    database = tmp_path / "mlflow.db"
    mlflow.set_tracking_uri(
        f"sqlite:///{database.as_posix()}"
    )

    try:
        experiment_names = [
            "synthetic-test-baseline",
            retrain.EXPERIMENT,
            "fraud-promotion-evaluations",
        ]

        for name in experiment_names:
            location = artifacts / name
            location.mkdir()
            mlflow.create_experiment(
                name,
                artifact_location=location.as_uri(),
            )

        # Train a baseline model on synthetic training data.
        baseline = Pipeline([
            ("scale", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=42,
                ),
            ),
        ])

        baseline.fit(
            train[retrain.FEATURES],
            train["Class"],
        )

        mlflow.set_experiment("synthetic-test-baseline")
        with mlflow.start_run(
            run_name="synthetic_baseline"
        ) as baseline_run:
            mlflow.sklearn.log_model(
                baseline,
                name="model",
            )
            baseline_id = baseline_run.info.run_id

        # Redirect baseline lookup to the temporary MLflow run.
        monkeypatch.setattr(
            retrain, "BASE_RUN_ID", baseline_id
        )

        # Check 3: Actual candidate model retraining.
        result = retrain.run_once(
            root=tmp_path,
            min_feedback=20,
            min_fraud=3,
            dry_run=False,
        )
        assert result == "trained_candidate"

        state_path = (
            feedback_dir / "auto_retrain_state.json"
        )
        state = json.loads(state_path.read_text())
        candidate_id = state["run_id"]

        assert state["eligible_feedback_rows"] == 20
        assert state["quality_gate_passed"] is True

        # Redirect evaluation paths to temporary files.
        evaluation_path = (
            feedback_dir / "promotion_evaluation.json"
        )

        monkeypatch.setattr(evaluation, "ROOT", tmp_path)
        monkeypatch.setattr(evaluation, "STATE", state_path)
        monkeypatch.setattr(
            evaluation, "OUTPUT", evaluation_path
        )
        monkeypatch.setattr(
            evaluation,
            "VALIDATION",
            splits / "validation.csv",
        )

        # Check 4: Actual MLflow candidate evaluation.
        evaluation.main()

        assert evaluation_path.exists()
        report = json.loads(evaluation_path.read_text())
        assert report["validation_passed"] is True
        assert report["candidate_run_id"] == candidate_id

        # Redirect promotion verification to temporary files.
        monkeypatch.setattr(promotion, "ROOT", tmp_path)
        monkeypatch.setattr(
            promotion, "STATE", state_path
        )
        monkeypatch.setattr(
            promotion, "EVALUATION", evaluation_path
        )

        # Check 5: Verify candidate; never call deployment.
        verified = promotion.verify_candidate()

        assert verified is not None
        verified_id, evaluation_id, threshold = verified

        assert verified_id == candidate_id
        assert evaluation_id == report["evaluation_run_id"]
        assert math.isclose(
            threshold,
            state["candidate_threshold"],
            abs_tol=1e-12,
        )

        # Check 6: A repeated feedback batch must not retrain.
        assert retrain.run_once(
            root=tmp_path,
            min_feedback=20,
            min_fraud=3,
            dry_run=False,
        ) == "already_processed"

        print("ISOLATED_PIPELINE_VERIFIED")
        print("No production model deployment performed")

    finally:
        mlflow.set_tracking_uri(previous_uri)
