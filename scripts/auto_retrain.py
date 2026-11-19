"""Feedback-triggered, leakage-aware MLflow candidate retraining.

Run from the project virtual environment:
  python scripts/auto_retrain.py --once
  python scripts/auto_retrain.py --watch --poll-seconds 60

This worker NEVER replaces the serving model automatically. A retraining attempt
produces a versioned MLflow candidate and gate report; promotion is separate.
Only genuinely new, externally confirmed labelled transactions are eligible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "data" / "processed" / "splits"
FEEDBACK = ROOT / "data" / "feedback" / "feedback.jsonl"
STATE = ROOT / "data" / "feedback" / "auto_retrain_state.json"
FEATURES = [f"V{i}" for i in range(1, 29)] + ["Amount"]
LABELS = {"legitimate": 0, "fraud": 1}
BASE_RUN_ID = os.getenv("MODEL_RUN_ID", "27da3507bddb413bab9cd0a12aeb092a")
EXPERIMENT = "fraud-feedback-retraining"
RECALL_TARGET = 0.85
LOG = logging.getLogger("auto_retrain")


def fingerprint(features: dict[str, Any]) -> str:
    """Stable feature-only fingerprint, independent of feedback metadata."""
    if set(features) != set(FEATURES):
        raise ValueError("Feedback has unexpected feature columns")
    vals = [float(features[col]) for col in FEATURES]
    if not all(math.isfinite(v) for v in vals) or vals[-1] < 0:
        raise ValueError("Feedback has invalid feature values")
    return hashlib.sha256(np.asarray(vals, dtype="<f8").tobytes()).hexdigest()


def split_frame(path: Path, with_labels: bool = True) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Required DVC split missing: {path}; run dvc pull/repro")
    frame = pd.read_csv(path)
    expected = FEATURES + (["Class"] if with_labels else [])
    if not set(expected).issubset(frame.columns):
        raise ValueError(f"Missing columns in {path}: {sorted(set(expected) - set(frame.columns))}")
    if not np.isfinite(frame[FEATURES].to_numpy(dtype=float)).all():
        raise ValueError(f"Invalid feature values in {path}")
    if with_labels and not frame["Class"].isin([0, 1]).all():
        raise ValueError(f"Invalid labels in {path}")
    return frame


def known_fingerprints(paths: list[Path]) -> set[str]:
    """Exclude duplicate rows from original train, validation and final test.

    Test split labels are NEVER loaded or used for selection; only features are
    read to prevent a user-uploaded test transaction leaking into retraining.
    """
    known: set[str] = set()
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Cannot enforce holdout isolation: {path} missing")
        frame = pd.read_csv(path, usecols=FEATURES)
        for row in frame[FEATURES].itertuples(index=False, name=None):
            known.add(hashlib.sha256(np.asarray(row, dtype="<f8").tobytes()).hexdigest())
    return known


def load_approvals(path: Path) -> dict:
    """Load approved labels; reject malformed or duplicate approvals."""
    approvals = {}
    if not path.exists():
        return approvals

    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        if not isinstance(item, dict):
            raise ValueError("Invalid approval record")

        required = (
            "feedback_id", "record_sha256", "approved_label",
            "verified_by", "evidence_reference"
        )
        if any(not isinstance(item.get(k), str) or not item[k].strip()
               for k in required):
            raise ValueError("Incomplete approval record")

        if item["approved_label"] not in LABELS:
            raise ValueError("Invalid approved label")

        fid = item["feedback_id"]
        if fid in approvals:
            raise ValueError("Duplicate feedback approval")

        approvals[fid] = item

    return approvals


def read_feedback(path: Path, known: set[str],
                  approved: dict | None = None) -> tuple[pd.DataFrame, dict[str, int]]:
    counts = {"records": 0, "invalid": 0, "duplicates": 0, "known_dataset": 0,
              "conflicting_labels": 0, "eligible": 0}
    counts["not_approved"] = 0
    if not path.exists():
        return pd.DataFrame(columns=FEATURES + ["Class"]), counts

    unique: dict[str, tuple[int, dict[str, float]]] = {}
    conflicts: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        counts["records"] += 1
        try:
            record = json.loads(line)

            if approved is not None:
                approval = approved.get(record.get("feedback_id"))
                if approval is None:
                    counts["not_approved"] += 1
                    continue

                digest = hashlib.sha256(
                    json.dumps(
                        record, sort_keys=True, separators=(",", ":"),
                        allow_nan=False
                    ).encode("utf-8")
                ).hexdigest()

                if (approval["record_sha256"] != digest or
                        approval["approved_label"] != record["corrected_label"]):
                    counts["invalid"] += 1
                    continue

            label = LABELS[record["corrected_label"]]
            features = record["features"]
            fp = fingerprint(features)
        except (KeyError, ValueError, TypeError, json.JSONDecodeError, OverflowError):
            counts["invalid"] += 1
            continue
        if fp in known:
            counts["known_dataset"] += 1
            continue
        if fp in unique:
            if unique[fp][0] != label:
                conflicts.add(fp)
            else:
                counts["duplicates"] += 1
            continue
        unique[fp] = (label, {col: float(features[col]) for col in FEATURES})
    for fp in conflicts:
        unique.pop(fp, None)
    counts["conflicting_labels"] = len(conflicts)
    rows = [{**values, "Class": label} for fp, (label, values) in sorted(unique.items())]
    counts["eligible"] = len(rows)
    return pd.DataFrame(rows, columns=FEATURES + ["Class"]), counts


def choose_threshold(y: np.ndarray, scores: np.ndarray) -> tuple[float, dict[str, float]]:
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Validation must contain both classes")
    precision, recall, thresholds = precision_recall_curve(y, scores)
    eligible = np.flatnonzero(recall[:-1] >= RECALL_TARGET)
    if not len(eligible):
        raise ValueError("Recall target not reached")
    threshold = float(thresholds[eligible[-1]])
    predicted = scores >= threshold
    caught = int(np.sum(predicted & (y == 1)))
    total_fraud = int(np.sum(y == 1))
    metrics = {
        "average_precision": float(average_precision_score(y, scores)),
        "roc_auc": float(roc_auc_score(y, scores)),
        "recall": caught / total_fraud,
        "alerts": int(predicted.sum()),
        "fraud_caught": caught,
        "alert_rate": float(predicted.mean()),
    }
    return threshold, metrics


def latest_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def write_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def run_once(*, min_feedback: int, min_fraud: int, dry_run: bool = False,
             root: Path = ROOT) -> str:
    splits = root / "data" / "processed" / "splits"
    feedback_path = root / "data" / "feedback" / "feedback.jsonl"
    state_path = root / "data" / "feedback" / "auto_retrain_state.json"
    train_path, val_path, test_path = (splits / f"{n}.csv" for n in ("train", "validation", "test"))

    known = known_fingerprints([train_path, val_path, test_path])
    approvals = load_approvals(
        root / "data" / "feedback" / "approved_feedback.jsonl"
    )
    new, stats = read_feedback(feedback_path, known, approved=approvals)
    LOG.info("Feedback eligibility: %s", stats)
    counts = new["Class"].value_counts() if not new.empty else pd.Series(dtype=int)
    fraud = int(counts.get(1, 0))
    normal = int(counts.get(0, 0))

    # Publish monitoring data through the existing shared feedback volume.
    # Dry-run checks must not write status files.
    if not dry_run:
        write_state(
            root / "data" / "feedback" / "retraining_monitor.json",
            {
                "schema_version": 1,
                "checked_at": datetime.now(timezone.utc).isoformat(),
                "feedback_received": stats["records"],
                "not_approved": stats["not_approved"],
                "invalid": stats["invalid"],
                "eligible": len(new),
                "eligible_fraud": fraud,
                "eligible_legitimate": normal,
                "required_feedback": min_feedback,
                "required_per_class": min_fraud,
                "threshold_met": (
                    len(new) >= min_feedback
                    and fraud >= min_fraud
                    and normal >= min_fraud
                ),
            },
        )

    if len(new) < min_feedback or fraud < min_fraud or normal < min_fraud:
        LOG.info("No retraining: need >=%s new eligible labels and >=%s in each class; got %s, fraud=%s, legitimate=%s",
                 min_feedback, min_fraud, len(new), fraud, normal)
        return "pending_feedback"

    signature = hashlib.sha256(new.to_csv(index=False).encode("utf-8")).hexdigest()
    previous = latest_state(state_path)
    if previous.get("feedback_signature") == signature:
        LOG.info("This exact feedback set was already trained in run %s", previous.get("run_id"))
        return "already_processed"
    if dry_run:
        LOG.info("Dry run: %s eligible feedback records can trigger candidate retraining", len(new))
        return "eligible_dry_run"

    # Import MLflow only after eligibility, so status checks are lightweight.
    import mlflow
    import mlflow.sklearn

    train = split_frame(train_path)
    val = split_frame(val_path)
    X = pd.concat([train[FEATURES], new[FEATURES]], ignore_index=True)
    y = pd.concat([train["Class"], new["Class"]], ignore_index=True).astype(int)
    Xv = val[FEATURES]
    yv = val["Class"].to_numpy(dtype=int)

    candidate = Pipeline([
        ("scale", StandardScaler()),
        ("classifier", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42)),
    ])
    LOG.info(
        "Training candidate with %s original training rows and %s eligible new labelled rows",
        len(train),
        len(new),
    )
    candidate.fit(X, y)
    scores = candidate.predict_proba(Xv)[:, 1]
    threshold, candidate_metrics = choose_threshold(yv, scores)

    tracking_uri = f"sqlite:///{(root / 'mlflow.db').as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)
    reference_model = mlflow.sklearn.load_model(f"runs:/{BASE_RUN_ID}/model")
    base_scores = reference_model.predict_proba(Xv)[:, 1]
    _, base_metrics = choose_threshold(yv, base_scores)
    gate = (candidate_metrics["average_precision"] >= base_metrics["average_precision"] - 0.01
            and candidate_metrics["recall"] >= RECALL_TARGET
            and candidate_metrics["alerts"] <= max(1, int(base_metrics["alerts"] * 1.15)))

    mlflow.set_experiment(EXPERIMENT)
    git_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                                capture_output=True, text=True, check=False).stdout.strip() or "unknown"
    with mlflow.start_run(run_name="feedback_triggered_logistic_regression") as run:
        mlflow.set_tags({
            "trigger": "new_feedback_count_threshold", "deployment": "candidate_only",
            "source_model_run_id": BASE_RUN_ID, "git_commit": git_commit,
            "quality_gate_passed": str(gate).lower(), "test_split_labels_used": "false",
            "feedback_signature": signature,
        })
        mlflow.log_params({
            "min_feedback": min_feedback, "min_fraud_per_class": min_fraud,
            "eligible_feedback_rows": len(new), "fraud_feedback": fraud,
            "legitimate_feedback": normal, "original_train_rows": len(train),
            "train_sha256": sha256_file(train_path), "validation_sha256": sha256_file(val_path),
            "recall_target": RECALL_TARGET, "candidate_threshold": threshold,
            "feedback_signature": signature,
        })
        mlflow.log_metrics({f"candidate_{k}": v for k, v in candidate_metrics.items()})
        mlflow.log_metrics({f"baseline_{k}": v for k, v in base_metrics.items()})
        mlflow.sklearn.log_model(candidate, name="model")
        mlflow.log_dict({"feedback_counts": stats, "candidate": candidate_metrics,
                         "baseline": base_metrics, "quality_gate_passed": gate,
                         "candidate_threshold": threshold,
                         "note": "Candidate logged; current FastAPI model is not automatically replaced."},
                        "retraining_report.json")
        run_id = run.info.run_id

    write_state(state_path, {
        "last_completed_at": datetime.now(timezone.utc).isoformat(),
        "feedback_signature": signature, "run_id": run_id,
        "eligible_feedback_rows": len(new), "quality_gate_passed": gate,
        "candidate_threshold": threshold,
    })
    LOG.info("Retraining complete: MLflow run %s; quality gate=%s; deployed model unchanged", run_id, gate)
    return "trained_candidate"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--once", action="store_true", help="Run one feedback check")
    mode.add_argument("--watch", action="store_true", help="Run checks continuously")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--min-feedback", type=int, default=20)
    parser.add_argument("--min-fraud", type=int, default=3,
                        help="Minimum new labelled fraud AND legitimate examples")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.poll_seconds < 1 or args.min_feedback < 1 or args.min_fraud < 1:
        parser.error("all numeric options must be positive")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.once:
        print(run_once(min_feedback=args.min_feedback, min_fraud=args.min_fraud, dry_run=args.dry_run))
        return
    LOG.info("Feedback auto-retrain worker running; checks every %ss", args.poll_seconds)
    while True:
        try:
            run_once(min_feedback=args.min_feedback, min_fraud=args.min_fraud, dry_run=args.dry_run)
        except Exception:
            LOG.exception("Retraining attempt failed; existing serving model remains untouched")
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
