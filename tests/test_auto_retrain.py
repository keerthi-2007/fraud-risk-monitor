"""Lightweight unit checks for feedback eligibility and evaluation selection."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "auto_retrain.py"
spec = importlib.util.spec_from_file_location("auto_retrain", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def values(k):
    out = {name: float(i + k) for i, name in enumerate(module.FEATURES)}
    out["Amount"] = 10.0 + k
    return out


def make_approvals(records):
    """Create matching approval fixtures for unit tests only."""
    from scripts.approve_feedback import record_hash

    approvals = {}
    for index, record in enumerate(records):
        feedback_id = f"unit-test-{index}"
        record["feedback_id"] = feedback_id
        approvals[feedback_id] = {
            "feedback_id": feedback_id,
            "record_sha256": record_hash(record),
            "approved_label": record["corrected_label"],
            "verified_by": "unit-test-reviewer",
            "evidence_reference": "unit-test-fixture",
        }
    return approvals


def test_rejects_missing_or_invalid_features():
    with pytest.raises(ValueError):
        module.fingerprint({"Amount": 1})
    row = values(0)
    row["Amount"] = -1
    with pytest.raises(ValueError):
        module.fingerprint(row)


def test_feedback_excludes_known_and_disagreements(tmp_path):
    lines = [
        {"features": values(0), "corrected_label": "fraud"},
        {"features": values(1), "corrected_label": "fraud"},
        {"features": values(1), "corrected_label": "legitimate"},
        {"features": values(2), "corrected_label": "legitimate"},
        {"features": values(2), "corrected_label": "legitimate"},
        {"features": {"bad": 1}, "corrected_label": "fraud"},
    ]
    path = tmp_path / "feedback.jsonl"
    approvals = make_approvals(lines)
    path.write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    frame, counts = module.read_feedback(
        path, {module.fingerprint(values(0))}, approved=approvals
    )
    assert len(frame) == 1 and frame["Class"].iat[0] == 0
    assert counts == {"records": 6, "invalid": 1, "duplicates": 1,
                      "known_dataset": 1, "conflicting_labels": 1,
                      "eligible": 1, "not_approved": 0}


def test_threshold_meets_recall_target():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    scores = np.array([0.02, 0.1, 0.2, 0.4, 0.55, 0.7, 0.8, 0.9])
    threshold, result = module.choose_threshold(y, scores)
    assert result["recall"] >= 0.85
    assert result["fraud_caught"] == 4
    assert result["alerts"] == 4
    assert threshold == pytest.approx(0.55)


def test_retraining_fails_closed_without_test_split(tmp_path):
    root = tmp_path / "fraud-risk-monitor"
    split_dir = root / "data" / "processed" / "splits"
    split_dir.mkdir(parents=True)
    frame = pd.DataFrame([{**values(k), "Class": k % 2} for k in range(5)])
    frame.to_csv(split_dir / "train.csv", index=False)
    frame.to_csv(split_dir / "validation.csv", index=False)
    with pytest.raises(FileNotFoundError, match="test.csv"):
        module.run_once(root=root, min_feedback=1, min_fraud=1, dry_run=True)


def test_eligible_feedback_dry_run_without_touching_mlflow(tmp_path):
    root = tmp_path / "fraud-risk-monitor"
    split_dir = root / "data" / "processed" / "splits"
    split_dir.mkdir(parents=True)
    originals = pd.DataFrame([{**values(k), "Class": k % 2} for k in range(6)])
    for name, start, end in [("train", 0, 2), ("validation", 2, 4), ("test", 4, 6)]:
        originals.iloc[start:end].to_csv(split_dir / f"{name}.csv", index=False)
    fb = root / "data" / "feedback" / "feedback.jsonl"
    fb.parent.mkdir(parents=True)
    fresh = [{"features": values(100 + k), "corrected_label": "fraud" if k % 2 else "legitimate"}
             for k in range(4)]
    approvals = make_approvals(fresh)
    fb.write_text("\n".join(json.dumps(x) for x in fresh) + "\n")

    approval_path = fb.parent / "approved_feedback.jsonl"
    approval_path.write_text(
        "\n".join(json.dumps(x) for x in approvals.values()) + "\n"
    )

    result = module.run_once(root=root, min_feedback=4, min_fraud=2, dry_run=True)
    assert result == "eligible_dry_run"
    assert not (fb.parent / "auto_retrain_state.json").exists()
