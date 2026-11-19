"""Tests for the /explain endpoint and reference-value derivation."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.main import (
    FEATURE_COLUMNS,
    app,
    reference_values_from_baseline,
)


class AmountDrivenModel:
    """Score depends only on Amount, so only Amount should get credit."""

    def predict_proba(self, frame):
        scores = np.where(frame["Amount"].to_numpy() >= 100, 0.95, 0.10)
        return np.column_stack((1 - scores, scores))


def make_client(references=None):
    app.state.model = AmountDrivenModel()
    app.state.model_run_id = "test-model"
    app.state.reference_values = references or {
        name: 0.0 for name in FEATURE_COLUMNS
    }
    return TestClient(app)


def features(amount):
    return {**{f"V{i}": 0.0 for i in range(1, 29)}, "Amount": amount}


def test_explain_attributes_score_to_driving_feature():
    response = make_client().post("/explain", json={"features": features(150)})

    assert response.status_code == 200
    body = response.json()
    assert body["decision"] == "review"
    assert body["method"] == "single_feature_occlusion"
    assert body["reference_score"] == pytest.approx(0.10)
    assert len(body["contributions"]) == len(FEATURE_COLUMNS)

    top = body["contributions"][0]
    assert top["feature"] == "Amount"
    assert top["value"] == 150
    assert top["reference_value"] == 0.0
    assert top["contribution"] == pytest.approx(0.85)
    assert all(item["contribution"] == 0 for item in body["contributions"][1:])


def test_explain_normal_transaction_has_no_positive_drivers():
    response = make_client().post("/explain", json={"features": features(10)})

    assert response.status_code == 200
    body = response.json()
    assert body["decision"] == "normal"
    assert all(item["contribution"] == 0 for item in body["contributions"])


def test_explain_rejects_invalid_payload():
    payload = features(10)
    del payload["V3"]
    response = make_client().post("/explain", json={"features": payload})
    assert response.status_code == 422

    response = make_client().post("/explain", json={"features": features(-1)})
    assert response.status_code == 422


def test_reference_values_interpolate_training_median():
    baseline = {
        "features": {
            name: {"edges": [0.0, 1.0, 2.0, 3.0, 4.0], "counts": [1, 1, 1, 1]}
            for name in FEATURE_COLUMNS
        }
    }
    references = reference_values_from_baseline(baseline)
    assert references["V1"] == pytest.approx(2.0)
    assert set(references) == set(FEATURE_COLUMNS)


def test_reference_values_from_repository_baseline_are_finite():
    from backend.drift_monitor import load_baseline
    from backend.main import DRIFT_BASELINE_PATH

    references = reference_values_from_baseline(load_baseline(DRIFT_BASELINE_PATH))
    assert all(np.isfinite(value) for value in references.values())
    assert references["Amount"] >= 0
