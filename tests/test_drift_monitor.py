"""Tests for the compact-baseline drift monitor."""

import json

import numpy as np
import pandas as pd
import pytest

from backend.drift_monitor import (
    FEATURE_COLUMNS,
    detect_drift_from_baseline,
    load_baseline,
)
from scripts.drift_detector import detect_drift


def make_data(rows=200):
    values = np.linspace(-3, 3, rows)
    return pd.DataFrame({
        feature: values.copy()
        for feature in FEATURE_COLUMNS
    })


def make_baseline(frame):
    features = {}

    for feature in FEATURE_COLUMNS:
        values = frame[feature].to_numpy(dtype=float)
        edges = np.unique(
            np.quantile(values, np.linspace(0, 1, 11))
        )

        counts = (
            [len(values)]
            if len(edges) == 1
            else np.bincount(
                np.searchsorted(edges[1:-1], values, side="right"),
                minlength=len(edges) - 1,
            ).tolist()
        )

        features[feature] = {
            "edges": edges.tolist(),
            "counts": counts,
        }

    return {
        "schema_version": 1,
        "source": "training_split",
        "reference_rows": len(frame),
        "requested_bins": 10,
        "features": features,
    }


def test_baseline_matches_direct_psi():
    reference = make_data()
    current = reference.copy()
    current["Amount"] += 5

    baseline = make_baseline(reference)

    direct = detect_drift(reference, current)
    compact = detect_drift_from_baseline(baseline, current)

    for feature in FEATURE_COLUMNS:
        assert compact["features"][feature]["psi"] == pytest.approx(
            direct["features"][feature]["psi"],
            abs=1e-12,
        )
        assert (
            compact["features"][feature]["status"]
            == direct["features"][feature]["status"]
        )


def test_missing_feature_rejected():
    baseline = make_baseline(make_data())
    current = make_data().drop(columns=["V1"])

    with pytest.raises(ValueError, match="Missing features"):
        detect_drift_from_baseline(baseline, current)


def test_insufficient_rows_rejected():
    baseline = make_baseline(make_data())

    with pytest.raises(ValueError, match="At least 100"):
        detect_drift_from_baseline(baseline, make_data(25))


def test_invalid_values_rejected():
    baseline = make_baseline(make_data())
    current = make_data()
    current.loc[0, "Amount"] = np.nan

    with pytest.raises(ValueError, match="Non-finite"):
        detect_drift_from_baseline(baseline, current)


def test_valid_baseline_loads(tmp_path):
    baseline = make_baseline(make_data())
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps(baseline))

    loaded = load_baseline(path)

    assert loaded["reference_rows"] == 200
    assert len(loaded["features"]) == 29


def test_invalid_baseline_rejected(tmp_path):
    baseline = make_baseline(make_data())
    baseline["features"]["V1"]["counts"][0] = -1

    path = tmp_path / "baseline.json"
    path.write_text(json.dumps(baseline))

    with pytest.raises(ValueError, match="Invalid drift baseline"):
        load_baseline(path)


def test_constant_reference_change_detected():
    reference = pd.DataFrame({
        feature: np.ones(200)
        for feature in FEATURE_COLUMNS
    })
    current = reference.copy()
    current["Amount"] = 2.0

    baseline = make_baseline(reference)
    result = detect_drift_from_baseline(baseline, current)

    assert result["features"]["Amount"]["status"] == "high"
    assert result["features"]["Amount"]["psi"] == float("inf")
