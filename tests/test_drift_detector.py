import numpy as np
import pandas as pd
import pytest

from scripts.drift_detector import (
    FEATURE_COLUMNS,
    calculate_psi,
    detect_drift,
)


def make_dataset(rows=200):
    values = np.linspace(-3, 3, rows)

    return pd.DataFrame({
        feature: values.copy()
        for feature in FEATURE_COLUMNS
    })


def test_identical_distributions_have_zero_psi():
    values = np.linspace(-3, 3, 200)

    psi = calculate_psi(values, values.copy())

    assert psi == pytest.approx(0.0)


def test_shifted_distribution_detected():
    reference = np.linspace(-3, 3, 200)
    current = reference + 5

    psi = calculate_psi(reference, current)

    assert psi >= 0.20


def test_detect_drift_checks_all_features():
    reference = make_dataset()
    current = reference.copy()

    result = detect_drift(reference, current)

    assert result["features_checked"] == 29
    assert result["reference_rows"] == 200
    assert result["current_rows"] == 200
    assert result["high_drift_features"] == []

    assert all(
        item["status"] == "stable"
        for item in result["features"].values()
    )


def test_detect_drift_flags_shifted_feature():
    reference = make_dataset()
    current = reference.copy()

    current["Amount"] += 10

    result = detect_drift(reference, current)

    assert "Amount" in result["high_drift_features"]
    assert result["features"]["Amount"]["status"] == "high"


def test_missing_features_rejected():
    reference = make_dataset()
    current = make_dataset().drop(columns=["V1"])

    with pytest.raises(ValueError, match="missing features"):
        detect_drift(reference, current)


def test_small_samples_rejected():
    reference = np.arange(200)
    current = np.arange(25)

    with pytest.raises(ValueError, match="at least 100 rows"):
        calculate_psi(reference, current)


def test_nonfinite_values_rejected():
    reference = np.arange(200, dtype=float)
    current = reference.copy()
    current[0] = np.nan

    with pytest.raises(ValueError, match="finite"):
        calculate_psi(reference, current)


def test_constant_identical_feature_is_stable():
    reference = np.ones(200)
    current = np.ones(200)

    assert calculate_psi(reference, current) == 0.0


def test_constant_feature_change_detected():
    reference = np.ones(200)
    current = np.full(200, 2.0)

    assert calculate_psi(reference, current) == float("inf")
