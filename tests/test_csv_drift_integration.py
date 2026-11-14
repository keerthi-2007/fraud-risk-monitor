"""Regression tests for CSV scoring with drift monitoring."""

from unittest.mock import patch

import pandas as pd

from backend.main import (
    app,
    DRIFT_ANALYSES,
    DRIFT_PSI,
    DRIFT_LAST_ANALYSIS,
)
from backend.drift_monitor import FEATURE_COLUMNS
from tests.test_api import make_client, csv_upload, valid_features


def make_csv(rows):
    return pd.DataFrame(
        [valid_features(50) for _ in range(rows)]
    ).to_csv(index=False).encode()


def fake_drift_result(status="stable", psi=0.01):
    return {
        "reference_rows": 170883,
        "current_rows": 100,
        "features_checked": 29,
        "high_drift_features": (
            ["Amount"] if status == "high" else []
        ),
        "features": {
            feature: {
                "psi": psi if feature == "Amount" else 0.01,
                "status": status if feature == "Amount" else "stable",
            }
            for feature in FEATURE_COLUMNS
        },
    }


def test_small_csv_scores_without_drift():
    client = make_client()

    with patch(
        "backend.main.detect_drift_from_baseline"
    ) as detector:
        response = csv_upload(client, make_csv(25))

    assert response.status_code == 200
    assert response.headers["X-Processed-Rows"] == "25"
    detector.assert_not_called()


def test_large_csv_triggers_drift(monkeypatch):
    client = make_client()
    baseline = {"test": "baseline"}

    monkeypatch.setattr(
        app.state,
        "drift_baseline",
        baseline,
        raising=False,
    )

    with patch(
        "backend.main.detect_drift_from_baseline",
        return_value=fake_drift_result(),
    ) as detector:
        response = csv_upload(client, make_csv(100))

    assert response.status_code == 200
    assert response.headers["X-Processed-Rows"] == "100"

    detector.assert_called_once()

    args = detector.call_args.args
    assert args[0] is baseline
    assert len(args[1]) == 100


def test_successful_drift_updates_metrics(monkeypatch):
    client = make_client()

    monkeypatch.setattr(
        app.state,
        "drift_baseline",
        {"test": "baseline"},
        raising=False,
    )

    counter = DRIFT_ANALYSES.labels(status="high")
    before = counter._value.get()

    with patch(
        "backend.main.detect_drift_from_baseline",
        return_value=fake_drift_result(
            status="high",
            psi=0.35,
        ),
    ):
        response = csv_upload(client, make_csv(100))

    assert response.status_code == 200
    assert counter._value.get() == before + 1

    assert DRIFT_PSI.labels(
        feature="Amount"
    )._value.get() == 0.35

    assert DRIFT_LAST_ANALYSIS._value.get() > 0


def test_monitoring_failure_does_not_break_scoring(
    monkeypatch,
    caplog,
):
    client = make_client()

    monkeypatch.setattr(
        app.state,
        "drift_baseline",
        {"test": "baseline"},
        raising=False,
    )

    counter = DRIFT_ANALYSES.labels(status="error")
    before = counter._value.get()

    # Simulate a previous successful drift analysis.
    DRIFT_LAST_ANALYSIS.set(1_700_000_000)

    with patch(
        "backend.main.detect_drift_from_baseline",
        side_effect=RuntimeError("Drift calculation failed"),
    ):
        response = csv_upload(client, make_csv(100))

    assert response.status_code == 200
    assert response.headers["X-Processed-Rows"] == "100"
    assert counter._value.get() == before + 1
    assert DRIFT_LAST_ANALYSIS._value.get() == 0
    assert "CSV drift monitoring failed" in caplog.text
