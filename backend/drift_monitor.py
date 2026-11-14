"""Compare incoming feature distributions with a saved training baseline."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [f"V{i}" for i in range(1, 29)] + ["Amount"]

MIN_ROWS = 100
WARNING_PSI = 0.10
HIGH_PSI = 0.20


def load_baseline(path: str | Path) -> dict:
    """Load and validate the reference-distribution artifact."""
    with Path(path).open(encoding="utf-8") as handle:
        baseline = json.load(handle)

    if baseline.get("schema_version") != 1:
        raise ValueError("Unsupported drift baseline schema.")

    if baseline.get("reference_rows", 0) < MIN_ROWS:
        raise ValueError("Insufficient reference rows.")

    features = baseline.get("features", {})

    if set(features) != set(FEATURE_COLUMNS):
        raise ValueError("Drift baseline feature schema mismatch.")

    for feature in FEATURE_COLUMNS:
        details = features[feature]
        edges = np.asarray(details["edges"], dtype=float)
        counts = np.asarray(details["counts"], dtype=float)

        if (
            edges.ndim != 1
            or counts.ndim != 1
            or len(edges) < 1
            or len(counts) != max(1, len(edges) - 1)
            or not np.isfinite(edges).all()
            or not np.isfinite(counts).all()
            or np.any(counts < 0)
            or np.any(np.diff(edges) <= 0)
            or counts.sum() != baseline["reference_rows"]
        ):
            raise ValueError(f"Invalid drift baseline for {feature}.")

    return baseline


def detect_drift_from_baseline(
    baseline: dict,
    current: pd.DataFrame,
) -> dict:
    """Calculate per-feature PSI using stored reference histogram counts."""
    missing = sorted(set(FEATURE_COLUMNS) - set(current.columns))

    if missing:
        raise ValueError(f"Missing features: {missing}")

    if len(current) < MIN_ROWS:
        raise ValueError(f"At least {MIN_ROWS} incoming rows required.")

    results = {}

    for feature in FEATURE_COLUMNS:
        values = pd.to_numeric(current[feature], errors="coerce").to_numpy(
            dtype=float
        )

        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite values in {feature}.")

        reference = baseline["features"][feature]
        edges = np.asarray(reference["edges"], dtype=float)
        reference_counts = np.asarray(reference["counts"], dtype=float)

        if len(edges) == 1:
            psi = (
                0.0
                if np.all(values == edges[0])
                else float("inf")
            )
        else:
            counts = np.bincount(
                np.searchsorted(edges[1:-1], values, side="right"),
                minlength=len(reference_counts),
            )

            epsilon = 1e-6
            expected = np.maximum(
                reference_counts / baseline["reference_rows"], epsilon
            )
            observed = np.maximum(counts / len(values), epsilon)

            psi = float(
                np.sum((observed - expected) * np.log(observed / expected))
            )

        status = (
            "high"
            if psi >= HIGH_PSI
            else "warning"
            if psi >= WARNING_PSI
            else "stable"
        )

        results[feature] = {
            "psi": psi,
            "status": status,
        }

    return {
        "reference_rows": baseline["reference_rows"],
        "current_rows": len(current),
        "features_checked": len(FEATURE_COLUMNS),
        "high_drift_features": [
            feature
            for feature, result in results.items()
            if result["status"] == "high"
        ],
        "features": results,
    }
