"""Feature-distribution drift detection using Population Stability Index."""

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [f"V{i}" for i in range(1, 29)] + ["Amount"]

MIN_ROWS = 100
WARNING_PSI = 0.10
HIGH_PSI = 0.20


def calculate_psi(reference, current, bins=10):
    """Compare numeric distributions using reference quantile bins."""
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)

    if reference.size < MIN_ROWS or current.size < MIN_ROWS:
        raise ValueError(
            f"Both datasets must contain at least {MIN_ROWS} rows."
        )

    if not np.isfinite(reference).all() or not np.isfinite(current).all():
        raise ValueError("Drift inputs must contain finite numeric values.")

    if bins < 2:
        raise ValueError("At least two bins are required.")

    edges = np.unique(
        np.quantile(reference, np.linspace(0, 1, bins + 1))
    )

    if len(edges) < 2:
        # Constant reference feature: a changed value is drift.
        return 0.0 if np.all(current == reference[0]) else float("inf")

    boundaries = edges[1:-1]

    reference_counts = np.bincount(
        np.searchsorted(boundaries, reference, side="right"),
        minlength=len(edges) - 1,
    )
    current_counts = np.bincount(
        np.searchsorted(boundaries, current, side="right"),
        minlength=len(edges) - 1,
    )

    # Smoothing prevents logarithms of zero.
    epsilon = 1e-6
    expected = np.maximum(reference_counts / reference.size, epsilon)
    observed = np.maximum(current_counts / current.size, epsilon)

    return float(
        np.sum((observed - expected) * np.log(observed / expected))
    )


def detect_drift(reference_df, current_df):
    """Return per-feature PSI and an overall drift assessment."""
    for name, frame in (
        ("Reference", reference_df),
        ("Current", current_df),
    ):
        missing = sorted(set(FEATURE_COLUMNS) - set(frame.columns))
        if missing:
            raise ValueError(f"{name} data missing features: {missing}")

        if len(frame) < MIN_ROWS:
            raise ValueError(
                f"{name} data requires at least {MIN_ROWS} rows."
            )

    results = {}

    for feature in FEATURE_COLUMNS:
        psi = calculate_psi(
            reference_df[feature],
            current_df[feature],
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
        "reference_rows": len(reference_df),
        "current_rows": len(current_df),
        "features_checked": len(FEATURE_COLUMNS),
        "high_drift_features": [
            name
            for name, result in results.items()
            if result["status"] == "high"
        ],
        "features": results,
    }
