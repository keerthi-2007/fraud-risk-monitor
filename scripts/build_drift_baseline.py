"""Build a feature-distribution baseline using training data only."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.drift_detector import FEATURE_COLUMNS

ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "data/processed/splits/train.csv"
OUTPUT_PATH = ROOT / "monitoring/drift_baseline.json"

BINS = 10


def build_baseline():
    frame = pd.read_csv(TRAIN_PATH, usecols=FEATURE_COLUMNS)

    if len(frame) < 100:
        raise ValueError("Insufficient training rows for drift baseline.")

    baseline = {
        "schema_version": 1,
        "source": "training_split",
        "reference_rows": len(frame),
        "requested_bins": BINS,
        "features": {},
    }

    for feature in FEATURE_COLUMNS:
        values = frame[feature].to_numpy(dtype=float)

        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite training values: {feature}")

        edges = np.unique(
            np.quantile(values, np.linspace(0, 1, BINS + 1))
        )

        if len(edges) == 1:
            counts = [len(values)]
        else:
            counts = np.bincount(
                np.searchsorted(edges[1:-1], values, side="right"),
                minlength=len(edges) - 1,
            ).tolist()

        baseline["features"][feature] = {
            "edges": edges.tolist(),
            "counts": counts,
        }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(baseline, indent=2, allow_nan=False) + "\n"
    )

    print("Baseline saved:", OUTPUT_PATH)
    print("Reference rows:", baseline["reference_rows"])
    print("Features:", len(baseline["features"]))

    assert len(baseline["features"]) == 29

    for feature, details in baseline["features"].items():
        assert sum(details["counts"]) == len(frame), feature

    print("Baseline validation PASSED.")


if __name__ == "__main__":
    build_baseline()
