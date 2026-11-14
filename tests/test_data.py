import json
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = ROOT / "data" / "processed" / "splits"
MANIFEST = json.loads((SPLIT_DIR / "split_manifest.json").read_text())

FEATURES = [f"V{i}" for i in range(1, 29)] + ["Amount"]
SPLITS = ("train", "validation", "test")


@pytest.fixture(scope="module")
def datasets():
    return {
        name: pd.read_csv(SPLIT_DIR / f"{name}.csv")
        for name in SPLITS
    }


def test_expected_columns(datasets):
    for name, frame in datasets.items():
        assert list(frame.columns) == FEATURES + ["Class"], name


def test_no_missing_values(datasets):
    for name, frame in datasets.items():
        assert not frame.isna().any().any(), name


def test_valid_class_labels(datasets):
    for name, frame in datasets.items():
        assert set(frame["Class"].unique()) == {0, 1}, name


def test_manifest_counts(datasets):
    for name, frame in datasets.items():
        expected = MANIFEST["splits"][name]
        assert len(frame) == expected["rows"]
        assert int(frame["Class"].sum()) == expected["fraud"]
        assert int((frame["Class"] == 0).sum()) == expected["legitimate"]


def test_no_feature_overlap_between_splits(datasets):
    fingerprints = {}

    for name, frame in datasets.items():
        fingerprints[name] = set(
            pd.util.hash_pandas_object(
                frame[FEATURES],
                index=False,
            ).tolist()
        )

    for i, first in enumerate(SPLITS):
        for second in SPLITS[i + 1:]:
            assert fingerprints[first].isdisjoint(
                fingerprints[second]
            ), f"Feature overlap: {first} and {second}"
