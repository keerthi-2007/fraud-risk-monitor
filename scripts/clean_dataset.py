"""DVC stage `prepare_data`: verify the raw CSV checksum, validate schema and values,
and write the prepared dataset plus a cleaning report.
"""

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "creditcard.csv"
MANIFEST_PATH = ROOT / "data" / "raw" / "manifest.json"
OUTPUT_DIR = ROOT / "data" / "processed"
OUTPUT_PATH = OUTPUT_DIR / "creditcard_prepared.csv"
REPORT_PATH = ROOT / "data" / "reports" / "cleaning_report.md"

manifest = json.loads(MANIFEST_PATH.read_text())

digest = sha256()
with RAW_PATH.open("rb") as file:
    for chunk in iter(lambda: file.read(1024 * 1024), b""):
        digest.update(chunk)

if digest.hexdigest() != manifest["csv_sha256"]:
    raise ValueError("Raw CSV checksum differs from the download manifest.")

df = pd.read_csv(RAW_PATH)

feature_columns = [f"V{i}" for i in range(1, 29)] + ["Amount"]
expected_columns = feature_columns + ["Class"]

if list(df.columns) != expected_columns:
    raise ValueError(f"Unexpected columns: {list(df.columns)}")

if df.isna().any().any():
    raise ValueError("Missing values found.")

df[expected_columns] = df[expected_columns].apply(
    pd.to_numeric,
    errors="raise",
)

if not np.isfinite(df[expected_columns].to_numpy(dtype=float)).all():
    raise ValueError("Non-finite numeric values found.")

df["Class"] = df["Class"].astype("int8")

if set(df["Class"].unique()) != {0, 1}:
    raise ValueError(f"Unexpected class labels: {sorted(df['Class'].unique())}")

if (df["Amount"] < 0).any():
    raise ValueError("Negative Amount values found; inspect before continuing.")

if len(df) != 284_807:
    raise ValueError(f"Expected 284807 rows, found {len(df)}")

if int((df["Class"] == 1).sum()) != 492:
    raise ValueError("Fraud count differs from the expected OpenML dataset.")

# Keep identical feature vectors together during later train/validation/test splits.
feature_hashes = pd.util.hash_pandas_object(
    df[feature_columns],
    index=False,
    categorize=True,
)
group_ids, _ = pd.factorize(feature_hashes, sort=False)

prepared = df[expected_columns].copy()
prepared["feature_group_id"] = group_ids.astype("int32")

label_counts_per_group = prepared.groupby("feature_group_id")["Class"].nunique()
conflicting_groups = int((label_counts_per_group > 1).sum())
if conflicting_groups:
    raise ValueError(
        f"{conflicting_groups} feature groups contain conflicting class labels."
    )

class_counts = Counter(prepared["Class"].astype(int))
zero_amount_counts = Counter(
    prepared.loc[prepared["Amount"] == 0, "Class"].astype(int)
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
prepared.to_csv(OUTPUT_PATH, index=False)

report = f"""# Cleaning and validation report

- Input: `data/raw/creditcard.csv`
- Output: `data/processed/creditcard_prepared.csv`
- Rows before: {len(df):,}
- Rows after: {len(prepared):,}
- Rows removed: {len(df) - len(prepared):,}
- Missing values: {int(prepared.isna().sum().sum())}
- Non-finite values: 0
- Negative Amount values: 0
- Legitimate rows: {class_counts[0]:,}
- Fraud rows: {class_counts[1]:,}
- Zero-amount legitimate rows: {zero_amount_counts[0]:,}
- Zero-amount fraud rows: {zero_amount_counts[1]:,}
- Unique feature groups: {len(label_counts_per_group):,}
- Groups with conflicting labels: {conflicting_groups}

## Decisions

- The raw CSV was preserved.
- Zero-amount and high-amount rows were retained.
- Repeated rows were retained; their feature groups are recorded for later splitting.
- `feature_group_id` is metadata for splitting and must be excluded from model inputs.
- Scaling or other learned transformations will be fitted after splitting, using training data only.
- No synthetic records were created.
"""

REPORT_PATH.write_text(report)
print(report)
print(f"Saved prepared data to: {OUTPUT_PATH}")
print(f"Saved cleaning report to: {REPORT_PATH}")
