from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "data" / "raw" / "creditcard.csv"
MANIFEST_PATH = ROOT / "data" / "raw" / "manifest.json"
REPORT_PATH = ROOT / "data" / "reports" / "dataset_quality.md"

manifest = json.loads(MANIFEST_PATH.read_text())

digest = sha256()
with CSV_PATH.open("rb") as file:
    for chunk in iter(lambda: file.read(1024 * 1024), b""):
        digest.update(chunk)

if digest.hexdigest() != manifest["csv_sha256"]:
    raise ValueError("CSV checksum differs from the download manifest.")

df = pd.read_csv(CSV_PATH)

expected = [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]
if list(df.columns) != expected:
    raise ValueError(f"Unexpected columns: {list(df.columns)}")

if len(df) != 284_807:
    raise ValueError(f"Expected 284807 rows, found {len(df)}")

if df.isna().any().any():
    raise ValueError("Missing values found; inspect the CSV before continuing.")

if set(df["Class"].unique()) != {0, 1}:
    raise ValueError(f"Unexpected Class labels: {sorted(df['Class'].unique())}")

class_counts = Counter(df["Class"].astype(int))
legitimate_count = int(class_counts[0])
fraud_count = int(class_counts[1])

duplicate_rows = int(df.duplicated().sum())
feature_columns = [column for column in df.columns if column != "Class"]
duplicate_feature_rows = int(df.duplicated(subset=feature_columns).sum())

amount = df["Amount"].describe(
    percentiles=[0.50, 0.95, 0.99]
).round(4)

amount_lines = []
for name, value in amount.items():
    if name == "count":
        amount_lines.append(f"- {name}: {int(value):,}")
    else:
        amount_lines.append(f"- {name}: {value:,.4f}")

report = f"""# Dataset quality report

- Source: OpenML dataset 1597
- Local file: `data/raw/creditcard.csv`
- Rows: {len(df):,}
- Columns: {len(df.columns)}
- Missing cells: {int(df.isna().sum().sum())}
- Exact duplicate rows: {duplicate_rows:,}
- Duplicate feature rows (excluding the label): {duplicate_feature_rows:,}
- Legitimate transactions: {legitimate_count:,}
- Fraudulent transactions: {fraud_count:,}
- Fraud rate: {fraud_count / len(df) * 100:.4f}%

## Amount summary

{chr(10).join(amount_lines)}

## Columns

{", ".join(df.columns)}
"""

REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
REPORT_PATH.write_text(report)

print(report)
print(f"Saved report to: {REPORT_PATH}")
