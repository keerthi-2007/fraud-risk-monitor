from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "data" / "raw" / "creditcard.csv"
REPORT_PATH = ROOT / "data" / "reports" / "duplicate_analysis.md"

df = pd.read_csv(CSV_PATH)
feature_columns = [column for column in df.columns if column != "Class"]

feature_hashes = pd.util.hash_pandas_object(
    df[feature_columns],
    index=False,
    categorize=True,
)

groups = pd.DataFrame(
    {
        "feature_hash": feature_hashes,
        "Class": df["Class"].astype(int),
    }
)

summary = groups.groupby("feature_hash")["Class"].agg(
    row_count="size",
    label_count="nunique",
)

repeated = summary[summary["row_count"] > 1]
repeated_hashes = repeated.index

rows_in_repeated_groups = groups[
    groups["feature_hash"].isin(repeated_hashes)
]
repeated_class_counts = (
    rows_in_repeated_groups["Class"]
    .value_counts()
    .sort_index()
)

conflicting_groups = int((repeated["label_count"] > 1).sum())
rows_in_repeated_groups_count = int(repeated["row_count"].sum())
extra_rows_if_collapsed = int((repeated["row_count"] - 1).sum())
largest_group = int(repeated["row_count"].max()) if len(repeated) else 1

report = f"""# Repeated feature analysis

- Total rows: {len(df):,}
- Unique feature vectors: {len(summary):,}
- Repeated feature groups: {len(repeated):,}
- Rows in repeated groups: {rows_in_repeated_groups_count:,}
- Extra rows beyond one per repeated group: {extra_rows_if_collapsed:,}
- Largest repeated group: {largest_group:,} rows
- Repeated groups containing both labels: {conflicting_groups:,}

## Labels among rows in repeated groups

- Legitimate rows: {int(repeated_class_counts.get(0, 0)):,}
- Fraudulent rows: {int(repeated_class_counts.get(1, 0)):,}

The raw CSV was read but not modified. Identical feature vectors do not establish
that records refer to the same transaction, especially because this dataset
version has no Time column.
"""

REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
REPORT_PATH.write_text(report)

print(report)
print(f"Saved report to: {REPORT_PATH}")
