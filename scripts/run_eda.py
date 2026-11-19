"""DVC stage `eda`: regenerate exploratory charts and the EDA summary report."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "data" / "raw" / "creditcard.csv"
REPORT_DIR = ROOT / "data" / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CSV_PATH)
feature_columns = [f"V{i}" for i in range(1, 29)]
model_columns = feature_columns + ["Amount"]
class_counts = df["Class"].value_counts().sort_index()
fraud_rate = df["Class"].mean() * 100

numeric = df.select_dtypes(include="number")
non_finite_count = int((~np.isfinite(numeric)).sum().sum())

# Class balance
fig, ax = plt.subplots(figsize=(7, 4))
labels = ["Legitimate", "Fraud"]
counts = [int(class_counts.get(0, 0)), int(class_counts.get(1, 0))]
bars = ax.bar(labels, counts, color=["#2563eb", "#dc2626"])
ax.set_yscale("log")
ax.set_ylabel("Transactions (log scale)")
ax.set_title("Class balance")
for bar, count in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width() / 2, count, f"{count:,}",
            ha="center", va="bottom")
fig.tight_layout()
fig.savefig(REPORT_DIR / "eda_class_balance.png", dpi=150)
plt.close(fig)

# Amount by class. Shared bins make the class shapes comparable.
amount_display = np.log1p(df["Amount"].to_numpy())
amount_bins = np.histogram_bin_edges(amount_display, bins=50)

fig, ax = plt.subplots(figsize=(8, 4))
for label, name, color in [(0, "Legitimate", "#2563eb"),
                           (1, "Fraud", "#dc2626")]:
    values = np.log1p(
        df.loc[df["Class"] == label, "Amount"].to_numpy()
    )
    ax.hist(values, bins=amount_bins, density=True, alpha=0.55,
            label=name, color=color)
ax.set_xlabel("log1p(Amount), for display only")
ax.set_ylabel("Density")
ax.set_title("Transaction amount distribution by class")
ax.legend()
fig.tight_layout()
fig.savefig(REPORT_DIR / "eda_amount_by_class.png", dpi=150)
plt.close(fig)

# Distributions for anonymized PCA features, using shared bins per feature.
fig, axes = plt.subplots(7, 4, figsize=(16, 20))
for index, column in enumerate(feature_columns):
    ax = axes.flat[index]
    shared_bins = np.histogram_bin_edges(df[column].to_numpy(), bins=40)

    for label, name, color in [(0, "Legitimate", "#2563eb"),
                               (1, "Fraud", "#dc2626")]:
        values = df.loc[df["Class"] == label, column].to_numpy()
        ax.hist(values, bins=shared_bins, density=True, alpha=0.5,
                label=name, color=color)

    ax.set_title(column)
    ax.tick_params(labelsize=7)

axes.flat[0].legend(fontsize=7)
fig.suptitle("V1–V28 distributions by class (density)", y=1.002)
fig.tight_layout()
fig.savefig(REPORT_DIR / "eda_feature_distributions.png",
            dpi=150, bbox_inches="tight")
plt.close(fig)

# Correlations are descriptive; they do not establish causation.
correlation = df.corr(numeric_only=True)
fig, ax = plt.subplots(figsize=(12, 10))
image = ax.imshow(correlation, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(len(correlation.columns)))
ax.set_yticks(range(len(correlation.columns)))
ax.set_xticklabels(correlation.columns, rotation=90, fontsize=7)
ax.set_yticklabels(correlation.columns, fontsize=7)
ax.set_title("Feature and label correlations (Pearson)")
fig.colorbar(image, ax=ax, shrink=0.75)
fig.tight_layout()
fig.savefig(REPORT_DIR / "eda_correlations.png", dpi=150)
plt.close(fig)

# Summarize repeated feature-vector group sizes without removing rows.
feature_hashes = pd.util.hash_pandas_object(
    df[model_columns], index=False, categorize=True
)
group_sizes = feature_hashes.value_counts()
repeated_sizes = group_sizes[group_sizes > 1]
bins = [
    int((repeated_sizes == 2).sum()),
    int(repeated_sizes.between(3, 5).sum()),
    int(repeated_sizes.between(6, 10).sum()),
    int((repeated_sizes >= 11).sum()),
]
bin_labels = ["2 rows", "3–5 rows", "6–10 rows", "11+ rows"]

fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(bin_labels, bins, color="#7c3aed")
ax.set_ylabel("Repeated feature groups")
ax.set_title("Size of repeated feature-vector groups")
fig.tight_layout()
fig.savefig(REPORT_DIR / "eda_duplicate_groups.png", dpi=150)
plt.close(fig)

amount_summary = (
    df.groupby("Class")["Amount"]
    .describe(percentiles=[0.50, 0.95, 0.99])
    .round(4)
    .to_string()
)
feature_correlations = (
    df[model_columns + ["Class"]]
    .corr(numeric_only=True)["Class"]
    .drop("Class")
    .sort_values(key=abs, ascending=False)
    .head(10)
    .round(4)
    .to_string()
)

report = f"""# Exploratory data analysis

## Dataset overview

- Rows: {len(df):,}
- Columns: {len(df.columns)}
- Legitimate transactions: {int(class_counts.get(0, 0)):,}
- Fraudulent transactions: {int(class_counts.get(1, 0)):,}
- Fraud rate: {fraud_rate:.4f}%
- Missing cells: {int(df.isna().sum().sum())}
- Non-finite numeric values: {non_finite_count}
- Zero-amount transactions: {int((df["Amount"] == 0).sum()):,}
- Negative-amount transactions: {int((df["Amount"] < 0).sum()):,}
- Repeated feature groups: {len(repeated_sizes):,}
- Rows in repeated groups: {int(repeated_sizes.sum()):,}

## Amount summary by class

{amount_summary}

## Ten largest absolute correlations with Class

These are descriptive correlations for anonymized inputs.

{feature_correlations}

## Charts

- `eda_class_balance.png`
- `eda_amount_by_class.png`
- `eda_feature_distributions.png`
- `eda_correlations.png`
- `eda_duplicate_groups.png`

The raw CSV was read but not modified. Charting with log1p changes only the
display scale; it does not transform or rewrite the dataset.
"""

report_path = REPORT_DIR / "eda_summary.md"
report_path.write_text(report)

print(report)
print(f"Saved report and charts under: {REPORT_DIR}")
