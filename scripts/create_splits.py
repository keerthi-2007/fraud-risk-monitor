"""DVC stage `create_splits`: stratified, group-aware train/validation/test splits.

Rows with an identical feature vector share a group ID, so repeats never cross splits.
"""

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "data" / "processed" / "creditcard_prepared.csv"
SPLIT_DIR = ROOT / "data" / "processed" / "splits"
MANIFEST_PATH = SPLIT_DIR / "split_manifest.json"

df = pd.read_csv(INPUT_PATH)

feature_columns = [f"V{i}" for i in range(1, 29)] + ["Amount"]
required_columns = feature_columns + ["Class", "feature_group_id"]
missing = [column for column in required_columns if column not in df.columns]
if missing:
    raise ValueError(f"Prepared data is missing columns: {missing}")

X = df[feature_columns]
y = df["Class"].astype("int8")
groups = df["feature_group_id"]

# Five folds reserve about 20% for test.
outer_splitter = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)
train_validation_indices, test_indices = next(
    outer_splitter.split(X, y, groups)
)

# Four folds of the remaining 80% reserve about 20% overall for validation.
inner_splitter = StratifiedGroupKFold(
    n_splits=4,
    shuffle=True,
    random_state=42,
)
train_relative_indices, validation_relative_indices = next(
    inner_splitter.split(
        X.iloc[train_validation_indices],
        y.iloc[train_validation_indices],
        groups.iloc[train_validation_indices],
    )
)

train_indices = train_validation_indices[train_relative_indices]
validation_indices = train_validation_indices[validation_relative_indices]

split_indices = {
    "train": train_indices,
    "validation": validation_indices,
    "test": test_indices,
}

group_sets = {
    name: set(groups.iloc[indices].tolist())
    for name, indices in split_indices.items()
}

names = list(group_sets)
for first_index, first_name in enumerate(names):
    for second_name in names[first_index + 1:]:
        overlap = group_sets[first_name] & group_sets[second_name]
        if overlap:
            raise ValueError(
                f"Feature groups overlap between {first_name} and {second_name}."
            )

SPLIT_DIR.mkdir(parents=True, exist_ok=True)
manifest = {
    "method": "StratifiedGroupKFold",
    "outer_folds": 5,
    "inner_folds": 4,
    "random_state": 42,
    "feature_columns": feature_columns,
    "target_column": "Class",
    "feature_group_id_used_for_splitting_only": True,
    "group_overlap_between_splits": 0,
    "splits": {},
}

for name, indices in split_indices.items():
    output = df.iloc[indices][feature_columns + ["Class"]].copy()
    output["Class"] = output["Class"].astype("int8")
    output_path = SPLIT_DIR / f"{name}.csv"
    output.to_csv(output_path, index=False)

    fraud_count = int(output["Class"].sum())
    manifest["splits"][name] = {
        "file": output_path.name,
        "rows": len(output),
        "legitimate": len(output) - fraud_count,
        "fraud": fraud_count,
        "fraud_rate_percent": round(fraud_count / len(output) * 100, 4),
        "feature_groups": len(group_sets[name]),
    }

MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n")

for name, details in manifest["splits"].items():
    print(
        f"{name}: rows={details['rows']:,}, "
        f"legitimate={details['legitimate']:,}, "
        f"fraud={details['fraud']:,}, "
        f"fraud_rate={details['fraud_rate_percent']}%"
    )

print("Feature-group overlap between splits: 0")
print(f"Saved split files and manifest under: {SPLIT_DIR}")
