from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_openml

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
CACHE_DIR = ROOT / "data" / "openml_cache"
CSV_PATH = RAW_DIR / "creditcard.csv"
MANIFEST_PATH = RAW_DIR / "manifest.json"

RAW_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

print("Fetching public OpenML dataset 1597. The first download may take a few minutes.")
dataset = fetch_openml(
    data_id=1597,
    as_frame=True,
    data_home=str(CACHE_DIR),
    parser="auto",
)

frame = dataset.frame.copy()

if "Class" not in frame.columns:
    target = dataset.target
    if isinstance(target, pd.DataFrame):
        if target.shape[1] != 1:
            raise ValueError(f"Expected one target column; got {list(target.columns)}")
        target = target.iloc[:, 0]
    frame["Class"] = target

frame["Class"] = pd.to_numeric(frame["Class"], errors="raise").astype("int8")

required = [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]
missing_columns = [column for column in required if column not in frame.columns]
if missing_columns:
    raise ValueError(f"Dataset is missing expected columns: {missing_columns}")

if len(frame) != 284_807:
    raise ValueError(f"Expected 284807 rows, got {len(frame)}")

fraud_count = int((frame["Class"] == 1).sum())
legitimate_count = int((frame["Class"] == 0).sum())
if fraud_count != 492 or legitimate_count != 284_315:
    raise ValueError(
        f"Unexpected class counts: legitimate={legitimate_count}, fraud={fraud_count}"
    )

ordered_columns = (["Time"] if "Time" in frame.columns else []) + required
frame = frame[ordered_columns]
missing_values = int(frame.isna().sum().sum())
frame.to_csv(CSV_PATH, index=False)

digest = sha256()
with CSV_PATH.open("rb") as file:
    for chunk in iter(lambda: file.read(1024 * 1024), b""):
        digest.update(chunk)

manifest = {
    "dataset": "OpenML creditcard",
    "openml_data_id": 1597,
    "source": "https://www.openml.org/api/v1/json/data/1597",
    "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
    "rows": len(frame),
    "columns": list(frame.columns),
    "legitimate_count": legitimate_count,
    "fraud_count": fraud_count,
    "fraud_rate_percent": round(fraud_count / len(frame) * 100, 4),
    "missing_values": missing_values,
    "csv_sha256": digest.hexdigest(),
}
MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n")

print(f"Rows: {len(frame)}")
print(f"Legitimate: {legitimate_count}")
print(f"Fraud: {fraud_count}")
print(f"Fraud rate: {manifest['fraud_rate_percent']}%")
print(f"Missing values: {missing_values}")
print(f"Columns: {', '.join(frame.columns)}")
print(f"Saved CSV: {CSV_PATH}")
print(f"Saved source/checksum record: {MANIFEST_PATH}")
