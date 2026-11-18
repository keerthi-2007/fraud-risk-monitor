# Cleaning and validation report

- Input: `data/raw/creditcard.csv`
- Output: `data/processed/creditcard_prepared.csv`
- Rows before: 284,807
- Rows after: 284,807
- Rows removed: 0
- Missing values: 0
- Non-finite values: 0
- Negative Amount values: 0
- Legitimate rows: 284,315
- Fraud rows: 492
- Zero-amount legitimate rows: 1,798
- Zero-amount fraud rows: 27
- Unique feature groups: 275,663
- Groups with conflicting labels: 0

## Decisions

- The raw CSV was preserved.
- Zero-amount and high-amount rows were retained.
- Repeated rows were retained; their feature groups are recorded for later splitting.
- `feature_group_id` is metadata for splitting and must be excluded from model inputs.
- Scaling or other learned transformations will be fitted after splitting, using training data only.
- No synthetic records were created.
