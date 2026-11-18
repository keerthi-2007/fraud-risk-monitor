# Repeated feature analysis

- Total rows: 284,807
- Unique feature vectors: 275,663
- Repeated feature groups: 5,149
- Rows in repeated groups: 14,293
- Extra rows beyond one per repeated group: 9,144
- Largest repeated group: 77 rows
- Repeated groups containing both labels: 0

## Labels among rows in repeated groups

- Legitimate rows: 14,261
- Fraudulent rows: 32

The raw CSV was read but not modified. Identical feature vectors do not establish
that records refer to the same transaction, especially because this dataset
version has no Time column.
