# Final test evaluation

- Model: Logistic Regression
- Source MLflow model run: `27da3507bddb413bab9cd0a12aeb092a`
- Evaluation MLflow run: `008277161614417d8da7e13727013676`
- Git commit: `c6c314272a5ca80497e6cd5ef9ad269dbb092dd8`
- Threshold rule: highest threshold with validation recall at least 85%
- Decision threshold: `0.87292890258894895`
- Validation rows: 56,962
- Validation fraud caught: 85 of 99
- Validation alerts: 307
- Test rows: 56,962
- Test fraud cases: 99
- Test CSV SHA-256: `3794c7aa78c879c51a8e1550ce246a31386c781dd1689f2c9c78bc90f50707f0`

## Test metrics

- Average Precision: 0.733082
- ROC-AUC: 0.976206
- Precision at selected threshold: 25.5072%
- Recall at selected threshold: 88.8889%
- F1 at selected threshold: 0.396396
- Alert rate: 0.6057%
- Fraud caught: 88
- False positives: 257
- Fraud missed: 11
- True negatives: 56606
