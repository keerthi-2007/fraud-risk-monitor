# Exploratory data analysis

## Dataset overview

- Rows: 284,807
- Columns: 30
- Legitimate transactions: 284,315
- Fraudulent transactions: 492
- Fraud rate: 0.1727%
- Missing cells: 0
- Non-finite numeric values: 0
- Zero-amount transactions: 1,825
- Negative-amount transactions: 0
- Repeated feature groups: 5,149
- Rows in repeated groups: 14,293

## Amount summary by class

          count      mean       std  min    50%      95%        99%       max
Class                                                                        
0      284315.0   88.2910  250.1051  0.0  22.00  364.409  1016.9664  25691.16
1         492.0  122.2113  256.6833  0.0   9.25  640.905  1357.4279   2125.87

## Ten largest absolute correlations with Class

These are descriptive correlations for anonymized inputs.

V17   -0.3265
V14   -0.3025
V12   -0.2606
V10   -0.2169
V16   -0.1965
V3    -0.1930
V7    -0.1873
V11    0.1549
V4     0.1334
V18   -0.1115

## Charts

- `eda_class_balance.png`
- `eda_amount_by_class.png`
- `eda_feature_distributions.png`
- `eda_correlations.png`
- `eda_duplicate_groups.png`

The raw CSV was read but not modified. Charting with log1p changes only the
display scale; it does not transform or rewrite the dataset.
