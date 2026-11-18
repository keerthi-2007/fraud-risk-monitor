# Fraud Risk Monitor
## Non-Technical User Manual

**Version:** 1.0
**Date:** 14 November 2026
**Audience:** Application users, reviewers and demonstrators

---

## 1. About the Application

Fraud Risk Monitor helps users identify financial
transactions that may require further investigation.

Users can upload transaction files, view risk scores,
review flagged transactions and download results.

The application also provides feedback submission
and pipeline monitoring features.

The application does not automatically confirm fraud
or make financial decisions.

## 2. Opening the Application

The application must already be running.

1. Open a web browser.
2. Visit http://localhost:8501.
3. Look for the Fraud Risk Monitor interface.
4. Check the prediction API connection indicator.

If the page says the prediction API is offline,
contact the application administrator.

The user interface contains three main tabs:

- Single CSV
- Bulk ZIP upload
- Pipeline Overview

## 3. Preparing a Transaction CSV

A transaction CSV must contain these columns:

- V1, V2, V3 and so on through V28
- Amount

Each row represents one transaction.

The file must contain numeric feature values.
The Amount must not be negative.

Do not include extra columns such as Class,
customer name or transaction ID in the prediction CSV.

The backend accepts:

- Up to 100,000 transaction rows per CSV
- A maximum CSV file size of 50 MiB

The application may apply additional upload limits
before sending a file to the backend.

## 4. Scoring a Single CSV

1. Select the Single CSV tab.
2. Locate the section titled Score one CSV.
3. Click Choose a transaction CSV.
4. Select a valid CSV file.
5. Click Score this CSV.
6. Review the displayed results.

The application displays:

- Rows processed
- Transactions marked for review
- Transactions marked normal
- A prediction preview

A download option is available for scored results.

## 5. Understanding Prediction Results

The prediction results contain:

| Result | Meaning |
|---|---|
| risk_score | Model-generated risk score |
| decision_threshold | Threshold used for classification |
| decision | normal or review |
| model_run_id | Identifier of the deployed model |

### Normal

A normal decision means the score is below
the configured review threshold.

It does not guarantee that a transaction
is legitimate.

### Review

A review decision means the score is at or
above the configured threshold.

It indicates that further investigation
may be appropriate.

It does not prove that fraud occurred.

### Risk scores

Scores are not calibrated probabilities.

For example, a score of 0.90 should not
be described as a verified 90% probability
of fraud.

## 6. Scoring Multiple CSV Files

1. Select the Bulk ZIP upload tab.
2. Prepare a ZIP archive containing valid CSV files.
3. Click Choose a ZIP archive.
4. Select the ZIP file.
5. Click Score each file separately.
6. Review the Per-file results section.

The application processes the eligible CSV
files separately.

Use the Review each file section to inspect
individual file results.

If a file is rejected, read the displayed
validation message and correct the input.

The bulk-upload interface has its own file
count and archive limits.

## 7. Saving Prediction Results

After scoring, use the application's
available download controls to save results.

Keep the downloaded files in a secure location.

Prediction results may contain transaction
feature values and should be handled carefully.

The result includes the model run ID, which
helps identify the model used for scoring.

## 8. Submitting Prediction Feedback

Feedback helps record the confirmed outcome of
a transaction after review.

1. Score a transaction CSV.
2. Open the relevant prediction results.
3. Locate the feedback section.
4. Select the transaction to review.
5. Choose Confirmed transaction outcome.
6. Select legitimate or fraud.
7. Click Save feedback.
8. Check the confirmation message.

Only submit feedback when the transaction outcome
has been independently confirmed.

Feedback is stored locally for monitoring.

Submitting feedback does not immediately retrain the model
or change future predictions.

### How feedback is used for model improvement

A background service checks for newly approved feedback.

Before feedback can be used for retraining:

1. The transaction outcome must be independently confirmed.
2. An authorised reviewer must approve the feedback.
3. The feedback must pass duplicate and data validation checks.
4. At least 20 eligible records must be available, including
   at least 3 fraud and 3 legitimate transactions.

When these conditions are met, the system can automatically
train a candidate model and evaluate it using validation data.

Only candidates passing the required quality checks are
eligible for promotion.

Automatic deployment is disabled by default. Therefore,
submitting or approving feedback does not immediately replace
the model used for predictions.

The final held-out test dataset is not used for automatic
candidate retraining or promotion decisions.

## 9. Understanding the Pipeline Overview

Select the Pipeline Overview tab.

The page contains information about the
data-processing pipeline and its recent executions.

### DVC pipeline stages

The dashboard displays four stages:

- prepare_data
- create_splits
- spark_audit
- build_drift_baseline

An up-to-date status means the recorded DVC
dependencies and outputs are consistent.

It does not mean a stage is currently running.

### Spark data audit

The Spark audit section displays information
about the original transaction dataset.

The verified audit processed 284,807 records,
including 492 fraud records.

### Recorded pipeline executions

This section displays recorded stage executions.

Available information includes:

- Stage name
- Execution status
- Start time
- Duration
- Number of rows processed
- Measured throughput

A successful recorded Spark audit took
approximately 15.40 seconds and processed
18,495.64 rows per second.

This result belongs to one observed execution.
It is not a guaranteed future processing speed.

### Snapshot freshness

Pipeline information comes from a saved snapshot.

It is not a real-time job execution monitor.

If the snapshot is old, request an updated
pipeline-status snapshot from the administrator.

## 10. Understanding Monitoring Information

The monitoring system collects information
about application behaviour and performance.

It includes:

- Request counts and response duration
- Number of scored transactions
- Transactions requiring review
- Input validation failures
- Feature distribution drift
- Available feedback indicators
- Container resource usage

Grafana dashboards display operational metrics.

Some metrics may reset when the application
container restarts.

Monitoring alerts are intended to help
administrators investigate possible issues.

An alert does not automatically mean
the prediction model is incorrect.

## 11. Common Problems and Solutions

| Problem | Possible cause | Recommended action |
|---|---|---|
| Application page does not open | Frontend unavailable | Contact administrator |
| Prediction API offline | Backend unavailable | Contact administrator |
| CSV upload rejected | Incorrect file type | Upload a CSV |
| Missing columns | Required features absent | Include V1 through V28 and Amount |
| Unexpected columns | Additional columns included | Remove unsupported columns |
| Invalid feature values | Text, blanks or non-finite values | Correct the input data |
| Negative Amount | Invalid transaction amount | Use a valid non-negative amount |
| CSV too large | File or row limit exceeded | Split the input into smaller files |
| ZIP upload rejected | Invalid archive or upload limit | Check archive contents and limits |
| Feedback rejected | Invalid form or access configuration | Check details or contact administrator |
| Pipeline status unavailable | Snapshot missing or inaccessible | Contact administrator |
| Pipeline information outdated | Snapshot not recently regenerated | Request a refreshed snapshot |

Do not repeatedly submit the same transaction
solely to obtain a different prediction.

## 12. Responsible Use

The application supports investigation,
not automatic financial decision-making.

Important precautions:

- Do not treat a review decision as proven fraud.
- Do not treat a normal decision as a fraud guarantee.
- Do not interpret model scores as calibrated probabilities.
- Use verified outcomes when submitting feedback.
- Keep transaction data and downloaded results secure.
- Consult an authorised reviewer before taking action.
- Report unexpected application behaviour to the administrator.

The application is an academic demonstration
using historical transaction data.

## 13. Demonstration Walkthrough

A simple classroom demonstration can follow
this sequence:

1. Open http://localhost:8501.
2. Confirm the prediction API is connected.
3. Open the Single CSV tab.
4. Upload a valid transaction CSV.
5. Click Score this CSV.
6. Show rows processed and review counts.
7. Explain the prediction preview.
8. Download the scored results if required.
9. Demonstrate the feedback form using a
   confirmed outcome, if appropriate.
10. Open the Pipeline Overview tab.
11. Show the four DVC stages.
12. Show Spark audit information.
13. Show the recorded execution duration
    and measured throughput.
14. Show the automatic retraining dashboard and explain that
    approved, eligible feedback can trigger candidate training.
    Automatic model deployment remains disabled by default.

Use prepared, authorised demonstration data.
Avoid submitting invented labels as genuine feedback.

## 14. Frequently Asked Questions

### Does review mean fraud?

No. It means the model score is high enough
to recommend further investigation.

### Can the application process multiple transactions?

Yes. Upload a CSV containing multiple transaction
rows or use the Bulk ZIP upload tab.

### Can I upload a spreadsheet?

The prediction interface expects CSV files,
not Excel workbooks.

### Can I change the model?

No. Ordinary users cannot change the approved
model through the Streamlit interface.

### Does feedback immediately improve the model?

No. Feedback is recorded for monitoring.
It does not trigger automatic retraining.

### Is the pipeline monitor live?

No. It displays a saved pipeline snapshot
and previously recorded executions.

### Can the risk score be interpreted as a probability?

No. The deployed model score is uncalibrated.

## 15. Conclusion

Fraud Risk Monitor provides a straightforward way
to upload transaction data, review risk scores,
download results and submit confirmed feedback.

The Pipeline Overview also helps users understand
the recorded data-processing status and execution
measurements.

Users should interpret results cautiously and
seek human review for suspicious transactions.
