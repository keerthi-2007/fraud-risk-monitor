# AIOps Fraud Risk Monitor
## High-Level Design Document

**Project:** AIOps Fraud Risk Monitor
**Version:** 1.1 — 19 November 2026

---

## 1. Purpose

This document describes the high-level design of the
Fraud Risk Monitor application.

It focuses on modules, responsibilities, workflows,
data movement, integration and design decisions.

The detailed API schemas are covered separately in
the Low-Level Design and API Specification.

## 2. System Objectives

The application is designed to:

1. Score individual financial transactions.
2. Process transaction CSV files.
3. Support bulk file processing through Streamlit.
4. Identify transactions requiring review.
5. Collect authorised prediction feedback.
6. Monitor data drift and application performance.
7. Display DVC pipeline status and execution history.
8. Provide observable, containerised deployment.
9. Support independently approved feedback-driven candidate retraining without automatically replacing the active model.

![System high-level design](../architecture_diagrams/01_HLD_System_Architecture.png)

## 3. Major Functional Modules

| Module | Technology | Responsibility |
|---|---|---|
| Data preparation | Python, DVC | Clean transaction data |
| Data splitting | Python, DVC | Create train, validation and test splits |
| Dataset auditing | PySpark, DVC | Check source data quality |
| Spark transformation | PySpark, DVC, Parquet | Ingest, validate and enrich raw data into partitioned Parquet |
| Exploratory analysis | Pandas, Matplotlib, DVC | Regenerate EDA charts and summary |
| Drift baseline | Python, DVC | Generate reference feature distributions |
| Model development | Scikit-learn, MLflow, DVC | Train candidates from `params.yaml`; track metrics and permutation feature importance |
| Model evaluation | MLflow | Record controlled evaluation results |
| Prediction service | FastAPI | Serve the approved model and per-prediction explanations |
| Native model serving | MLflow scoring server | Serve the same approved model at `/invocations` |
| User interface | Streamlit | Accept inputs and display results |
| Feedback service | FastAPI, JSONL, Fernet | Store corrected labels with features encrypted at rest |
| Reviewer approval | Host-side Python CLI, JSONL | Record independent approval and label integrity |
| Candidate retraining worker | Python, MLflow | Check eligible feedback and train candidate when thresholds are met |
| Candidate evaluation and promotion | Python, MLflow | Apply validation gates and preserve approved model unless deployment explicitly enabled |
| Retraining status endpoint | FastAPI, JSON | Serve token-protected aggregate snapshot |
| Pipeline monitoring | DVC, JSON snapshot | Display pipeline state and recorded runs |
| Observability | Prometheus, Grafana | Record and visualise metrics |
| Alert processing | Alertmanager, Python | Route and record notifications |
| Containerisation | Docker Compose | Deploy application services |

## 4. Offline Data Processing Workflow

The offline workflow begins with the historical
credit-card transaction dataset.

Data preparation creates the processed dataset.

The splitting stage generates separate training,
validation and test datasets.

The training split is also used to generate the
drift-monitoring baseline.

A separate Spark stage audits the original dataset.

A Spark transformation stage ingests the raw CSV with an
explicit schema, validates it, adds engineered columns
(`log_amount`, `is_zero_amount`, `amount_zscore`,
`feature_hash`, `duplicate_count`) and writes Parquet
partitioned by class. It processed 284,807 rows in about
14 s (≈20,400 rows/second).

An EDA stage regenerates the exploratory charts and summary.

DVC tracks dependencies and outputs for these stages. The raw
dataset is versioned by `data/raw.dvc`.

The pipeline has seven DVC stages: `prepare_data`,
`create_splits`, `spark_audit`, `spark_transform`,
`build_drift_baseline`, `eda` and `train_baselines`.

## 5. Model Development Workflow

Training is the `train_baselines` DVC stage; final test-set
evaluation remains a controlled, manual MLflow operation.

The training workflow:

1. Loads prepared training and validation data.
2. Reads hyperparameters from the DVC-tracked `params.yaml`.
3. Trains candidate classification models.
4. Calculates validation metrics.
5. Computes permutation feature importance on a stratified validation sample.
6. Logs parameters, Git commit, split and params hashes, metrics,
   feature importance and artifacts to MLflow.
7. Writes `metrics/train_metrics.json` for `dvc metrics show`.

Changing a value in `params.yaml` invalidates the training stage,
so every experiment maps to a Git commit plus an MLflow run ID.

The final evaluation workflow records results
against the selected model.

The approved Logistic Regression model is registered
as FraudRiskClassifier version 1.

Routine prediction requests do not initiate training.

## 6. Single-Transaction Prediction Workflow

1. A user submits transaction features through Streamlit.
2. Streamlit calls the FastAPI prediction endpoint.
3. FastAPI validates the 29 expected features.
4. The approved MLflow model calculates a score.
5. FastAPI compares the score against the approved threshold.
6. FastAPI returns a normal or review decision.
7. Streamlit displays the prediction result.

### Prediction explanation

On request, Streamlit calls `POST /explain` for a chosen row.
FastAPI rescores the transaction 29 times in one batch, each time
resetting one feature to its training median, and returns each
feature's contribution to the score. Streamlit shows the top
factors as a bar chart in plain language.

The score is uncalibrated and must not be described
as a verified probability of fraud.

## 7. Batch Prediction Workflow

1. The user supplies a CSV file.
2. The application validates the upload.
3. FastAPI verifies the expected columns and values.
4. The approved model scores the transaction rows.
5. The API produces scored results.
6. Streamlit presents the results to the user.
7. Eligible uploads also trigger drift analysis.

Backend CSV limits:

- Maximum file size: 50 MiB
- Maximum number of rows: 100,000
- Required features: V1 through V28 and Amount

Bulk ZIP processing is coordinated by the frontend.
The backend provides CSV scoring rather than a
separate ZIP prediction endpoint.

## 8. Feedback and Candidate Retraining Workflow

![Feedback lifecycle](../architecture_diagrams/05_Feedback_Sequence.png)

1. A reviewer supplies a confirmed outcome through Streamlit, which submits it to the token-protected FastAPI feedback endpoint.
2. FastAPI validates and stores the feedback in JSONL, encrypting the transaction features with the configured Fernet key. Submission **does not** initiate training immediately.
3. An authorised reviewer independently verifies and approves the outcome using the local approval utility.
4. A separate host-side watcher filters unapproved, invalid, duplicate and data-overlapping feedback.
5. When there are **at least 20 eligible** new records including **at least 3 fraud and 3 legitimate**, the watcher can train a candidate with the original training split.
6. Validation-based quality checks and MLflow candidate evaluation run separately from inference; the held-out test set is excluded from candidate-selection decisions.
7. Candidate promotion eligibility is checked. **Automatic deployment is disabled by default** and requires explicit opt-in.
8. The worker publishes aggregate progress to a JSON snapshot read through `GET /monitoring/retraining-status` and shown in Streamlit.

These operations are separate from the DVC DAG. An insufficient or unverified feedback sample is never presented as reliable production-model performance.

## 9. Pipeline Monitoring Workflow

The DVC status generator reads declared pipeline
stages and dependency information.

The stage execution runner records actual run
outcomes and elapsed duration.

For successful Spark audits, it can calculate
rows processed per second.

The generator saves this information in
monitoring/pipeline_status.json.

FastAPI reads the saved snapshot through a
token-protected, read-only endpoint.

Streamlit displays stage status, Spark audit
information and recorded execution history.

The snapshot may become stale between updates.

## 10. Monitoring and Alert Workflow

Prometheus collects application and container metrics
at a configured 15-second scrape interval.

The system monitors:

- API request counts and latency
- Scored transaction counts
- CSV validation failures
- Feature distribution drift
- Recorded feedback indicators
- Container resource metrics

Grafana provides dashboards for the collected metrics.

Prometheus evaluates alert rules every 15 seconds.

Alerts are forwarded to Alertmanager, which routes
notifications to the local webhook receiver.

The webhook stores received notifications in JSONL.

## 11. Component Communication

| Source | Destination | Communication |
|---|---|---|
| Browser | Streamlit | HTTP |
| Streamlit | FastAPI | HTTP REST |
| FastAPI | MLflow storage | Local database and mounted artifacts |
| FastAPI | Feedback storage | Local mounted JSONL (features encrypted) |
| Streamlit | FastAPI `/explain` | HTTP REST |
| Client / evaluator | MLflow scoring server | HTTP REST `/invocations` on port 5001 |
| Host retraining worker | Feedback and approval JSONL | Local file read + validation |
| Worker | MLflow | Candidate training, evaluation and gated promotion |
| Worker | Retraining snapshot | Local JSON file write |
| Streamlit | Retraining-status endpoint | Token-authenticated HTTP GET |
| FastAPI | Pipeline snapshot | Read-only mounted JSON |
| Prometheus | FastAPI | HTTP metrics scraping |
| Prometheus | cAdvisor | HTTP metrics scraping |
| Grafana | Prometheus | HTTP queries |
| Prometheus | Alertmanager | Alert notifications |
| Alertmanager | Webhook receiver | HTTP POST |

Offline scripts run outside the inference containers.

## 12. Design Decisions

### Separate development and inference

Training and final evaluation are not automatically
executed by the deployed prediction service.

### Approved model enforcement

The backend loads the configured registered
model and approved decision threshold.

### Snapshot-based pipeline monitoring

The backend reads a generated status file instead
of executing DVC commands from API requests.

### Independent drift monitoring

Drift analysis is best-effort and does not block
valid prediction requests when monitoring fails.

### Local persistent feedback and gated retraining

Feedback is recorded separately from MLflow model-training artifacts. Independent label confirmation, explicit approval, minimum sample and class counts, duplicate/leakage checks and validation quality gates are required before a candidate is eligible. The watcher does not replace the live model by default.

### Encryption of stored feedback

Feedback contains real transaction feature vectors, so they are
encrypted at rest with Fernet (AES-128-CBC with HMAC-SHA256)
before being written. The key lives only in the git-ignored
`.env`; the backend refuses to start without it, and tampered or
wrongly-keyed records are rejected when read for retraining.

### Model-agnostic explanations

Single-feature occlusion against training medians was chosen
over SHAP-style libraries because it works for any scikit-learn
model, needs no extra dependency and costs one batched
prediction call.

### Two serving paths

FastAPI is the UI-facing API (validation, explanations, metrics,
feedback). MLflow's native scoring server exposes the same
approved model for standard MLflow clients. It loads the model by
its source run URI because the artifact store is mounted
read-only.

### Separate monitoring services

Prometheus, Grafana, Alertmanager and cAdvisor
operate independently of the inference service.

## 13. Security and Failure Handling

The application includes:

- Encryption at rest (Fernet) for stored feedback features
- Encryption key held only in the git-ignored `.env`
- Separate feedback and monitoring tokens
- Rejection of invalid prediction inputs
- CSV upload size and row limits
- Non-root backend and frontend containers
- Read-only artifact and snapshot mounts
- Backend health checks
- Error responses for unavailable snapshots
- Best-effort drift monitoring
- Independent approval and minimum-feedback gates for retraining
- Read-only, token-protected retraining-status API
- Opt-in deployment with health verification and rollback safeguards

The current deployment is intended for a
local academic demonstration.

## 14. Quality Attributes

| Attribute | Design approach |
|---|---|
| Reproducibility | DVC, Git, MLflow |
| Traceability | Model run IDs and execution records |
| Maintainability | Separate modules and services |
| Testability | Automated unit and integration tests |
| Observability | Prometheus, Grafana and alerts |
| Reliability | Input validation and health checks |
| Security | Encryption at rest, tokens and restricted service bindings |
| Explainability | Per-prediction feature contributions and tracked feature importance |
| Code quality | Ruff (PEP 8) enforced in CI |

## 15. Verified Evidence

The implementation has demonstrated:

- 160 automated tests in the suite; latest run passed all 154 that do not need the private dataset/model
- 147 passing tests in the last recorded complete VM suite
- Five passing targeted deployment tests following the clean-CI fixture fix
- Successful hosted GitHub Actions CI for commit `96b8113`
- Seven DVC stages, including Spark transformation, EDA and training
- Successful Spark audit of 284,807 records
- Measured audit throughput of 18,495.64 rows/second
- Spark transformation of 284,807 rows in 13.95 s (20,411.6 rows/second)
- MLflow scoring server healthy and returning predictions on port 5001
- Registered MLflow model version 1
- Eight Docker services, including the MLflow scoring server
- Healthy FastAPI prediction service
- Successful Alertmanager webhook delivery

GitHub Actions has run successfully on the hosted runner. Its portable suite excludes local data- and MLflow-artifact-dependent tests.

## 16. Conclusion

The high-level design separates data engineering,
model development, inference and observability.

These modules work together to provide fraud
risk scoring with reproducible processing,
traceable model deployment and operational monitoring.
