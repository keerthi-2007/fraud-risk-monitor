# Fraud Risk Monitor — Fraud Detection and Monitoring System

An end-to-end machine learning application for detecting potentially fraudulent credit card transactions, supported by data engineering, experiment tracking, model deployment, automated testing, monitoring and alerting.

The project demonstrates how **Machine Learning and AIOps practices** can be combined to develop an observable and maintainable fraud risk monitoring system.

The application runs locally on an Ubuntu virtual machine using Docker Compose.

## 1. Project Overview

Fraud Risk Monitor uses a machine learning model to analyse transaction features and assign risk scores.

Transactions exceeding the configured decision threshold are flagged for further review.

In addition to fraud prediction, the application implements:

- Data processing and validation
- Machine learning experiment tracking
- Model registration and versioning
- CSV and bulk ZIP transaction scoring
- Human feedback collection
- DVC pipeline monitoring
- Spark data auditing, Spark transformation to Parquet and throughput measurement
- Per-prediction explanations (`/explain`) and tracked feature importance
- Encryption at rest for stored feedback (Fernet)
- Native MLflow model serving alongside FastAPI
- Feature drift monitoring using PSI
- Prometheus metrics and alerting
- Grafana operational dashboards
- Docker containerisation
- Automated tests and GitHub Actions CI

**Note:** This is an educational fraud risk monitoring application, not a production banking system. Model scores are uncalibrated and should not be interpreted as confirmed fraud probabilities.

## 2. Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Frontend | Streamlit |
| Backend API | FastAPI, Pydantic |
| Machine Learning | Scikit-learn |
| Experiment Tracking | MLflow |
| Data Pipeline Management | DVC |
| Data Engineering | Apache Spark / PySpark |
| Containerisation | Docker, Docker Compose |
| Application Monitoring | Prometheus |
| Monitoring Dashboards | Grafana |
| Container Monitoring | cAdvisor |
| Alert Management | Prometheus Alert Rules, Alertmanager |
| Notifications | Local webhook receiver |
| Testing | Pytest |
| Continuous Integration | GitHub Actions |
| Version Control | Git, GitHub |

## 3. Application URLs

All URLs below refer to services running locally on the Ubuntu VM.

They are **not publicly hosted URLs**. The corresponding services must be running before the pages can be accessed.

### Application and Monitoring Services

| Service | URL | Description |
|---|---|---|
| Streamlit Application | http://127.0.0.1:8501 | Main fraud risk monitoring application |
| FastAPI Documentation | http://127.0.0.1:8000/docs | Interactive REST API documentation |
| FastAPI Health | http://127.0.0.1:8000/health | Backend health and model run information |
| FastAPI Metrics | http://127.0.0.1:8000/metrics | Prometheus-formatted application metrics |
| MLflow | http://127.0.0.1:5000 | ML experiments and Model Registry |
| Prometheus | http://127.0.0.1:9091 | Metrics monitoring |
| Prometheus Targets | http://127.0.0.1:9091/targets | Monitored service availability |
| Prometheus Alerts | http://127.0.0.1:9091/alerts | Alert rules and firing alerts |
| Prometheus Query | http://127.0.0.1:9091/query | PromQL queries |
| Grafana | http://127.0.0.1:3001 | Operational dashboards |
| Alertmanager | http://127.0.0.1:9093 | Alert management |

### Grafana Dashboards

| Dashboard | URL |
|---|---|
| Fraud Operations | http://127.0.0.1:3001/d/fraud-operations/fraud-operations |
| Container Health | http://127.0.0.1:3001/d/fraud-container-health/container-health |
| Fraud Risk Monitor | http://127.0.0.1:3001 — select the Fraud Risk Monitor dashboard |

The project also uses internal Docker network services:

- `cadvisor:8080` — container resource metrics
- `alert-webhook:8085` — alert notification receiver

These internal services are not separately exposed as public browser interfaces.

## 4. Dataset

The project uses the **Credit Card Fraud Detection** dataset, containing anonymised credit card transactions.

### Dataset Source

The project uses the real Credit Card Fraud Detection dataset developed by the ULB Machine Learning Group.

**Primary source (used in this project):**
https://www.openml.org/d/1597

**Alternative source (Kaggle):**
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

The dataset was downloaded using Scikit-learn's `fetch_openml` function through `scripts/fetch_dataset.py`.

To download the dataset in a new environment:

```bash
python -m scripts.fetch_dataset
```

The dataset is stored locally at `data/raw/creditcard.csv`.

It contains 284,807 transactions, including 492 fraudulent cases. The prediction model uses the anonymised features V1–V28 and Amount. The Class column contains the fraud labels.

### Dataset Information

| Property | Value |
|---|---|
| Total transactions | 284,807 |
| Fraudulent transactions | 492 |
| Training records | 170,883 |
| Validation records | 56,962 |
| Test records | 56,962 |
| Model input features | V1–V28 and Amount |
| Target variable | Class |

The model uses 29 numerical input features. The `Class` column is used as the ground-truth label during training and evaluation, not as a prediction input.

### Dataset Location

The original dataset is not stored in the GitHub repository.

Download it and place it at:

```text
data/raw/creditcard.csv
```

The DVC pipeline uses the raw dataset to prepare and create the train, validation and test datasets.

The dataset is highly imbalanced, making metrics such as average precision and recall especially relevant.

## 5. Machine Learning Model

Two model families were evaluated:

- Logistic Regression
- HistGradientBoosting

The selected model is a **Logistic Regression pipeline** using StandardScaler and balanced class weights.

### Selected Model Information

| Property | Value |
|---|---|
| Registered Model | `FraudRiskClassifier` |
| Model Version | 1 |
| Algorithm | Logistic Regression |
| Number of Features | 29 |
| Scaler | StandardScaler |
| Class Weight | Balanced |
| Maximum Iterations | 2000 |
| Deployment Decision Threshold | 0.87292890258894895 |

### Recorded Evaluation Results

| Metric | Result |
|---|---|
| Validation Recall | 85.859% |
| Fraud Cases Detected in Validation | 85 of 99 |
| Validation Review Alerts | 307 |
| Test Average Precision | 0.733082 |
| Test ROC-AUC | 0.976206 |

The threshold was selected using validation data and subsequently used for final evaluation.

Transactions with scores at or above the decision threshold are classified as `review`. Other transactions are classified as `normal`.

A `review` decision means that further investigation is recommended. It does not indicate confirmed fraud.

## 6. Project Setup and Installation

### Prerequisites

The application was developed and verified in an Ubuntu VM with a Python 3.14 virtual environment.

The main requirements are:

- Ubuntu or another compatible Linux environment
- Python and `venv`
- pip
- Git
- Docker Engine
- Docker Compose plugin
- Project dependencies
- Credit card fraud dataset
- Approved MLflow model artifacts for inference

Internet access may be required to download dependencies, Docker images and the original dataset.

### Clone the Repository

```bash
git clone https://github.com/keerthi-2007/fraud-risk-monitor.git
cd fraud-risk-monitor
```

If the GitHub repository is private, access is required before cloning.

### Create and Activate a Python Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Confirm the Python version:

```bash
python --version
```

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
```

Additional service-specific dependency specifications are available in:

```text
requirements-api.txt
requirements-frontend.txt
```

### Important: Fresh Clone Requirements

The repository includes source code, configuration and documentation, but does not distribute the original raw dataset or local MLflow database and model artifacts.

A fresh clone therefore needs additional setup before the prediction backend can load the approved model.

The approved model must be reproduced through the documented training and evaluation workflow, or its verified artifacts must be provisioned separately.

The instructions in the next section describe starting an **already configured project environment**, such as the Ubuntu VM used for development and testing.

## 7. Running the Application on Ubuntu

### Step 1 — Open the Project

Open an Ubuntu terminal:

```bash
cd ~/Documents/fraud-risk-monitor
source .venv/bin/activate
```

### Step 2 — Start Docker Services

Start the existing containers:

```bash
docker compose up -d
```

When initial image building is required, use:

```bash
docker compose up -d --build
```

The application uses eight Docker services:

1. `backend`
2. `frontend`
3. `mlflow-serve` (MLflow scoring server, http://127.0.0.1:5001)
4. `prometheus`
5. `grafana`
6. `alertmanager`
7. `cadvisor`
8. `alert-webhook`

`.env` must contain `MONITORING_TOKEN`, `FEEDBACK_TOKEN` and `FEEDBACK_ENCRYPTION_KEY` (see *Security: Encryption at Rest*).

### Step 3 — Verify Container Status

```bash
docker compose ps
```

Check that the services are running and that the backend health check reports healthy.

If an individual service stopped after a VM restart, it can be started separately.

For example:

```bash
docker compose start prometheus grafana
```

### Step 4 — Open Streamlit

Open:

http://127.0.0.1:8501

The frontend provides the fraud prediction and pipeline monitoring interface.

### Step 5 — Open FastAPI

Interactive API documentation:

http://127.0.0.1:8000/docs

API health endpoint:

http://127.0.0.1:8000/health

### Step 6 — Open Prometheus

Open:

http://127.0.0.1:9091

To check monitored services:

http://127.0.0.1:9091/targets

The configured targets include the Fraud Risk API, cAdvisor, Grafana and Prometheus.

### Step 7 — Open Grafana

Open:

http://127.0.0.1:3001

Navigate to the Fraud Risk Monitor dashboard folder to access the operational dashboards.

### Step 8 — Open Alertmanager

Open:

http://127.0.0.1:9093

Alertmanager receives configured Prometheus alerts and routes them to the local webhook receiver.

## 8. Running MLflow

MLflow is used to track machine learning experiments, model parameters, metrics, artifacts and registered models.

The local project uses a SQLite-backed MLflow tracking store.

### Start MLflow UI

Open a separate Ubuntu terminal:

```bash
cd ~/Documents/fraud-risk-monitor
source .venv/bin/activate

mlflow ui \
  --backend-store-uri sqlite:///mlflow.db \
  --host 127.0.0.1 \
  --port 5000
```

Keep the terminal open while using MLflow.

Then open:

http://127.0.0.1:5000

### MLflow Project Information

| Property | Value |
|---|---|
| Experiment Name | `fraud-detection-baselines` |
| Registered Model | `FraudRiskClassifier` |
| Registered Version | 1 |
| Selected Model Run ID | `27da3507bddb413bab9cd0a12aeb092a` |

The selected run records model configuration, dataset split identifiers, Git commit information, metrics and model artifacts.

The experiment also contains alternative model runs and final evaluation records.

`MLproject` declares a pinned `python_env.yaml` environment and entry points for every lifecycle step (`main` = `dvc repro`, `fetch_data`, `prepare_data`, `create_splits`, `spark_audit`, `eda`, `train`, `evaluate`, `test`), e.g. `mlflow run . -e train`.

Each training run additionally logs `params.yaml`, its SHA-256, permutation feature importance (`feature_importance.json` and one `perm_importance_<feature>` metric per feature) and a `top_feature` tag.

The registered model is also served natively by MLflow from a dedicated image (`mlflow_serving/Dockerfile`, compose service `mlflow-serve`): `POST http://127.0.0.1:5001/invocations`, health at `GET /ping`.

The MLflow UI command reads an existing tracking database. It does not create the original experiment results or retrain the approved model.

## 9. Streamlit Application Features

### Single CSV Scoring

1. Open Streamlit.
2. Select **Single CSV**.
3. Upload a transaction CSV.
4. Click **Score this CSV**.
5. View the prediction summary.
6. Inspect the transaction risk scores.
7. Download the prediction results.
8. Optionally open **Why was a row scored this way?**, choose a row and click **Explain this row** to see which values raised or lowered its risk score (backed by `POST /explain`).

The application displays the number of processed transactions and the numbers classified as `review` and `normal`.

### Bulk ZIP Upload

1. Open the **Bulk ZIP upload** tab.
2. Upload a ZIP archive containing transaction CSV files.
3. Click **Score each file separately**.
4. Review the results for each input file.
5. Download the processed outputs.

Each CSV is processed separately.

### Feedback Collection

The application allows users to record a confirmed transaction outcome after reviewing a prediction.

The stored feedback may indicate whether the transaction was:

- `legitimate`
- `fraud`

Feedback is recorded through the backend and retained for monitoring and analysis.

Feedback submission alone does not retrain or replace the model. A separate background worker can train a candidate **only after** independent confirmation, reviewer approval, duplicate/leakage screening and minimum feedback eligibility are satisfied. Candidate evaluation and promotion checks are separate from the prediction API; automatic deployment is disabled by default.

### Retraining status in Pipeline Overview

The dashboard reads the token-protected `GET /monitoring/retraining-status` endpoint and displays the latest saved background-worker snapshot, including feedback received, awaiting approval, eligible fraud/legitimate labels and threshold progress. It is **snapshot-based**; opening the page does not train a model. The background worker (`python -m scripts.retraining_cycle --watch`) runs on the host separately from the eight Docker Compose services and the DVC pipeline.

### Pipeline Overview

The Pipeline Overview tab displays:

- Recorded DVC pipeline status
- Stage information
- Spark data audit results
- Pipeline execution history
- Recorded stage duration
- Throughput information
- Model workflow information

This provides operational visibility without requiring users to run pipeline commands manually.

## 10. FastAPI Backend

FastAPI handles transaction validation, model inference, batch scoring, monitoring and feedback storage.

**API documentation:**

http://127.0.0.1:8000/docs

### Implemented Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Health check and model run information |
| POST | `/predict` | Single transaction prediction |
| POST | `/predict/csv` | Batch CSV scoring |
| POST | `/explain` | Single transaction prediction with per-feature contributions |
| POST | `/feedback` | Store corrected transaction outcome |
| GET | `/metrics` | Expose Prometheus application metrics |
| GET | `/monitoring/pipeline-status` | Read recorded pipeline status |
| GET | `/monitoring/retraining-status` | Read authenticated retraining eligibility snapshot |
| POST | `/monitoring/validation-failure` | Record frontend validation failures |

Single-transaction requests require 29 numerical input features.

Batch CSV uploads are validated for the expected schema, valid values, file size and row limits.

The CSV prediction API accepts files up to 50 MiB and 100,000 transaction rows per request.

Selected monitoring and feedback endpoints are protected using configured request tokens.

## 11. DVC Pipeline

DVC manages the data-processing stages and their dependencies.

### Implemented DVC Stages

| Stage | Purpose |
|---|---|
| `prepare_data` | Prepare the original dataset |
| `create_splits` | Generate training, validation and test splits |
| `spark_audit` | Independently audit raw data using Spark |
| `build_drift_baseline` | Build reference statistics for drift monitoring |
| `spark_transform` | **Spark** ingestion/transformation: typed schema, validation, engineered columns (`log_amount`, `is_zero_amount`, `amount_zscore`, `feature_hash`, `duplicate_count`) written as Parquet partitioned by `Class`; throughput in `metrics/spark_transform.json` |
| `eda` | Regenerate exploratory data analysis charts and summary |
| `train_baselines` | Train candidate models with `params.yaml` hyperparameters, log runs + permutation feature importance to MLflow, write `metrics/train_metrics.json` |

The DAG root is `data/raw.dvc`, which versions the raw dataset and its download manifest.
The Spark audit is an independent stage using the raw dataset.

Training is part of the DAG, so changing a hyperparameter in `params.yaml` invalidates `train_baselines`:

```bash
dvc repro train_baselines
dvc metrics show
dvc params diff
```

Final test-set evaluation (`scripts/evaluate_final.py`) and model promotion stay manual, gated steps so the test split is never used for model selection.

### Inspect the DVC Pipeline

From the project directory:

```bash
source .venv/bin/activate

dvc status
```

Display the pipeline DAG:

```bash
dvc dag
```

Inspect dependency and output information:

```bash
cat dvc.yaml
```

To reproduce the existing DVC processing stages intentionally, after ensuring the required data and dependencies are available:

```bash
dvc repro
```

**Note:** Running `dvc repro` may execute processing stages when dependencies or outputs have changed. It is not necessary merely to view pipeline status.

### Spark Audit

The project includes an Apache Spark data audit.

A recorded successful Spark audit processed **284,807 rows**.

| Measurement | Recorded Result |
|---|---|
| Processed Rows | 284,807 |
| Recorded Execution Duration | Approximately 15.40 seconds |
| Measured End-to-End Throughput | Approximately 18,496 rows/second |

This throughput measurement includes stage execution overhead and Spark startup, not solely Spark's internal data-processing time.

### Spark Transformation

`dvc repro spark_transform` ingested, validated and enriched all **284,807 rows** in **13.95 s (≈20,412 rows/second)**, writing Parquet partitioned by `Class` to `data/processed/spark_features/`. Results are in `metrics/spark_transform.json` (`dvc metrics show`).

The recorded execution history and pipeline status snapshot are available through the application's monitoring features.

## 12. Data Drift Monitoring

The system uses **Population Stability Index (PSI)** to measure differences between incoming transaction feature distributions and the reference training distribution.

When a valid CSV contains at least 100 rows, the backend attempts drift analysis.

Drift monitoring is a best-effort operation and does not block successful fraud scoring if monitoring encounters an error.

### High Drift Alert

The configured Prometheus rule `FraudFeatureDriftHigh` evaluates feature-level PSI.

The configured alert threshold is:

```text
PSI > 0.20
```

The rule also checks that the drift analysis is recent.

A high PSI indicates that a feature distribution has shifted. It does not independently establish that predictions are incorrect.

## 13. Prometheus Monitoring

Prometheus collects operational metrics from the configured scrape targets.

### Prometheus URLs

**Main UI:** http://127.0.0.1:9091

**Targets:** http://127.0.0.1:9091/targets

**Alerts:** http://127.0.0.1:9091/alerts

**Query:** http://127.0.0.1:9091/query

### Monitored Metrics

The application exposes metrics relating to:

- API requests
- Request duration
- Transactions scored
- Review and normal decisions
- Input validation failures
- Feature drift
- Explanations served
- Unhandled server errors
- Feedback records
- Container activity and resource usage

### Alert Rules

Five Prometheus alert rules are configured:

- `FraudAPIDown`
- `FraudHighServerErrors`
- `FraudHighAPIErrorRate`
- `FraudFeatureDriftHigh`
- `FraudDriftMonitoringFailure`

Prometheus evaluates these conditions and routes alerts to Alertmanager.

Alertmanager is configured to deliver notifications to a local webhook receiver.

## 14. Grafana Dashboards

**Grafana UI:**

http://127.0.0.1:3001

The project includes three main dashboards.

### Fraud Operations

Displays fraud scoring and operational metrics, including:

- Transactions processed
- Transactions flagged for review
- Normal transactions
- Review rate
- Saved feedback
- Feedback-derived indicators
- High-drift features
- Drift analysis counts
- Drift monitoring failures
- Per-feature PSI bars coloured by the stable / watch / drift bands
- Reported outcomes (TP / FP / TN / FN) per model version
- Precision, recall, false alarms and missed fraud estimated from analyst feedback
- Explanations served and unhandled API errors

### Container Health

Displays:

- Scrape target health
- Container CPU usage
- Container memory working set
- Network received per second
- Network sent per second
- Container uptime

The dashboard is configured using metrics available in the local Docker environment.

### Fraud Risk Monitor

Provides API monitoring: availability, request rate and p95 latency by route,
responses by HTTP status, 4xx/5xx counts and error rate, rejected uploads by
reason, and the Prometheus alerts currently firing or pending.

### Accessible colours

All dashboards use the Okabe–Ito colour-blind-safe palette instead of
red/green: blue means normal or OK, orange means review or warning, and
vermillion means a problem. Status is never shown by colour alone. Panels
carry text labels (UP/DOWN, FIRING), legends list each series with its latest
value, and neighbouring lines alternate between solid and dashed. Only the
numbers are coloured; tile backgrounds are not filled.

Dashboard JSON files are stored in:

```text
monitoring/grafana/dashboards/
```

Grafana dashboards are provisioned automatically from the configured files.

## 15. Automated Testing

The project includes automated tests for the main functional and operational components.

### Run the Complete Local Verification

From the project root:

```bash
source .venv/bin/activate

make check
```

The verification checks include:

- Python compilation
- DVC pipeline status
- Automated Pytest test suite

### Run Pytest Directly

```bash
python -m pytest tests/ -q
```

### Recorded Local Verification

**147 automated tests passed in the last verified local full-suite run.** After the isolated deployment-test fixture correction, all five targeted deployment tests passed.

The test suite covers:

- API health and predictions
- CSV validation
- Drift calculation
- Model loading
- Feedback storage
- Feedback metrics
- Pipeline status
- Execution history
- Spark throughput monitoring
- API integration
- Feedback approval and contamination/duplicate protection
- Feedback-triggered candidate training and MLflow evaluation
- Promotion checks, deployment/rollback safeguards and status API

The complete test plan, test cases and verification report are documented in:

`docs/04_test_plan_and_report.md`

Some local integration tests depend on datasets and model artifacts that are not included in the GitHub repository.

## 16. Continuous Integration — GitHub Actions

The project includes a GitHub Actions workflow:

```text
.github/workflows/ci.yml
```

The workflow provides code-style checks (Ruff, configured in `pyproject.toml`), DVC DAG validation (`dvc dag`, rendered into the job summary), portable automated tests and Docker image build checks.

The CI configuration runs tests and build checks without depending on the complete local training environment.

A verified GitHub Actions run for commit `96b8113` passed both portable tests and Docker image builds. The CI workflow selects 105 pre-existing portable cases plus the additional retraining, promotion and status API test modules. Six local dataset/model-dependent cases are excluded from hosted CI.

The repository's **Actions** tab can be used to inspect workflow runs and their results.

**Repository:**

https://github.com/keerthi-2007/fraud-risk-monitor

## 17. Stopping and Restarting Services

### Check Running Containers

```bash
docker compose ps
```

### Stop Containers

```bash
docker compose stop
```

This stops the services without deliberately deleting persistent Docker volumes.

### Restart Existing Containers

```bash
docker compose start
```

### Start All Configured Services

```bash
docker compose up -d
```

### Restart Only Monitoring Services

```bash
docker compose start prometheus grafana
```

### View Service Logs

For example:

```bash
docker compose logs --tail=100 backend
```

Or:

```bash
docker compose logs --tail=100 prometheus grafana
```

These commands can be used for basic local troubleshooting.

## 18. Project Structure

```text
fraud-risk-monitor/
├── .github/
│   └── workflows/
│       └── ci.yml
├── backend/
│   ├── main.py
│   ├── feedback_store.py
│   ├── feedback_metrics.py
│   ├── feedback_monitor.py
│   ├── drift_monitor.py
│   ├── pipeline_status.py
│   ├── retraining_status.py
│   └── Dockerfile
├── frontend/
│   ├── app.py
│   ├── pipeline_client.py
│   └── Dockerfile
├── scripts/
│   ├── clean_dataset.py
│   ├── create_splits.py
│   ├── spark_audit.py
│   ├── train_baselines.py
│   ├── evaluate_final.py
│   ├── build_drift_baseline.py
│   ├── build_pipeline_status.py
│   ├── run_pipeline_stage.py
│   ├── approve_feedback.py
│   ├── auto_retrain.py
│   ├── retraining_cycle.py
│   ├── prepare_candidate_evaluation.py
│   ├── promotion_controller.py
│   ├── auto_promotion.py
│   ├── check_promotion_candidate.py
│   ├── deployment_env.py
│   └── deploy_candidate.py
├── monitoring/
│   ├── grafana/
│   ├── alertmanager/
│   ├── alerts/
│   ├── webhook/
│   └── prometheus.yml
├── tests/
├── docs/
│   ├── 01_system_architecture.md
│   ├── 02_high_level_design.md
│   ├── 03_low_level_design_api.md
│   ├── 04_test_plan_and_report.md
│   ├── 05_user_manual.md
├── architecture_diagrams/
├── Screenshots/
├── compose.yaml
├── dvc.yaml
├── dvc.lock
├── MLproject
├── Makefile
├── requirements.txt
├── requirements-api.txt
├── requirements-dev.txt
├── requirements-frontend.txt
└── README.md
```

Additional project files and generated data are created in the configured local environment.

## 19. Architecture Documentation

The project includes five primary documentation files.

| Document | File |
|---|---|
| System Architecture | `docs/01_system_architecture.md` |
| High-Level Design | `docs/02_high_level_design.md` |
| Low-Level Design and API Specification | `docs/03_low_level_design_api.md` |
| Test Plan and Test Report | `docs/04_test_plan_and_report.md` |
| User Manual | `docs/05_user_manual.md` |

### Architecture Diagrams

The `architecture_diagrams/` folder contains the visual system designs.

These include:

1. High-Level System Architecture
2. Low-Level Component Architecture
3. DVC Pipeline DAG
4. CSV Prediction Sequence
5. Feedback Sequence
6. Monitoring Architecture
7. Docker Deployment
8. Data Flow Diagram — Level 0
9. Data Flow Diagram — Level 1
10. API Interaction
11. End-to-End AIOps Workflow
12. A modified version of Low-Level Component Architecture with the file name ```12_Data_Preparation_LLD_Clean.png```

A refreshed architecture overview contact sheet (`architecture_diagrams/00__Overview_Contact_Sheet.pdf`) is also provided.

## 20. Project Screenshots

Screenshots of the working application and monitoring tools are organised under:

```text
Screenshots/
├── streamlit_app/        # scoring, feedback, explanations, 7-stage pipeline view
├── mlflow/               # experiments, registry, feature importance, native serving
├── prometheus/           # scrape targets (incl. Alertmanager), alert rules, drift alert
├── Grafana_dashboards/   # fraud operations, container health, risk monitor
├── github_actions/       # CI runs: lint + DVC validation, tests, Docker builds
├── spark/                # Spark transformation metrics (≈20,400 rows/s)
└── docker/               # all eight Compose services running
```

These screenshots document:

- Streamlit fraud prediction and batch scoring
- Streamlit feedback and pipeline overview (seven DVC stages)
- Per-prediction explanations ("Why was a row scored this way?")
- MLflow experiments, metrics and Model Registry
- MLflow permutation feature importance and native model serving (`/invocations`)
- Prometheus scrape targets and alert rules
- Prometheus data drift alerts
- Grafana fraud operations metrics
- Grafana container monitoring
- GitHub Actions CI (code style, DVC DAG validation, tests, Docker builds)
- Spark transformation throughput
- All eight Docker Compose services running

The screenshots provide evidence of the locally executed application and monitoring stack.

## 21. Reproducibility and Limitations

### Reproducibility

The repository contains the application code, DVC pipeline definition, tests, configurations, documentation and CI workflow.

MLflow records include model parameters, metrics, artifacts, run identifiers and training metadata.

However, the original dataset and local MLflow model artifacts are not published in GitHub.

Therefore, cloning the repository alone does not immediately provide the same trained and deployed model.

A new environment needs the original dataset, installed dependencies, environment settings and approved model artifacts before the complete application can be reproduced.

### Scope and Validation Boundaries

- **Application scope:** The system is evaluated locally using historical transactions, consistent with the assignment's no-cloud requirement. Live banking and automatic transaction blocking were not evaluated.
- **Prediction interpretation:** Model scores are not calibrated probabilities. Transactions flagged for review require human judgement.
- **Model lifecycle:** Verified, approved feedback can trigger candidate retraining. Automatic live deployment is disabled by default, and promotion/rollback safeguards were tested in isolation rather than through unattended live rollout.
- **Monitoring and reproducibility:** Operational and drift monitoring use recorded evidence. Reliable real-world feedback performance needs more confirmed outcomes; reproducing the deployed model also requires the original data and MLflow artifacts.

### Candidate retraining and deployment controls

A host-side retraining watcher checks confirmed, approved feedback by default every 60 seconds. Training becomes eligible with **at least 20** approved, genuinely new, valid records including **at least 3 fraud and 3 legitimate** outcomes. It combines eligible new rows with the existing training split, evaluates on validation data, logs a candidate to MLflow and applies quality/promotion gates. The held-out final test split is **not** used to select candidates. An isolated test validates candidate training and promotion evaluation; live automatic production rollout has **not** been verified. `FRAUD_AUTO_DEPLOY` defaults to disabled, so qualifying candidates never replace the live approved model without explicit opt-in.

## 22. Conclusion

Fraud Risk Monitor demonstrates the integration of machine learning with operational monitoring and data pipeline management.

The application combines **Streamlit, FastAPI, Scikit-learn, DVC, MLflow, Apache Spark, Docker, Prometheus, Grafana and GitHub Actions** into one locally deployed system.

It provides fraud risk scoring, recorded pipeline execution information, experiment tracking, automated testing, drift monitoring, feedback capture and operational dashboards.

The project illustrates how AIOps practices can improve the observability, traceability and maintainability of an applied machine learning system.

## Security: Encryption at Rest

Feedback submitted from the UI contains real transaction feature vectors. The backend encrypts them with **Fernet** (AES-128-CBC + HMAC-SHA256, `cryptography` package) before writing `data/feedback/feedback.jsonl`; only `features_encrypted` is stored, and tampering is detected on read. Retraining scripts decrypt transparently via `backend/feedback_crypto.py`.

Generate a key once and add it to the git-ignored `.env`:

```bash
echo "FEEDBACK_ENCRYPTION_KEY=$(python -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())')" >> .env
```

`docker compose up` refuses to start without it. Keep a copy of the key: encrypted feedback cannot be recovered without it. Other controls: token-protected feedback/monitoring endpoints, services bound to `127.0.0.1`, non-root containers, secrets only in `.env`.

