# AIOps Fraud Risk Monitor
## Requirements Traceability Matrix

**Documentation revision:** 19 November 2026

This document maps every requirement in the two assignment documents to the
part of this repository that implements it, and to the evidence that shows it
working.

- **AI Application Evaluation Guideline** (Sections A–E below)
- **Guidelines: Building an AI Application with AIOps** (Sections F–J below)

Status values: **Done** = implemented and verified. **Done (choice)** = the
guideline lists options and the project uses one of them; the reason is given.

---

## A. Demonstration [5 points]

### A1. Web Application Front-end UI/UX Design [3]

| Requirement | Implementation | Evidence | Status |
|---|---|---|---|
| UX intuitive for the problem | Three tabs in plain language: *Single CSV*, *Bulk ZIP upload*, *Pipeline Overview*; results show a clear **review / normal** decision and a summary count | `frontend/app.py`; `Screenshots/streamlit_app/` | Done |
| Easy for non-technical users | Upload a file, read the decision table, download results; captions and info boxes explain each step; no technical input needed | `docs/05_user_manual.md` | Done |
| Foolproof UX | Files checked before sending: size (50 MiB), row limit (100,000), required columns, numeric values, no negatives/NaN; friendly error messages; rejected uploads counted in Prometheus | `read_single_csv`, `read_bulk_zip`, `validate_frame` in `frontend/app.py`; backend re-validates (HTTP 422) | Done |
| Free of UI errors | Backend errors caught and shown as messages, never stack traces; frontend tests | `api_error` in `frontend/app.py`; `tests/test_api.py` | Done |
| Colours, shapes, look-and-feel | Consistent Streamlit theme and icons; Grafana uses the Okabe–Ito colour-blind-safe palette, with text labels in addition to colour | `frontend/app.py`; README §14 "Accessible colours" | Done |
| Responsive UI | Streamlit layout adapts to window width; CSV scoring is vectorised so results return in seconds | `Screenshots/streamlit_app/` | Done |
| User manual | Step-by-step manual for non-technical users | `docs/05_user_manual.md` | Done |

### A2. ML Pipeline Visualization [2]

| Requirement | Implementation | Evidence | Status |
|---|---|---|---|
| Separate UI screen for the ML pipeline, including data ingestion/engineering | *Pipeline Overview* tab lists all DVC stages from ingestion to training, with status and dependencies | `frontend/app.py` (Pipeline Monitor); `Screenshots/streamlit_app/streamlit_pipeline_view_*.png` | Done |
| Orchestrating multiple tool UIs | Streamlit for the pipeline and app; MLflow UI for experiments; Grafana for live metrics; Prometheus/Alertmanager for alerts. Each is linked from the README's application URL table | README §3 | Done |
| Pipeline management console | Pipeline Overview shows each stage's status, "up to date" or outstanding changes, plus retraining status | `backend/pipeline_status.py`, `backend/retraining_status.py` | Done |
| Console of errors, failures and successful runs | "Recent execution history" table: each run's stage, status (success/failed), duration and rows/s | `scripts/run_pipeline_stage.py`; `monitoring/pipeline_executions.jsonl` | Done |
| Speed and throughput of the pipeline | Spark transform: 284,807 rows in 13.95 s (≈20,412 rows/s); Spark audit ≈18,496 rows/s; shown in the console and `dvc metrics show` | `metrics/spark_transform.json`; `docs/04` §9 | Done |

---

## B. Software Engineering [5 points]

### B1. Design Principle [2]

| Requirement | Implementation | Evidence | Status |
|---|---|---|---|
| Design document | System architecture and HLD documents | `docs/01_system_architecture.md`, `docs/02_high_level_design.md` | Done |
| OO or functional paradigm | Mainly functional and modular; OO where natural (Pydantic schemas, scikit-learn Pipeline, FastAPI app state). Rationale documented | `docs/02` §12 "Programming paradigm" | Done |
| LLD with API endpoints and I/O specs | Every endpoint with method, request schema, response schema, status codes and examples | `docs/03_low_level_design_api.md` | Done |
| Architecture diagram | 12 diagrams, including system, deployment, monitoring and data flow | `architecture_diagrams/` | Done |
| HLD diagram | `01_HLD_System_Architecture.png`, embedded in the HLD | `docs/02` | Done |
| Loose coupling: UI and backend connected only via configurable REST | Streamlit never imports model code; it calls FastAPI over HTTP at `FRAUD_API_URL`; separate containers | `frontend/app.py`; `compose.yaml` | Done |

### B2. Implementation [2]

| Requirement | Implementation | Evidence | Status |
|---|---|---|---|
| Standard Python coding style | PEP 8 enforced by Ruff (`pyproject.toml`); 0 violations; checked in CI | `.github/workflows/ci.yml` (lint job) | Done |
| Logging | Python `logging` throughout; `LOG_LEVEL` configurable; prediction, CSV, rejection and model-load events logged | `backend/main.py`; `scripts/*.py` | Done |
| Comprehensive exception handling | Validation errors return HTTP 422; catch-all middleware logs unexpected errors and returns JSON 500 (counted in `fraud_api_unhandled_errors_total`); scripts fail with clear messages | `backend/main.py` (`observe_http_requests`) | Done |
| Design document adhered to | Endpoints, schemas and modules match `docs/03`; API tests check the documented contracts | `tests/test_api.py`, `tests/test_explain.py` | Done |
| Inline documentation | Docstrings on every module, class and function (98/98) plus inline comments on non-obvious logic | `backend/`, `frontend/`, `scripts/` | Done |
| APIs follow the interface spec | Pydantic models enforce the documented schemas (`extra="forbid"`); OpenAPI docs generated at `/docs` | `backend/main.py` | Done |
| Unit tests | 160 automated pytest cases across 29 files | `tests/` | Done |

### B3. Testing [1]

| Requirement | Implementation | Evidence | Status |
|---|---|---|---|
| Test plan | Scope, approach, entry/exit criteria | `docs/04` §1–4 | Done |
| List of test cases | Every automated case listed by ID (AT-xxx) plus 19 manual integration cases (MT-xx) | `docs/04` §5, §8 | Done |
| Test report with number passed and failed | Latest: 154 passed, 0 failed (6 data-dependent cases run in the VM); last full VM run 147/147 | `docs/04` §6, §13 | Done |
| Acceptance criteria defined | 14 measurable criteria (AC-01 to AC-14) | `docs/04` §4 "Acceptance Criteria" | Done |
| Acceptance criteria met | All 14 met, with evidence per criterion | `docs/04` §4, §14 | Done |

---

## C. AIOps Implementation [10 points]

### C1. Data Engineering [2]

| Requirement | Implementation | Evidence | Status |
|---|---|---|---|
| Data ingestion/transformation pipeline using Ray, Spark or Kubernetes | Apache Spark in two DVC stages. `spark_audit` independently validates the raw data. `spark_transform` applies a typed schema, validates, adds engineered columns and writes Parquet partitioned by `Class` | `scripts/spark_audit.py`, `scripts/spark_transform.py`; `dvc.yaml` | Done |
| Throughput and speed | 284,807 rows; transform ≈20,412 rows/s, audit ≈18,496 rows/s | `metrics/spark_transform.json`; `docs/04` §9; `Screenshots/spark/` | Done |

### C2. Source Control & Continuous Integration [2]

| Requirement | Implementation | Evidence | Status |
|---|---|---|---|
| CI implemented using DVC | `dvc.yaml` defines 7 stages with dependencies, params and outputs; `dvc repro` reruns only what changed; `dvc.lock` pins hashes. GitHub Actions validates the DAG on every push | `dvc.yaml`, `dvc.lock`, `params.yaml`; `.github/workflows/ci.yml` | Done |
| DVC DAG of the CI pipeline | `dvc dag`: raw data → prepare → splits → Spark audit/transform, drift baseline, EDA, training; also rendered in the CI job summary | `architecture_diagrams/03_DVC_Pipeline_DAG.png` | Done |
| Git, Git LFS and/or DVC for versioning | Git for code and configs; DVC for data (`data/raw.dvc`) and pipeline outputs; MLflow Model Registry for models | `.dvc/`, `data/raw.dvc` | Done (choice: DVC instead of Git LFS, since the guideline says "and/or") |

### C3. Experiment Tracking [2]

| Requirement | Implementation | Evidence | Status |
|---|---|---|---|
| Experiments tracked during model building | Every training run logged to MLflow (experiment, run, model artifact) | `scripts/train_baselines.py`; `Screenshots/mlflow/` | Done |
| Metrics, parameters and artifacts tracked | Params: all hyperparameters from `params.yaml`. Metrics: validation AP, ROC-AUC, precision, recall and alert counts at the threshold. Artifacts: model, `feature_importance.json`, `params.yaml` | MLflow UI | Done |
| Beyond autolog | Git commit hash, SHA-256 of each data split and of `params.yaml`, permutation feature importance per feature, `top_feature` tag, final test evaluation runs, retraining candidate tags (`quality_gate_passed`, `test_split_labels_used`) | `scripts/train_baselines.py`, `scripts/auto_retrain.py` | Done |

### C4. Exporter Instrumentation & Visualization [2]

| Requirement | Implementation | Evidence | Status |
|---|---|---|---|
| Prometheus instrumentation | `prometheus_client` counters, gauges and histograms exposed at `/metrics` | `backend/main.py`, `backend/feedback_*.py` | Done |
| Information points monitored | Requests by route/status, latency histogram, transactions by decision, upload rejections by reason, per-feature PSI drift, drift analyses/failures, explanations, unhandled errors, feedback records, confusion counts per model | README §13 | Done |
| All components monitored | Backend via app metrics; every container (frontend, backend, MLflow serving, Prometheus, Grafana, Alertmanager, webhook) via cAdvisor; Prometheus, Grafana and Alertmanager self-metrics | `monitoring/prometheus.yml` | Done |
| Grafana visualises in near real time | 3 provisioned dashboards refreshing every 15 s: Fraud Operations, Fraud Risk Monitor, Container Health; 5 alert rules routed through Alertmanager to a webhook | `monitoring/grafana/`, `monitoring/alerts/`; `Screenshots/Grafana_dashboards/` | Done |

### C5. Software Packaging [2]

| Requirement | Implementation | Evidence | Status |
|---|---|---|---|
| MLflow for APIfication | Backend loads the model from MLflow; the `mlflow-serve` service exposes the same model through MLflow's native `/invocations` REST API on port 5001 | `compose.yaml`; `Screenshots/mlflow/mlflow_serving_invocations.png` | Done |
| MLprojects for identical dev and test environments | `MLproject` with `python_env.yaml` (pinned Python and requirements) and entry points for each stage | `MLproject`, `python_env.yaml` | Done |
| FastAPI for the UI's APIs | `/predict`, `/predict/csv`, `/explain`, `/feedback`, `/health`, `/metrics`, monitoring endpoints | `backend/main.py`; `docs/03` | Done |
| Backend and frontend dockerised | Separate Dockerfiles, non-root users | `backend/Dockerfile`, `frontend/Dockerfile` | Done |
| Docker Compose with separate services | 8 services: backend, frontend, mlflow-serve, Prometheus, Grafana, Alertmanager, cAdvisor, alert-webhook | `compose.yaml`; `Screenshots/docker/docker_compose_ps.png` | Done |

---

## D. Viva [5 points]

Preparation is outside the repository; the README, design documents and this
matrix provide the material.

## E. Mandatory documents

| Document | File | Status |
|---|---|---|
| Architecture diagram with explanation of the blocks | `docs/01_system_architecture.md`, `architecture_diagrams/` | Done |
| HLD with design choices and rationale | `docs/02_high_level_design.md` (§12 Design Decisions) | Done |
| LLD with endpoint definitions and I/O specification | `docs/03_low_level_design_api.md` | Done |
| Test plan and test cases | `docs/04_test_plan_and_report.md` | Done |
| User manual for non-technical users | `docs/05_user_manual.md` | Done |

---

## F. Core Principles (Development Guideline §I)

| Principle | Implementation | Status |
|---|---|---|
| Automation | DVC automates ingestion → preprocessing → Spark → EDA → training; CI automates lint, tests and image builds; monitoring and alerting are automatic; a watcher automates feedback-driven retraining | Done |
| Reproducibility (Git commit + MLflow run ID) | Every training run logs its Git commit and data hashes; the deployed model is pinned by MLflow run ID `27da3507bddb413bab9cd0a12aeb092a`; `dvc.lock` pins every input and output | Done |
| Continuous Integration | GitHub Actions: Ruff lint, DVC DAG validation, portable tests, Docker builds | Done |
| Monitoring and logging | Prometheus/Grafana for performance, drift and infrastructure; structured logging in the backend and scripts | Done |
| Collaboration | Shared GitHub repo with pull requests and CI checks; shared MLflow and Grafana UIs; documentation in `docs/` | Done |
| Version control of all artifacts | Code and configs in Git; data in DVC; models in the MLflow Model Registry | Done |
| Environment parity with Docker | Same images for every environment through `compose.yaml`; MLproject for the training environment | Done |

## G. Lifecycle Stages (Development Guideline §II)

| Stage | Requirement | Implementation | Status |
|---|---|---|---|
| A. Problem & data | Business problem and measurable success metrics | `docs/02` §1 "Business problem and success metrics" | Done |
| | Data sources, formats, potential biases | `docs/02` §1 "Data sources and known biases"; README §4 | Done |
| | EDA | `eda` DVC stage → `data/reports/eda_summary.md` + charts | Done |
| | Version data collection scripts and configs | `scripts/fetch_dataset.py`, `data/raw.dvc` in Git | Done |
| | Automated ingestion and validation | `fetch_dataset` with checksum manifest; `prepare_data` verifies the checksum and schema | Done |
| | Data quality checks and monitoring | Quality, cleaning and duplicate reports; Spark audit; live PSI drift monitoring | Done |
| B. Preprocessing & features | Cleaning and transformation | `clean_dataset.py`, `spark_transform.py` | Done |
| | Feature engineering | `spark_transform`: `log_amount`, `is_zero_amount`, `amount_zscore`, `feature_hash`, `duplicate_count`; StandardScaler in the model pipeline | Done |
| | Automated, version-controlled preprocessing | DVC stages, scripts in Git | Done |
| | Track feature importance | Permutation importance logged per feature to MLflow and `metrics/feature_importance.json` | Done |
| C. Model development | Model selection | Logistic Regression and HistGradientBoosting compared on average precision and recall | Done |
| | Hyperparameter experiments | `params.yaml` tracked by DVC (`dvc params diff`) and logged to MLflow | Done |
| | Evaluation and comparison | Validation metrics per model; held-out test evaluation of the selected model | Done |
| | Automated training/evaluation; tracked versions and metrics; MLflow; Docker | `train_baselines` DVC stage; MLflow Registry; Docker/MLproject environments | Done |
| D. Deployment | Deployment strategy | Online REST (single transaction) and batch (CSV/ZIP) scoring | Done |
| | Model serving | FastAPI and MLflow Serve | Done |
| | CI/CD deployment | CI builds deployable images; `deploy_candidate.py` deploys a verified candidate with health checks | Done |
| | Monitor production performance | Feedback-based precision/recall, false alarms and missed fraud in Grafana | Done |
| | Rollback for failed deployments | `deploy_candidate.py` restores the previous model if health checks fail | Done |
| E. Monitoring & maintenance | Performance monitoring | API latency, error rate and feedback-based model quality | Done |
| | Data drift detection | Per-feature PSI on every batch of 100+ rows; `FraudFeatureDriftHigh` alert | Done |
| | Retraining | Feedback-triggered: at least 20 approved new records (3+ fraud, 3+ legitimate) → candidate trained, quality-gated, logged. Drift raises an alert; retraining waits for labelled feedback, because drift alone gives no labels to learn from | Done |
| | Automated alerting, drift/performance history, model versioning | Alertmanager; Prometheus time series; MLflow Registry | Done |

## H. Technology Stack (Development Guideline §III)

| Recommendation | Used | Status |
|---|---|---|
| Version control: Git LFS, DVC | DVC (with Git) | Done (choice) |
| Data engineering: Spark, Ray, Kubernetes | Apache Spark | Done (choice) |
| Experiment tracking: MLflow | MLflow | Done |
| Containerisation: Docker, Compose, Docker Hub | Docker and Docker Compose; images built locally and in CI | Done (choice: images are not published to Docker Hub) |
| CI/CD: DVC | DVC pipeline plus GitHub Actions | Done |
| Model serving: MLflow Serve, Ray Serve, REST APIs | MLflow Serve and FastAPI REST | Done |
| Monitoring: Prometheus, Grafana | Prometheus, Grafana, Alertmanager, cAdvisor | Done |
| Cloud platforms: not allowed | Everything runs locally | Done |

## I. Best Practices (Development Guideline §IV)

| Practice | Implementation | Status |
|---|---|---|
| Code quality | Modular code, Ruff-enforced style, full docstrings | Done |
| Testing: unit, integration, end-to-end | Unit tests; API and integration tests; isolated end-to-end retraining test (`tests/test_isolated_retraining_e2e.py`); 19 manual integration checks | Done |
| Security (encryption) | Feedback features encrypted at rest with Fernet (AES-128-CBC + HMAC-SHA256); token-protected feedback/monitoring endpoints; services bound to `127.0.0.1`; non-root containers; secrets only in a git-ignored `.env` | Done |
| Scalability | Stateless API replicas, bounded vectorised batch scoring, Spark, independent services; see `docs/02` §12 "Scalability" | Done |
| Explainability | `/explain` per-prediction feature contributions shown in the UI; global permutation feature importance in MLflow | Done |
| Documentation | README, 6 documents in `docs/`, architecture diagrams, screenshots, docstrings | Done |

## J. Continuous Improvement (Development Guideline §V)

| Practice | Implementation | Status |
|---|---|---|
| Regularly review and update | Changes go through pull requests with CI checks; monitoring and feedback feed retraining | Done |
| Culture of experimentation | Candidate models are tracked in MLflow and compared against gates before promotion | Done |
