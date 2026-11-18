# Fraud Risk Monitor — Architecture Diagram Pack
**Contents:** 11 numbered architecture figures as SVG and PNG, one additional data-preparation PNG, and a contact sheet. Most updated figures include editable Graphviz `.dot` source; the feedback figure is directly editable as SVG. Original `.mmd` source files are not part of this package.


## Styles and interpretation

- Blue = user experience / Streamlit; green = FastAPI / ML; orange = DVC and data; purple = observability; grey = storage; red = error/alert.
- Solid arrows = request, flow or dependency; dashed arrows = independently controlled operation, representative failure, or historical snapshot relationship.
- SVG diagrams are scalable for Word/LaTeX/report use; PNG diagrams are convenient for slides. Revised Graphviz figures have editable `.dot` sources; the feedback diagram is an editable SVG; unchanged original SVG figures remain available.
- The feedback-sequence figure shows the verified approval/eligibility and evaluation flow; the prediction-sequence figure remains unchanged.
- Diagram groupings are explanatory architecture boundaries, not necessarily literal Docker networking or Python packages.


## Diagram index

### 1. High-Level System Architecture

An overview of the four principal areas: user interaction, online inference, controlled offline data/model operations, and operational monitoring. Solid arrows represent actual request paths or data dependencies. Spark audit is deliberately independent of the train/split path, and confirmed, approved, eligible feedback can feed a separately controlled candidate retraining loop.

**Placement:** Architecture / HLD — main figure
**Caption:** Figure 1. High-level architecture of the local Fraud Risk Monitor AIOps application.
**Files:** `01_HLD_System_Architecture.svg`, `01_HLD_System_Architecture.png`
**Relevant source paths to cross-check:** `frontend/app.py`, `backend/main.py`, `backend/feedback_store.py`, `dvc.yaml`, `MLproject`, `compose.yaml`, `monitoring/prometheus.yml`, `monitoring/alertmanager/alertmanager.yml`

### 2. Low-Level Backend and Frontend Components

The component diagram separates FastAPI routes, the approved in-memory MLflow model, validation, drift and feedback logic, and the read-only pipeline monitoring endpoint. The single-transaction JSON route is an API capability rather than a separate Streamlit screen. Stored feedback is only considered after manual approval, validation and sample eligibility; deployment is opt-in.

**Placement:** LLD — component design
**Caption:** Figure 2. Low-level component architecture, including API, persistence and monitoring boundaries.
**Files:** `02_LLD_Component_Architecture.svg`, `02_LLD_Component_Architecture.png`, `12_Data_Preparation_LLD_Clean.png`
**Relevant source paths to cross-check:** `frontend/app.py`, `frontend/pipeline_client.py`, `backend/main.py`, `backend/feedback_store.py`, `backend/feedback_metrics.py`, `backend/drift_monitor.py`, `backend/pipeline_status.py`

### 3. DVC Pipeline Dependency DAG

The exact conceptual DVC DAG has four stages. The Spark audit reads raw data independently; it neither produces the split data nor gates training. Drift baseline generation consumes the training split. MLflow training and evaluation are shown dashed because they are controlled commands outside the DVC graph.

**Placement:** HLD — data pipeline; LLD — dependency mapping
**Caption:** Figure 3. Four-stage DVC pipeline, independent Spark audit, and separately managed MLflow lifecycle.
**Files:** `03_DVC_Pipeline_DAG.svg`, `03_DVC_Pipeline_DAG.png`
**Relevant source paths to cross-check:** `dvc.yaml`, `dvc.lock`, `scripts/clean_dataset.py`, `scripts/create_splits.py`, `scripts/spark_audit.py`, `scripts/build_drift_baseline.py`, `MLproject`

### 4. CSV Prediction Sequence

A chronological view of one CSV-scoring request, including frontend validation, API limits, model inference, the review threshold, optional PSI drift monitoring and downloadable results. Rejection paths and the non-blocking nature of monitoring are explicit. This diagram describes the Single CSV experience; bulk ZIP processing repeats individual CSV scoring.

**Placement:** LLD — API sequence diagrams
**Caption:** Figure 4. UML sequence diagram for CSV fraud scoring and optional drift analysis.
**Files:** `04_Prediction_Sequence.svg`, `04_Prediction_Sequence.png`
**Relevant source paths to cross-check:** `frontend/app.py`, `backend/main.py`, `backend/drift_monitor.py`

### 5. Corrected Outcome Feedback Sequence

A reviewer can submit a confirmed legitimate or fraud outcome after a prediction. The backend checks an access token, validates the full request, stores the record in JSONL and returns an identifier. Feedback metrics may later aggregate records, and records are later filtered and approved for candidate training; the active model remains unchanged by default.

**Placement:** LLD — feedback sequence
**Caption:** Figure 5. Feedback lifecycle from authenticated collection through independent approval, candidate training and deployment safety.
**Files:** `05_Feedback_Sequence.svg`, `05_Feedback_Sequence.png`
**Relevant source paths to cross-check:** `frontend/app.py`, `backend/feedback_store.py`, `scripts/approve_feedback.py`, `scripts/auto_retrain.py`, `scripts/retraining_cycle.py`

### 6. Observability and Alerting Pipeline

Prometheus scrapes application and container metrics and evaluates five alert rules. Grafana reads Prometheus for dashboard visualisations. Alertmanager routes firing alerts to an internal webhook receiver that appends notification records. A controlled Alertmanager-to-webhook delivery was verified, but not every alert condition has been tested end-to-end.

**Placement:** HLD — monitoring subsystem
**Caption:** Figure 6. Prometheus-based metrics, dashboard and alert notification architecture.
**Files:** `06_Monitoring_Architecture.svg`, `06_Monitoring_Architecture.png`
**Relevant source paths to cross-check:** `backend/main.py`, `backend/feedback_monitor.py`, `monitoring/prometheus.yml`, `monitoring/alerts/fraud-alerts.yml`, `monitoring/alertmanager/alertmanager.yml`, `monitoring/webhook/receiver.py`, `compose.yaml`

### 7. Local Docker Compose Deployment

The local deployment contains seven Docker Compose services. Streamlit is published on port 8501; the backend, Prometheus, Grafana and Alertmanager bind to loopback host ports. cAdvisor and the webhook receiver are internal services. Models, feedback and the pipeline snapshot use mounts, while Prometheus and Grafana retain data in managed volumes.

**Placement:** HLD — deployment architecture
**Caption:** Figure 7. Seven-container local deployment with separate host retraining watcher and persisted feedback.
**Files:** `07_Docker_Deployment.svg`, `07_Docker_Deployment.png`
**Relevant source paths to cross-check:** `compose.yaml`, `backend/Dockerfile`, `frontend/Dockerfile`, `monitoring/prometheus.yml`, `monitoring/alertmanager/alertmanager.yml`

### 8. Data Flow Diagram — Level 0

This context-level diagram treats Fraud Risk Monitor as one process. It identifies the reviewing user, source transactions, approved model, saved pipeline status and feedback records. Operational monitoring consumes metrics but does not trigger autonomous model changes.

**Placement:** HLD — context diagram
**Caption:** Figure 8. DFD Level 0: external entities, data stores and the Fraud Risk Monitor boundary.
**Files:** `08_DFD_Level_0.svg`, `08_DFD_Level_0.png`
**Relevant source paths to cross-check:** `frontend/app.py`, `backend/main.py`, `backend/feedback_store.py`, `backend/pipeline_status.py`, `compose.yaml`

### 9. Data Flow Diagram — Level 1

DFD Level 1 decomposes the application into file intake, validation, scoring, drift monitoring, feedback capture, saved pipeline-status retrieval and metric export. It intentionally models best-effort drift independently of the scoring result and a separate approval and candidate retraining branch.

**Placement:** LLD — data-flow model
**Caption:** Figure 9. DFD Level 1: detailed application processes and persistent data stores.
**Files:** `09_DFD_Level_1.svg`, `09_DFD_Level_1.png`
**Relevant source paths to cross-check:** `frontend/app.py`, `frontend/pipeline_client.py`, `backend/main.py`, `backend/feedback_store.py`, `backend/drift_monitor.py`, `backend/pipeline_status.py`, `scripts/auto_retrain.py`

### 10. FastAPI Endpoint Map

The endpoint map shows the eight supported FastAPI routes, including the authenticated retraining-status endpoint. The prediction endpoints return scores and decisions; the feedback route is token-protected and persists corrected outcomes; monitoring routes use a separate token. The metrics route exposes Prometheus exposition data. Dashed links indicate representative error categories.

**Placement:** LLD — API overview
**Caption:** Figure 10. FastAPI endpoint inventory, dependencies and key validation responses.
**Files:** `10_API_Interaction.svg`, `10_API_Interaction.png`
**Relevant source paths to cross-check:** `backend/main.py`, `backend/feedback_store.py`, `backend/pipeline_status.py`, `backend/retraining_status.py`

### 11. End-to-End AIOps Lifecycle

This holistic workflow separates repeatable DVC data work, separately controlled MLflow experiments, online inference and operational monitoring. Human-reviewed feedback is retained for monitoring. Sufficient independently approved, eligible feedback can trigger a host-side candidate retraining watcher; drift alerts alone do not trigger training. Deployment is opt-in, and the pipeline dashboard reads a generated snapshot.

**Placement:** Architecture / HLD — closing lifecycle figure
**Caption:** Figure 11. End-to-end AIOps lifecycle with gated candidate retraining and disabled-by-default model deployment.
**Files:** `11_End_to_End_AIOps_Workflow.svg`, `11_End_to_End_AIOps_Workflow.png`, `11_End_to_End_AIOps_Workflow.dot`, `00__Overview_Contact_Sheet.pdf`
**Relevant source paths to cross-check:** `dvc.yaml`, `MLproject`, `scripts/train_baselines.py`, `scripts/evaluate_final.py`, `backend/main.py`, `frontend/app.py`, `monitoring/prometheus.yml`, `compose.yaml`

