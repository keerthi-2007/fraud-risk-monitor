# FraudGuard - AI-Powered Financial Fraud Detection and Risk Monitoring System

**A local-first educational AI application with AIOps tooling.** Built as original project code, informed by external examples (see `docs/ATTRIBUTION_AND_LIMITATIONS.md`). No cloud platform is required.

> **IMPORTANT: The included dataset and demonstration model are SYNTHETIC. The app is not suitable for production banking or automatic payment blocking. Model scores are not proven calibrated probabilities.**

## What you get

- **Frontend:** Streamlit risk dashboard, CSV scoring, export, corrections, pipeline status, model comparison.
- **Backend:** Separate FastAPI REST service (`/predict`, `/feedback`, `/health`, `/model/info`, `/model/evaluation`, `/pipeline/status`, `/system/overview`, `/metrics`).
- **ML:** Two candidate classifiers (class-weighted logistic regression, Random Forest), validation PR-AUC model selection, F2 threshold selection, held-out test metrics.
- **AIOps:** DVC pipeline DAG, MLflow candidate runs, Git + optional Git LFS, optional Ray Data ingestion, Prometheus/Grafana runtime monitoring, local feedback review before retraining.
- **Packaging:** Slim Python containers in Docker Compose; monitoring services optional for small VMs.
- **Documentation:** Architecture diagram, HLD, LLD, test plan, user guide, attribution and verification report.

## Quick start on Linux VM (recommended path)

**Requirements:** Linux VM with Python 3.11, `python3.11-venv`, pip, Docker Engine + Docker Compose plugin (to use containers), and Git. Have internet access to install Python/Docker dependencies, even though runtime uses no cloud platform.

```bash
unzip FraudGuard_AIOps_Project.zip
cd fraudguard-aiops
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-train.txt
```

For an all-in-one bootstrap on an Ubuntu-like VM (after installing Python 3.11, Git and Docker Compose), run `bash scripts/setup_vm.sh`. This installs training dependencies, initializes local DVC, trains on the demo dataset if necessary, runs tests and starts two containers. It does **not** start Ray or Grafana by default. You can also follow the individual commands below.

The ZIP may include a **pretrained synthetic demo artifact**, but retrain once inside your VM to ensure identical sklearn versions and prove reproducibility:

```bash
python -m scripts.generate_demo --rows 3600
python -m pipeline.run
python -m pytest -q
```

If you have the real permitted dataset, save it as `data/raw/creditcard.csv` and rerun the pipeline. The real dataset is **not** included in this ZIP; original data takes precedence over demo data.

### Launch the application (2 containers, minimal stack)

```bash
docker compose up -d --build
```

- Dashboard: `http://localhost:8501`
- Backend docs: `http://localhost:8000/docs`
- API health: `http://localhost:8000/health`

Docker ports are bound to `127.0.0.1` on the VM. Use the VM browser, or forward ports over SSH if your VM is headless. For file permissions, ensure `data/feedback/` and `run/` are writable by UID `1000` (default Compose) or export `LOCAL_UID=$(id -u)` and `LOCAL_GID=$(id -g)` before `docker compose up`. On SELinux-enabled systems, bind mounts may also need labeling.

### Start observability tools only when needed

```bash
docker compose --profile monitoring up -d
```

- Prometheus: `http://localhost:9090/targets` (backend target should be UP)
- Grafana: `http://localhost:3000` (`admin` / value of `GRAFANA_ADMIN_PASSWORD`)
- Included Grafana dashboard: **FraudGuard | Runtime and Fraud Risk**.

For local demo only, Compose uses an insecure fallback Grafana password. Set a strong `GRAFANA_ADMIN_PASSWORD` in a `.env` file before any network exposure. Monitoring containers add memory/disk overhead.

### DVC / Git / Git LFS (separate roles)

```bash
bash scripts/bootstrap.sh
git add .
git commit -m 'Initial FraudGuard pipeline'
GIT_COMMIT=$(git rev-parse HEAD) dvc repro
dvc dag
dvc metrics show
```

- **Git:** code and version references (commit hash).
- **DVC:** data pipeline dependencies, hashed data/model artifacts, local storage under `storage/dvc/`.
- **Git LFS:** optional `*.onnx`, `*.pt`, `*.bin` assets outside DVC; do **not** track the same model in both DVC and LFS. `git lfs install` is optional and only works when installed.
- **MLflow:** experiments under local `mlruns/`; each model candidate logs params, metrics and the run ID. Run `mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5000` and open `http://localhost:5000`.
- **Provenance:** export `GIT_COMMIT=$(git rev-parse HEAD)` on training commands to record commit in MLflow and metadata. A DVC lockfile is generated when running `dvc repro`; it is deliberately not fabricated in this distribution.

`python -m pipeline.run` is a first-boot convenience route that uses the same stages as DVC. To demonstrate DVC reproducibility, run **`dvc repro`**, not just the convenience runner.

### Optional Ray ingestion (save VM space by default)

```bash
pip install -r requirements-ray.txt
DATA_ENGINE=ray python -m pipeline.ingest
# or when DVC is initialized:
DATA_ENGINE=ray dvc repro --force ingest
```

The Ray reader uses `ray.data.read_csv` and records ingestion time and rows/second in `run/pipeline_status.json`. It is opt-in because Ray installs a larger dependency footprint. Do not claim that pandas-mode runs were Ray runs. The current sklearn model step still collects data in memory; this is not a fully streaming fraud analytics system.

### Feedback loop (review-first)

1. In the UI, score transactions and submit an incorrect prediction. It goes to `data/feedback/pending_feedback.csv`.
2. Inspect proposed corrections: `python -m scripts.review_feedback`.
3. **Only for transaction labels that a human has verified:** `python -m scripts.review_feedback --approve TRANSACTION_ID`.
4. Rerun `dvc repro`, evaluate and manually approve the new candidate before using it operationally.

For a local demo, `FEEDBACK_TOKEN` can be set in `.env` and passed to both UI and API; full authentication is beyond this project.

### Tests & pipeline management

```bash
bash scripts/ci.sh
python -m scripts.pipeline_console dag
python -m scripts.pipeline_console status
python -m scripts.pipeline_console run
python -m scripts.pipeline_console metrics
```

To enable local pre-commit CI: `git config core.hooksPath .githooks`. For CI that also reruns the DVC DAG: `RUN_DVC_REPRO=1 bash scripts/ci.sh`.

CLI operations are intentionally restricted to the local VM; there is no anonymous HTTP endpoint for running arbitrary shell commands. The dashboard has a read-only per-stage management screen with refresh, statuses and timings.

## Project layout

```text
fraudguard/           settings, schema, drift, FastAPI inference
pipeline/             ingest, optional Ray reader, split, train, evaluate, report
frontend/             Streamlit web app
scripts/              synthetic data, feedback review, bootstrap, audit, console
infra/                slim Dockerfiles, Prometheus/Grafana configuration
artifacts/            generated model, evaluation, drift reference and report
data/                 raw, verified/pending feedback, interim, train/val/test
mlruns/               MLflow tracking files (generated when MLflow installed)
docs/                 HLD, LLD, architecture, test plan, user manual
tests/                automated API, validation, end-to-end pipeline checks
dvc.yaml              source-of-truth data/ML DAG
docker-compose.yml    separate frontend/backend, optional monitoring
```

## Honest implementation notes

- The ZIP includes working source, generated synthetic artifacts and automated tests. The supplied environment can test Python but **does not provide Docker, DVC, MLflow or Ray binaries**, so those parts are *configured but require verification on your VM*.
- UI and docker-compose are authored but a real-browser UI test and full Docker validation on the target VM are still required. Check `docs/TEST_PLAN.md`.
- No automatic model retraining is triggered by untrusted feedback; this is intentional.
- No cardholder data should be uploaded. The original dataset columns are anonymized, not directly interpretable personal financial attributes.
- Verify external licence terms and disclose AI/external reference use per course policy. See attribution document.

## Instructor rubric crosswalk

| Evaluation point | Implementation / evidence |
|---|---|
| User-friendly UI | `frontend/app.py`, `docs/USER_MANUAL.md` |
| Pipeline visualization | Pipeline control page, `run/pipeline_status.json`, `dvc dag` |
| HLD, LLD, architecture | `docs/HLD.md`, `docs/LLD.md`, `docs/architecture.png` |
| Tests + acceptance criteria | `tests/`, `docs/TEST_PLAN.md`, `docs/TEST_REPORT.md` |
| Distributed processing | `pipeline/ray_data.py` + explicit `DATA_ENGINE=ray` demonstration |
| DVC DAG + CI | `dvc.yaml`, `scripts/ci.sh`, optional `.githooks/pre-commit`, `scripts/pipeline_console.py` |
| Version control | Git bootstrap + `.gitattributes` LFS guidance |
| MLflow experiments | `pipeline/train.py` logs candidate run IDs, params, artifacts |
| Monitoring and visualization | `/metrics`, `infra/prometheus.yml`, Grafana dashboard JSON |
| Packaging | Dockerfiles, Docker Compose, separate services, local ports |

## Attribution

See `docs/ATTRIBUTION_AND_LIMITATIONS.md`. No original senior/Kaggle code is copied into this package. These resources were used to understand the educational assignment and compare approaches; respective rights remain with the authors.
