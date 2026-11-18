from contextlib import asynccontextmanager
from time import perf_counter
from io import BytesIO, StringIO
import math
import logging
import os
import secrets
import time
from pathlib import Path
from typing import Annotated, Literal
from fastapi.middleware.cors import CORSMiddleware
from backend import feedback_store
from backend.feedback_store import router as feedback_router
from backend.feedback_monitor import refresh_feedback_metrics
from backend.pipeline_status import router as pipeline_status_router
from backend.retraining_status import router as retraining_status_router
from backend.drift_monitor import load_baseline, detect_drift_from_baseline

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.responses import Response, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel, ConfigDict, Field

ROOT = Path(__file__).resolve().parents[1]
FEATURE_COLUMNS = [f"V{i}" for i in range(1, 29)] + ["Amount"]

MODEL_RUN_ID = os.getenv(
    "MODEL_RUN_ID",
    "27da3507bddb413bab9cd0a12aeb092a",
)
DECISION_THRESHOLD = float(
    os.getenv("DECISION_THRESHOLD", "0.87292890258894895")
)
TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    f"sqlite:///{ROOT / 'mlflow.db'}",
)

MAX_CSV_BYTES = 50 * 1024 * 1024
MAX_CSV_ROWS = 100_000

HTTP_REQUESTS = Counter(
    "fraud_api_requests_total",
    "HTTP requests handled by the fraud risk API.",
    ["method", "route", "status"],
)
HTTP_REQUEST_DURATION = Histogram(
    "fraud_api_request_duration_seconds",
    "API request duration in seconds.",
    ["method", "route"],
)
TRANSACTIONS_SCORED = Counter(
    "fraud_transactions_scored_total",
    "Transaction rows successfully scored by the fraud model.",
    ["source", "decision"],
)


CSV_VALIDATION_FAILURES = Counter(
    "fraud_csv_validation_failures_total",
    "CSV or ZIP uploads rejected by frontend validation.",
    ["upload_type", "reason"],
)


DRIFT_PSI = Gauge(
    "fraud_feature_drift_psi",
    "Latest PSI value for a monitored transaction feature.",
    ["feature"],
)

DRIFT_ANALYSES = Counter(
    "fraud_drift_analyses_total",
    "Completed transaction-distribution drift analyses.",
    ["status"],
)

DRIFT_LAST_ANALYSIS = Gauge(
    "fraud_drift_last_analysis_timestamp_seconds",
    "Unix timestamp of the latest completed drift analysis.",
)


class ValidationFailureEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    upload_type: Literal["single_csv", "bulk_zip"]
    reason: Literal[
        "missing_columns",
        "unexpected_columns",
        "invalid_values",
        "negative_amount",
        "empty_file",
        "unreadable_file",
        "file_limit",
        "invalid_archive",
        "other",
    ]


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    features: dict[str, float] = Field(
        description="Exactly V1 through V28 and Amount."
    )


class PredictionResponse(BaseModel):
    model_run_id: str
    risk_score: float
    score_type: Literal["uncalibrated_model_score"]
    decision_threshold: float
    decision: Literal["review", "normal"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    mlflow.set_tracking_uri(TRACKING_URI)
    model_uri = os.getenv(
        "MODEL_URI",
        f"runs:/{MODEL_RUN_ID}/model",
    )
    if model_uri.startswith("runs:/"):
        parts = model_uri.removeprefix("runs:/").split("/")
        if len(parts) != 2 or parts[1] != "model" or not parts[0]:
            raise ValueError(f"Unsupported MLflow run URI: {model_uri}")
        actual_run_id = parts[0]
    elif model_uri.startswith("models:/"):
        parts = model_uri.removeprefix("models:/").split("/")
        if len(parts) != 2 or not parts[0] or not parts[1].isdigit():
            raise ValueError(
                "Use a numbered registered model URI, such as "
                "models:/FraudRiskClassifier/1"
            )
        actual_run_id = mlflow.tracking.MlflowClient().get_model_version(
            parts[0], parts[1]
        ).run_id
        if not actual_run_id:
            raise ValueError("Registered model has no source run ID.")
    else:
        raise ValueError(f"Unsupported MLflow model URI: {model_uri}")

    evaluation_run_id = os.getenv(
        "EVALUATION_RUN_ID",
        "008277161614417d8da7e13727013676",
    )
    evaluation = mlflow.tracking.MlflowClient().get_run(
        evaluation_run_id
    )

    if evaluation.info.status != "FINISHED":
        raise ValueError("Approved model evaluation is not finished.")

    if evaluation.data.params.get("source_model_run_id") != actual_run_id:
        raise ValueError(
            "Selected model does not match the approved evaluation run."
        )

    evaluated_threshold = evaluation.data.params.get("decision_threshold")
    if evaluated_threshold is None or float(evaluated_threshold) != DECISION_THRESHOLD:
        raise ValueError(
            "Decision threshold does not match the approved evaluation."
        )

    # Load through the source run URI to keep registered-model
    # deployment compatible with read-only artifact storage.
    load_uri = f"runs:/{actual_run_id}/model"

    app.state.model = mlflow.sklearn.load_model(load_uri)
    app.state.model_uri = model_uri
    app.state.model_run_id = actual_run_id
    app.state.drift_baseline = load_baseline(
        ROOT / "monitoring/drift_baseline.json"
    )
    yield


app = FastAPI(
    title="Fraud Risk Monitor API",
    version="0.1.0",
    description="Local API for scoring transaction risk.",
    lifespan=lifespan,
)
@app.middleware("http")
async def observe_http_requests(request, call_next):
    started_at = perf_counter()
    response = await call_next(request)
    route = request.scope.get("route")
    route_path = getattr(route, "path", "unmatched")

    if route_path != "/metrics":
        HTTP_REQUESTS.labels(
            method=request.method,
            route=route_path,
            status=str(response.status_code),
        ).inc()
        HTTP_REQUEST_DURATION.labels(
            method=request.method,
            route=route_path,
        ).observe(perf_counter() - started_at)

    return response

app.include_router(feedback_router)
app.include_router(pipeline_status_router)
app.include_router(retraining_status_router)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/metrics", include_in_schema=False)
def metrics():
    try:
        refresh_feedback_metrics(feedback_store.FEEDBACK_FILE)
    except (OSError, ValueError):
        logging.getLogger(__name__).exception(
            "Feedback monitoring refresh failed."
        )

    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.post("/monitoring/validation-failure", status_code=204)
def record_validation_failure(
    event: ValidationFailureEvent,
    x_monitoring_token: str | None = Header(default=None),
):
    expected_token = os.getenv("MONITORING_TOKEN")
    if (
        not expected_token
        or not x_monitoring_token
        or not secrets.compare_digest(x_monitoring_token, expected_token)
    ):
        raise HTTPException(status_code=403, detail="Monitoring access denied.")

    CSV_VALIDATION_FAILURES.labels(
        upload_type=event.upload_type,
        reason=event.reason,
    ).inc()
    return Response(status_code=204)


@app.get("/health")
def health(request: Request):
    return {
        "status": "ok",
        "model_run_id": request.app.state.model_run_id,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictRequest, request: Request):
    supplied = set(payload.features)
    expected = set(FEATURE_COLUMNS)
    missing = sorted(expected - supplied)
    extra = sorted(supplied - expected)

    if missing or extra:
        raise HTTPException(
            status_code=422,
            detail={"missing_features": missing, "unexpected_features": extra},
        )

    values = {
        name: float(payload.features[name])
        for name in FEATURE_COLUMNS
    }

    if not all(math.isfinite(value) for value in values.values()):
        raise HTTPException(
            status_code=422,
            detail="Feature values must all be finite numbers.",
        )

    if values["Amount"] < 0:
        raise HTTPException(
            status_code=422,
            detail="Amount must be zero or greater.",
        )

    frame = pd.DataFrame([values], columns=FEATURE_COLUMNS)
    score = float(request.app.state.model.predict_proba(frame)[0, 1])
    decision = "review" if score >= DECISION_THRESHOLD else "normal"
    TRANSACTIONS_SCORED.labels(source="json", decision=decision).inc()

    return PredictionResponse(
        model_run_id=request.app.state.model_run_id,
        risk_score=score,
        score_type="uncalibrated_model_score",
        decision_threshold=DECISION_THRESHOLD,
        decision=decision,
    )


@app.post("/predict/csv")
async def predict_csv(
    request: Request,
    file: Annotated[
        UploadFile,
        File(description="CSV containing V1 through V28 and Amount."),
    ],
):
    if not file.filename or Path(file.filename).suffix.lower() != ".csv":
        raise HTTPException(status_code=415, detail="Upload a .csv file.")

    content = await file.read(MAX_CSV_BYTES + 1)
    await file.close()

    if len(content) > MAX_CSV_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"CSV exceeds the {MAX_CSV_BYTES // (1024 * 1024)} MB limit.",
        )

    try:
        frame = pd.read_csv(BytesIO(content))
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail="Could not read the uploaded CSV.",
        ) from exc

    if frame.empty:
        raise HTTPException(status_code=422, detail="CSV has no transaction rows.")

    if len(frame) > MAX_CSV_ROWS:
        raise HTTPException(
            status_code=413,
            detail=f"CSV exceeds the {MAX_CSV_ROWS:,}-row limit.",
        )

    supplied = set(frame.columns)
    expected = set(FEATURE_COLUMNS)
    missing = sorted(expected - supplied)
    extra = sorted(supplied - expected)

    if missing or extra:
        raise HTTPException(
            status_code=422,
            detail={
                "missing_features": missing,
                "unexpected_columns": extra,
                "required_columns": FEATURE_COLUMNS,
            },
        )

    features = frame.loc[:, FEATURE_COLUMNS].apply(
        pd.to_numeric,
        errors="coerce",
    )
    values = features.to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise HTTPException(
            status_code=422,
            detail="CSV features must contain only finite numeric values.",
        )

    if (features["Amount"] < 0).any():
        raise HTTPException(
            status_code=422,
            detail="Amount values must be zero or greater.",
        )

    scores = request.app.state.model.predict_proba(features)[:, 1]
    review_mask = scores >= DECISION_THRESHOLD
    review_count = int(review_mask.sum())
    normal_count = len(scores) - review_count
    TRANSACTIONS_SCORED.labels(source="csv", decision="review").inc(review_count)
    TRANSACTIONS_SCORED.labels(source="csv", decision="normal").inc(normal_count)

    # Drift monitoring is best-effort and must not interrupt scoring.
    # Small uploads are insufficient for reliable distribution analysis.
    if len(features) >= 100:
        try:
            drift_result = detect_drift_from_baseline(
                request.app.state.drift_baseline,
                features,
            )

            statuses = [
                item["status"]
                for item in drift_result["features"].values()
            ]
            overall_status = (
                "high" if "high" in statuses
                else "warning" if "warning" in statuses
                else "stable"
            )

            for feature, result in drift_result["features"].items():
                psi = result["psi"]

                # Preserve +Inf as an internal high-drift result while
                # keeping exported Prometheus gauge values finite.
                DRIFT_PSI.labels(feature=feature).set(
                    min(psi, 1_000_000.0)
                )

            DRIFT_ANALYSES.labels(status=overall_status).inc()
            DRIFT_LAST_ANALYSIS.set(time.time())

        except Exception:
            # Invalidate the previous analysis so stale PSI readings
            # cannot be interpreted as the latest monitoring result.
            DRIFT_LAST_ANALYSIS.set(0)
            DRIFT_ANALYSES.labels(status="error").inc()
            logging.getLogger(__name__).exception(
                "CSV drift monitoring failed."
            )

    output = frame.copy()
    output["risk_score"] = scores
    output["decision_threshold"] = DECISION_THRESHOLD
    output["decision"] = np.where(review_mask, "review", "normal")
    output["model_run_id"] = request.app.state.model_run_id

    buffer = StringIO()
    output.to_csv(buffer, index=False)

    return StreamingResponse(
        iter([buffer.getvalue()]),
        media_type="text/csv",
        headers={
            "Content-Disposition": (
                'attachment; filename="fraud_risk_predictions.csv"'
            ),
            "X-Processed-Rows": str(len(output)),
            "X-Review-Count": str(review_count),
            "X-Normal-Count": str(len(output) - review_count),
        },
    )
