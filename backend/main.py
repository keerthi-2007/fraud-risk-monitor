"""FastAPI inference service for the Fraud Risk Monitor.

Loads the approved MLflow model at startup, exposes scoring, explanation,
feedback and monitoring endpoints, and exports Prometheus metrics.
"""

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
from fastapi.responses import JSONResponse, Response, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel, ConfigDict, Field

ROOT = Path(__file__).resolve().parents[1]
DRIFT_BASELINE_PATH = ROOT / "monitoring/drift_baseline.json"

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("fraud_api")
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

EXPLANATIONS = Counter(
    "fraud_explanations_total",
    "Single-transaction explanations produced by /explain.",
    ["decision"],
)

UNHANDLED_ERRORS = Counter(
    "fraud_api_unhandled_errors_total",
    "Requests that failed with an unexpected server-side exception.",
    ["route"],
)


class ValidationFailureEvent(BaseModel):
    """Upload rejected by frontend validation, reported for monitoring."""
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
    """Request body for /predict and /explain: exactly V1-V28 and Amount."""
    model_config = ConfigDict(extra="forbid")

    features: dict[str, float] = Field(
        description="Exactly V1 through V28 and Amount."
    )


class PredictionResponse(BaseModel):
    """Model score, threshold and review/normal decision for one transaction."""
    model_run_id: str
    risk_score: float
    score_type: Literal["uncalibrated_model_score"]
    decision_threshold: float
    decision: Literal["review", "normal"]


class FeatureContribution(BaseModel):
    """One feature's effect on the score in an explanation."""
    feature: str
    value: float
    reference_value: float
    contribution: float = Field(
        description=(
            "Risk score minus the score obtained when only this feature is "
            "reset to its training-median reference value. Positive values "
            "pushed the transaction towards review."
        )
    )


class ExplanationResponse(PredictionResponse):
    """Prediction plus per-feature contributions from /explain."""
    method: Literal["single_feature_occlusion"]
    reference_score: float = Field(
        description="Score of a transaction whose features all equal the reference values."
    )
    contributions: list[FeatureContribution]


def reference_values_from_baseline(baseline: dict) -> dict[str, float]:
    """Approximate each feature's training median from the drift baseline.

    The baseline stores histogram edges and counts from the training split;
    interpolating the cumulative distribution at 0.5 gives the median.
    """
    references = {}
    for feature in FEATURE_COLUMNS:
        details = baseline["features"][feature]
        edges = np.asarray(details["edges"], dtype=float)
        counts = np.asarray(details["counts"], dtype=float)
        cumulative = np.concatenate(([0.0], np.cumsum(counts) / counts.sum()))
        if len(edges) == 1:
            references[feature] = float(edges[0])
        else:
            references[feature] = float(np.interp(0.5, cumulative, edges))
    return references


def get_reference_values(app_state) -> dict[str, float]:
    """Return cached reference values, loading them lazily if needed."""
    references = getattr(app_state, "reference_values", None)
    if references is None:
        baseline = getattr(app_state, "drift_baseline", None)
        if baseline is None:
            baseline = load_baseline(DRIFT_BASELINE_PATH)
        references = reference_values_from_baseline(baseline)
        app_state.reference_values = references
    return references


def validated_feature_values(payload: PredictRequest) -> dict[str, float]:
    """Validate a single-transaction payload and return ordered features."""
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
    return values


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the approved MLflow model and explanation reference values at startup."""
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

    logger.info("Loading model %s (source run %s)", model_uri, actual_run_id)
    app.state.model = mlflow.sklearn.load_model(load_uri)
    app.state.model_uri = model_uri
    app.state.model_run_id = actual_run_id
    app.state.drift_baseline = load_baseline(DRIFT_BASELINE_PATH)
    app.state.reference_values = reference_values_from_baseline(
        app.state.drift_baseline
    )
    logger.info(
        "Model ready: run=%s threshold=%s evaluation=%s",
        actual_run_id,
        DECISION_THRESHOLD,
        evaluation_run_id,
    )
    yield
    logger.info("Fraud Risk Monitor API shutting down.")


app = FastAPI(
    title="Fraud Risk Monitor API",
    version="0.1.0",
    description="Local API for scoring transaction risk.",
    lifespan=lifespan,
)
@app.middleware("http")
async def observe_http_requests(request, call_next):
    """Record request metrics and turn unexpected exceptions into a logged JSON 500."""
    started_at = perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        route = request.scope.get("route")
        route_path = getattr(route, "path", "unmatched")
        UNHANDLED_ERRORS.labels(route=route_path).inc()
        logger.exception(
            "Unhandled error on %s %s", request.method, request.url.path
        )
        response = JSONResponse(
            status_code=500,
            content={"detail": "Internal server error."},
        )
    route = request.scope.get("route")
    route_path = getattr(route, "path", "unmatched")
    duration = perf_counter() - started_at

    # Access log; health/metrics polling is kept at DEBUG to avoid noise.
    log_level = (
        logging.DEBUG if route_path in {"/metrics", "/health"} else logging.INFO
    )
    logger.log(
        log_level,
        "%s %s -> %s (%.1f ms)",
        request.method,
        route_path,
        response.status_code,
        duration * 1000,
    )

    if route_path != "/metrics":
        HTTP_REQUESTS.labels(
            method=request.method,
            route=route_path,
            status=str(response.status_code),
        ).inc()
        HTTP_REQUEST_DURATION.labels(
            method=request.method,
            route=route_path,
        ).observe(duration)

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
    """Expose Prometheus metrics after refreshing feedback-derived gauges."""
    try:
        refresh_feedback_metrics(feedback_store.FEEDBACK_FILE)
    except (OSError, ValueError):
        logger.exception("Feedback monitoring refresh failed.")

    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.post("/monitoring/validation-failure", status_code=204)
def record_validation_failure(
    event: ValidationFailureEvent,
    x_monitoring_token: str | None = Header(default=None),
):
    """Count a frontend upload rejection (token-protected)."""
    expected_token = os.getenv("MONITORING_TOKEN")
    if (
        not expected_token
        or not x_monitoring_token
        or not secrets.compare_digest(x_monitoring_token, expected_token)
    ):
        logger.warning("Rejected validation-failure event: bad monitoring token.")
        raise HTTPException(status_code=403, detail="Monitoring access denied.")

    CSV_VALIDATION_FAILURES.labels(
        upload_type=event.upload_type,
        reason=event.reason,
    ).inc()
    return Response(status_code=204)


@app.get("/health")
def health(request: Request):
    """Liveness check returning the loaded model's MLflow run ID."""
    return {
        "status": "ok",
        "model_run_id": request.app.state.model_run_id,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictRequest, request: Request):
    """Score one transaction and return its review/normal decision."""
    values = validated_feature_values(payload)

    frame = pd.DataFrame([values], columns=FEATURE_COLUMNS)
    score = float(request.app.state.model.predict_proba(frame)[0, 1])
    decision = "review" if score >= DECISION_THRESHOLD else "normal"
    TRANSACTIONS_SCORED.labels(source="json", decision=decision).inc()
    logger.info("Scored single transaction: decision=%s score=%.4f", decision, score)

    return PredictionResponse(
        model_run_id=request.app.state.model_run_id,
        risk_score=score,
        score_type="uncalibrated_model_score",
        decision_threshold=DECISION_THRESHOLD,
        decision=decision,
    )


@app.post("/explain", response_model=ExplanationResponse)
def explain(payload: PredictRequest, request: Request):
    """Explain one prediction with single-feature occlusion.

    For every feature, the score is recomputed with only that feature reset
    to its training-median reference value. The difference is that feature's
    local contribution. All 30 variants are scored in one batch call, so the
    method is model-agnostic and costs a single extra predict_proba call.
    """
    values = validated_feature_values(payload)
    references = get_reference_values(request.app.state)

    rows = [values]
    for feature in FEATURE_COLUMNS:
        rows.append({**values, feature: references[feature]})
    rows.append(dict(references))

    frame = pd.DataFrame(rows, columns=FEATURE_COLUMNS)
    scores = request.app.state.model.predict_proba(frame)[:, 1]
    score = float(scores[0])
    decision = "review" if score >= DECISION_THRESHOLD else "normal"

    contributions = [
        FeatureContribution(
            feature=feature,
            value=values[feature],
            reference_value=references[feature],
            contribution=score - float(scores[index + 1]),
        )
        for index, feature in enumerate(FEATURE_COLUMNS)
    ]
    contributions.sort(key=lambda item: abs(item.contribution), reverse=True)

    EXPLANATIONS.labels(decision=decision).inc()
    logger.info(
        "Explained transaction: decision=%s top_feature=%s",
        decision,
        contributions[0].feature,
    )

    return ExplanationResponse(
        model_run_id=request.app.state.model_run_id,
        risk_score=score,
        score_type="uncalibrated_model_score",
        decision_threshold=DECISION_THRESHOLD,
        decision=decision,
        method="single_feature_occlusion",
        reference_score=float(scores[-1]),
        contributions=contributions,
    )


@app.post("/predict/csv")
async def predict_csv(
    request: Request,
    file: Annotated[
        UploadFile,
        File(description="CSV containing V1 through V28 and Amount."),
    ],
):
    """Score every row of an uploaded CSV and run best-effort drift analysis."""
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
        logger.warning("Rejected CSV %r: unreadable (%s)", file.filename, exc)
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
            logger.exception("CSV drift monitoring failed.")

    logger.info(
        "Scored CSV %r: rows=%d review=%d normal=%d",
        file.filename,
        len(scores),
        review_count,
        normal_count,
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
