from datetime import datetime, timezone
import json
import math
import os
import secrets
from pathlib import Path
from threading import Lock
from typing import Literal
from uuid import uuid4

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from prometheus_client import Counter


router = APIRouter()
FEEDBACK_SUBMISSIONS = Counter(
    "fraud_feedback_submissions_total",
    "Feedback records successfully saved.",
    ["corrected_label", "prediction_was_correct"],
)
WRITE_LOCK = Lock()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEEDBACK_DIR = PROJECT_ROOT / "data" / "feedback"
FEEDBACK_FILE = FEEDBACK_DIR / "feedback.jsonl"

FEATURE_NAMES = [f"V{i}" for i in range(1, 29)] + ["Amount"]


class FeedbackSubmission(BaseModel):
    model_config = ConfigDict(extra="forbid")

    features: dict[str, float]
    risk_score: float = Field(ge=0, le=1)
    decision_threshold: float = Field(ge=0, le=1)
    predicted_decision: Literal["normal", "review"]
    corrected_label: Literal["legitimate", "fraud"]
    model_run_id: str | None = Field(default=None, max_length=100)
    source_file: str | None = Field(default=None, max_length=255)
    row_number: int | None = Field(default=None, ge=1)


@router.post("/feedback", status_code=201)
def save_feedback(
    submission: FeedbackSubmission,
    x_feedback_token: str | None = Header(default=None),
):
    expected_token = os.getenv("FEEDBACK_TOKEN")
    if (
        not expected_token
        or not x_feedback_token
        or not secrets.compare_digest(x_feedback_token, expected_token)
    ):
        raise HTTPException(
            status_code=403,
            detail="Feedback access denied.",
        )

    received_features = set(submission.features)
    expected_features = set(FEATURE_NAMES)

    if received_features != expected_features:
        raise HTTPException(
            status_code=422,
            detail={
                "missing_features": sorted(expected_features - received_features),
                "unexpected_features": sorted(received_features - expected_features),
            },
        )

    if any(not math.isfinite(value) for value in submission.features.values()):
        raise HTTPException(status_code=422, detail="Feature values must be finite numbers.")

    if submission.features["Amount"] < 0:
        raise HTTPException(status_code=422, detail="Amount cannot be negative.")

    predicted_label = (
        "fraud" if submission.predicted_decision == "review" else "legitimate"
    )

    record = {
        "feedback_id": str(uuid4()),
        "received_at": datetime.now(timezone.utc).isoformat(),
        "model_run_id": submission.model_run_id,
        "source_file": submission.source_file,
        "row_number": submission.row_number,
        "risk_score": submission.risk_score,
        "decision_threshold": submission.decision_threshold,
        "predicted_decision": submission.predicted_decision,
        "corrected_label": submission.corrected_label,
        "prediction_was_correct": predicted_label == submission.corrected_label,
        "features": {
            name: submission.features[name]
            for name in FEATURE_NAMES
        },
    }

    try:
        FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
        with WRITE_LOCK, FEEDBACK_FILE.open("a", encoding="utf-8") as output:
            output.write(json.dumps(record, allow_nan=False) + "\n")
    except OSError as exc:
        raise HTTPException(
            status_code=500,
            detail="Feedback could not be saved locally.",
        ) from exc

    FEEDBACK_SUBMISSIONS.labels(
        corrected_label=submission.corrected_label,
        prediction_was_correct=str(record["prediction_was_correct"]).lower(),
    ).inc()

    return {
        "status": "stored",
        "feedback_id": record["feedback_id"],
        "prediction_was_correct": record["prediction_was_correct"],
    }
