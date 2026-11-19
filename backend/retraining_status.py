"""Authenticated, read-only retraining monitoring endpoint."""

import json
import os
import secrets
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Header, HTTPException

router = APIRouter()

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_PATH = ROOT / "data/feedback/retraining_monitor.json"

COUNTS = (
    "feedback_received",
    "not_approved",
    "invalid",
    "eligible",
    "eligible_fraud",
    "eligible_legitimate",
    "required_feedback",
    "required_per_class",
)


def load_retraining_status(path=None):
    """Read a monitoring snapshot without running retraining."""
    path = Path(path) if path is not None else SNAPSHOT_PATH

    try:
        snapshot = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise HTTPException(
            status_code=503,
            detail="Retraining monitoring snapshot unavailable.",
        ) from exc

    if not isinstance(snapshot, dict):
        raise HTTPException(
            status_code=503,
            detail="Invalid retraining monitoring snapshot.",
        )

    try:
        timestamp = snapshot["checked_at"]
        parsed_time = datetime.fromisoformat(
            timestamp.replace("Z", "+00:00")
        )

        valid = (
            type(snapshot["schema_version"]) is int
            and snapshot["schema_version"] == 1
            and isinstance(timestamp, str)
            and parsed_time.tzinfo is not None
            and all(
                type(snapshot[key]) is int and snapshot[key] >= 0
                for key in COUNTS
            )
            and snapshot["required_feedback"] >= 1
            and snapshot["required_per_class"] >= 1
            and type(snapshot["threshold_met"]) is bool
            and snapshot["eligible"] == (
                snapshot["eligible_fraud"]
                + snapshot["eligible_legitimate"]
            )
            and snapshot["eligible"] <= snapshot["feedback_received"]
            and snapshot["not_approved"] <= snapshot["feedback_received"]
        )

        expected_threshold = (
            snapshot["eligible"] >= snapshot["required_feedback"]
            and snapshot["eligible_fraud"] >= snapshot["required_per_class"]
            and snapshot["eligible_legitimate"] >= snapshot["required_per_class"]
        )

        valid = valid and (
            snapshot["threshold_met"] == expected_threshold
        )
    except (KeyError, TypeError, ValueError, AttributeError):
        valid = False

    if not valid:
        raise HTTPException(
            status_code=503,
            detail="Invalid retraining monitoring snapshot.",
        )

    # Return only approved monitoring fields.
    return {
        "schema_version": 1,
        "checked_at": snapshot["checked_at"],
        **{key: snapshot[key] for key in COUNTS},
        "threshold_met": snapshot["threshold_met"],
    }


@router.get("/monitoring/retraining-status")
def get_retraining_status(
    x_monitoring_token: str | None = Header(default=None),
):
    """Return feedback counts and retraining eligibility (token-protected)."""
    expected = os.getenv("MONITORING_TOKEN")

    if (
        not expected
        or not x_monitoring_token
        or not secrets.compare_digest(x_monitoring_token, expected)
    ):
        raise HTTPException(
            status_code=403,
            detail="Monitoring access denied.",
        )

    return load_retraining_status()
