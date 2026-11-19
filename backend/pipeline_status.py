"""Read-only API for the recorded DVC pipeline snapshot."""

import json
import os
import secrets
from pathlib import Path

from fastapi import APIRouter, Header, HTTPException

router = APIRouter()

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_PATH = ROOT / "monitoring/pipeline_status.json"


def load_pipeline_snapshot(path=SNAPSHOT_PATH):
    """Read and validate a previously generated pipeline snapshot."""
    path = Path(path)

    try:
        with path.open(encoding="utf-8") as handle:
            snapshot = json.load(handle)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail="Pipeline snapshot is unavailable.",
        ) from exc
    except (OSError, ValueError) as exc:
        raise HTTPException(
            status_code=503,
            detail="Pipeline snapshot could not be read.",
        ) from exc

    if (
        not isinstance(snapshot, dict)
        or snapshot.get("schema_version") != 1
        or not isinstance(snapshot.get("generated_at"), str)
        or not isinstance(snapshot.get("overall_up_to_date"), bool)
        or not isinstance(snapshot.get("stages"), list)
    ):
        raise HTTPException(
            status_code=503,
            detail="Pipeline snapshot has an invalid structure.",
        )

    return snapshot


@router.get("/monitoring/pipeline-status")
def get_pipeline_status(
    x_monitoring_token: str | None = Header(default=None),
):
    """Return the recorded DVC pipeline status snapshot (token-protected)."""
    expected_token = os.getenv("MONITORING_TOKEN")

    if (
        not expected_token
        or not x_monitoring_token
        or not secrets.compare_digest(
            x_monitoring_token, expected_token
        )
    ):
        raise HTTPException(
            status_code=403,
            detail="Monitoring access denied.",
        )

    return load_pipeline_snapshot()
