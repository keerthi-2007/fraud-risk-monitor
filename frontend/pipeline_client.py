"""Read the recorded DVC pipeline snapshot through FastAPI."""

import requests


def fetch_pipeline_status(api_url, token, timeout=5):
    """Return the recorded pipeline snapshot or raise an error."""
    if not token:
        raise ValueError("Monitoring token is not configured.")

    response = requests.get(
        f"{api_url.rstrip('/')}/monitoring/pipeline-status",
        headers={"X-Monitoring-Token": token},
        timeout=timeout,
    )
    response.raise_for_status()

    snapshot = response.json()

    if (
        not isinstance(snapshot, dict)
        or snapshot.get("schema_version") != 1
        or not isinstance(snapshot.get("generated_at"), str)
        or not isinstance(snapshot.get("overall_up_to_date"), bool)
        or not isinstance(snapshot.get("stages"), list)
    ):
        raise ValueError("Invalid pipeline status response.")

    return snapshot



def fetch_retraining_status(api_url, token, timeout=5):
    """Fetch authenticated, read-only retraining status."""
    if not token:
        raise ValueError("Monitoring token is not configured.")

    response = requests.get(
        f"{api_url.rstrip('/')}/monitoring/retraining-status",
        headers={"X-Monitoring-Token": token},
        timeout=timeout,
    )
    response.raise_for_status()

    snapshot = response.json()

    required_counts = (
        "feedback_received",
        "not_approved",
        "invalid",
        "eligible",
        "eligible_fraud",
        "eligible_legitimate",
        "required_feedback",
        "required_per_class",
    )

    if (
        not isinstance(snapshot, dict)
        or snapshot.get("schema_version") != 1
        or not isinstance(snapshot.get("checked_at"), str)
        or type(snapshot.get("threshold_met")) is not bool
        or any(
            type(snapshot.get(key)) is not int
            or snapshot[key] < 0
            for key in required_counts
        )
    ):
        raise ValueError("Invalid retraining status response.")

    return snapshot
