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
