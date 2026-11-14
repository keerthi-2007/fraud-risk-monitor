"""Tests for the frontend pipeline-status API client."""

from unittest.mock import Mock, patch

import pytest
import requests

from frontend.pipeline_client import fetch_pipeline_status


def valid_snapshot():
    return {
        "schema_version": 1,
        "generated_at": "2026-11-14T15:44:29+00:00",
        "overall_up_to_date": True,
        "stages": [
            {
                "name": "prepare_data",
                "up_to_date": True,
            }
        ],
    }


def test_fetch_valid_snapshot():
    response = Mock()
    response.json.return_value = valid_snapshot()

    with patch(
        "frontend.pipeline_client.requests.get",
        return_value=response,
    ) as get:
        result = fetch_pipeline_status(
            "http://backend:8000/",
            "test-token",
        )

    assert result["overall_up_to_date"] is True
    get.assert_called_once_with(
        "http://backend:8000/monitoring/pipeline-status",
        headers={"X-Monitoring-Token": "test-token"},
        timeout=5,
    )
    response.raise_for_status.assert_called_once()


def test_missing_token_rejected():
    with patch("frontend.pipeline_client.requests.get") as get:
        with pytest.raises(ValueError, match="token"):
            fetch_pipeline_status("http://backend:8000", "")

    get.assert_not_called()


def test_http_error_propagated():
    response = Mock()
    response.raise_for_status.side_effect = (
        requests.exceptions.HTTPError("403 Forbidden")
    )

    with patch(
        "frontend.pipeline_client.requests.get",
        return_value=response,
    ):
        with pytest.raises(requests.exceptions.HTTPError):
            fetch_pipeline_status(
                "http://backend:8000",
                "wrong-token",
            )


def test_connection_failure_propagated():
    with patch(
        "frontend.pipeline_client.requests.get",
        side_effect=requests.exceptions.ConnectionError(
            "API unavailable"
        ),
    ):
        with pytest.raises(requests.exceptions.ConnectionError):
            fetch_pipeline_status(
                "http://backend:8000",
                "test-token",
            )


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {},
        {"schema_version": 2},
        {
            "schema_version": 1,
            "generated_at": "recorded-time",
            "overall_up_to_date": "yes",
            "stages": [],
        },
        {
            "schema_version": 1,
            "generated_at": "recorded-time",
            "overall_up_to_date": True,
            "stages": "not-a-list",
        },
    ],
)
def test_invalid_snapshot_rejected(payload):
    response = Mock()
    response.json.return_value = payload

    with patch(
        "frontend.pipeline_client.requests.get",
        return_value=response,
    ):
        with pytest.raises(ValueError, match="Invalid pipeline"):
            fetch_pipeline_status(
                "http://backend:8000",
                "test-token",
            )
