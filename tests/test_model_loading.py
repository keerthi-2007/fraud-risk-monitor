from types import SimpleNamespace
from unittest.mock import patch

import pytest

from backend.main import app, lifespan, DECISION_THRESHOLD


@pytest.fixture
def anyio_backend():
    return "asyncio"


def evaluation_record(source_run, threshold=DECISION_THRESHOLD, status="FINISHED"):
    return SimpleNamespace(
        info=SimpleNamespace(status=status),
        data=SimpleNamespace(
            params={
                "source_model_run_id": source_run,
                "decision_threshold": str(threshold),
            }
        ),
    )


@pytest.mark.anyio
async def test_load_model_from_run_uri(monkeypatch):
    monkeypatch.setenv("MODEL_URI", "runs:/test-run/model")

    with patch("backend.main.mlflow.set_tracking_uri"), \
         patch("backend.main.mlflow.tracking.MlflowClient") as client, \
         patch("backend.main.mlflow.sklearn.load_model") as load_model:

        client.return_value.get_run.return_value = evaluation_record(
            "test-run"
        )

        async with lifespan(app):
            assert app.state.model_run_id == "test-run"
            assert app.state.model_uri == "runs:/test-run/model"
            load_model.assert_called_once_with("runs:/test-run/model")


@pytest.mark.anyio
async def test_load_registered_model_version(monkeypatch):
    monkeypatch.setenv("MODEL_URI", "models:/FraudRiskClassifier/1")

    with patch("backend.main.mlflow.set_tracking_uri"), \
         patch("backend.main.mlflow.tracking.MlflowClient") as client, \
         patch("backend.main.mlflow.sklearn.load_model") as load_model:

        client.return_value.get_model_version.return_value = (
            SimpleNamespace(run_id="registered-source-run")
        )
        client.return_value.get_run.return_value = evaluation_record(
            "registered-source-run"
        )

        async with lifespan(app):
            assert app.state.model_run_id == "registered-source-run"
            assert app.state.model_uri == "models:/FraudRiskClassifier/1"
            load_model.assert_called_once_with(
                "runs:/registered-source-run/model"
            )


@pytest.mark.anyio
async def test_reject_unsupported_model_uri(monkeypatch):
    monkeypatch.setenv("MODEL_URI", "invalid:model")

    with patch("backend.main.mlflow.set_tracking_uri"), \
         patch("backend.main.mlflow.sklearn.load_model") as load_model:

        with pytest.raises(ValueError, match="Unsupported MLflow model URI"):
            async with lifespan(app):
                pass

        load_model.assert_not_called()


@pytest.mark.anyio
async def test_reject_unapproved_model_run(monkeypatch):
    monkeypatch.setenv("MODEL_URI", "runs:/different-run/model")

    with patch("backend.main.mlflow.set_tracking_uri"), \
         patch("backend.main.mlflow.tracking.MlflowClient") as client, \
         patch("backend.main.mlflow.sklearn.load_model") as load_model:

        client.return_value.get_run.return_value = evaluation_record(
            "approved-run"
        )

        with pytest.raises(ValueError, match="does not match"):
            async with lifespan(app):
                pass

        load_model.assert_not_called()


@pytest.mark.anyio
async def test_reject_wrong_decision_threshold(monkeypatch):
    monkeypatch.setenv("MODEL_URI", "runs:/test-run/model")

    with patch("backend.main.mlflow.set_tracking_uri"), \
         patch("backend.main.mlflow.tracking.MlflowClient") as client, \
         patch("backend.main.mlflow.sklearn.load_model") as load_model:

        client.return_value.get_run.return_value = evaluation_record(
            "test-run", threshold=0.5
        )

        with pytest.raises(ValueError, match="Decision threshold"):
            async with lifespan(app):
                pass

        load_model.assert_not_called()
