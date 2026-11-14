from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
import pytest
import requests

from backend.main import DECISION_THRESHOLD, FEATURE_COLUMNS, MODEL_RUN_ID

ROOT = Path(__file__).resolve().parents[1]
API_URL = "http://localhost:8000"


def test_deployed_api_matches_mlflow_model():
    test_path = ROOT / "data/processed/splits/test.csv"
    tracking_uri = f"sqlite:///{ROOT / 'mlflow.db'}"

    frame = pd.read_csv(test_path, nrows=1)
    features = frame[FEATURE_COLUMNS]

    mlflow.set_tracking_uri(tracking_uri)
    model = mlflow.sklearn.load_model(
        f"runs:/{MODEL_RUN_ID}/model"
    )

    expected_score = float(model.predict_proba(features)[0, 1])

    response = requests.post(
        f"{API_URL}/predict",
        json={"features": features.iloc[0].to_dict()},
        timeout=30,
    )

    assert response.status_code == 200
    result = response.json()

    assert result["model_run_id"] == MODEL_RUN_ID
    assert result["risk_score"] == pytest.approx(expected_score, abs=1e-10)
    assert result["decision_threshold"] == pytest.approx(
        DECISION_THRESHOLD
    )
    assert result["decision"] == (
        "review" if expected_score >= DECISION_THRESHOLD else "normal"
    )
