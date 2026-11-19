from fastapi.testclient import TestClient

from backend.main import app, DECISION_THRESHOLD


class FakeModel:
    def predict_proba(self, frame):
        import numpy as np

        scores = np.where(frame["Amount"].to_numpy() >= 100, 0.95, 0.10)
        return np.column_stack((1 - scores, scores))


def make_client():
    app.state.model = FakeModel()
    app.state.model_run_id = "test-model"
    return TestClient(app)


def valid_features(amount=50):
    return {
        **{f"V{i}": 0.0 for i in range(1, 29)},
        "Amount": amount,
    }


def test_health():
    response = make_client().get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model_run_id": "test-model",
    }


def test_normal_prediction():
    response = make_client().post(
        "/predict", json={"features": valid_features(50)}
    )
    assert response.status_code == 200
    result = response.json()
    assert result["decision"] == "normal"
    assert result["risk_score"] == 0.10
    assert result["decision_threshold"] == DECISION_THRESHOLD


def test_review_prediction():
    response = make_client().post(
        "/predict", json={"features": valid_features(150)}
    )
    assert response.status_code == 200
    assert response.json()["decision"] == "review"


def test_missing_feature_rejected():
    features = valid_features()
    del features["V1"]

    response = make_client().post("/predict", json={"features": features})
    assert response.status_code == 422
    assert "V1" in response.json()["detail"]["missing_features"]


def test_negative_amount_rejected():
    response = make_client().post(
        "/predict", json={"features": valid_features(-10)}
    )
    assert response.status_code == 422


def test_extra_feature_rejected():
    features = valid_features()
    features["unexpected"] = 1.0

    response = make_client().post("/predict", json={"features": features})
    assert response.status_code == 422
    assert "unexpected" in response.json()["detail"]["unexpected_features"]


def csv_upload(client, content, filename="transactions.csv"):
    return client.post(
        "/predict/csv",
        files={"file": (filename, content, "text/csv")},
    )


def test_csv_predictions():
    import pandas as pd

    frame = pd.DataFrame([
        valid_features(50),
        valid_features(150),
    ])
    response = csv_upload(
        make_client(),
        frame.to_csv(index=False).encode(),
    )

    assert response.status_code == 200
    assert response.headers["X-Processed-Rows"] == "2"
    assert response.headers["X-Review-Count"] == "1"

    from io import StringIO
    result = pd.read_csv(StringIO(response.text))
    assert result["decision"].tolist() == ["normal", "review"]


def test_csv_missing_column():
    import pandas as pd

    frame = pd.DataFrame([valid_features()])
    frame = frame.drop(columns=["V1"])

    response = csv_upload(
        make_client(),
        frame.to_csv(index=False).encode(),
    )
    assert response.status_code == 422
    assert "V1" in response.json()["detail"]["missing_features"]


def test_csv_invalid_extension():
    response = csv_upload(
        make_client(),
        b"invalid content",
        filename="transactions.txt",
    )
    assert response.status_code == 415


def test_csv_negative_amount():
    import pandas as pd

    frame = pd.DataFrame([valid_features(-5)])
    response = csv_upload(
        make_client(),
        frame.to_csv(index=False).encode(),
    )
    assert response.status_code == 422


def test_csv_empty_file():
    response = csv_upload(make_client(), b"")
    assert response.status_code == 400


def test_decision_threshold_boundary():
    import numpy as np

    class BoundaryModel:
        def predict_proba(self, frame):
            score = float(frame["Amount"].iloc[0])
            return np.array([[1 - score, score]])

    client = make_client()
    app.state.model = BoundaryModel()

    for score, expected in [
        (DECISION_THRESHOLD - 0.01, "normal"),
        (DECISION_THRESHOLD, "review"),
        (DECISION_THRESHOLD + 0.01, "review"),
    ]:
        response = client.post(
            "/predict",
            json={"features": valid_features(score)},
        )
        assert response.status_code == 200
        assert response.json()["decision"] == expected


def test_csv_validation_failure_metric(monkeypatch):
    from backend.main import CSV_VALIDATION_FAILURES

    monkeypatch.setenv("MONITORING_TOKEN", "test-secret")

    counter = CSV_VALIDATION_FAILURES.labels(
        upload_type="single_csv",
        reason="missing_columns",
    )
    before = counter._value.get()

    response = make_client().post(
        "/monitoring/validation-failure",
        json={
            "upload_type": "single_csv",
            "reason": "missing_columns",
        },
        headers={"X-Monitoring-Token": "test-secret"},
    )

    assert response.status_code == 204
    assert counter._value.get() == before + 1


def test_csv_validation_failure_rejects_missing_token(monkeypatch):
    monkeypatch.setenv("MONITORING_TOKEN", "test-secret")

    response = make_client().post(
        "/monitoring/validation-failure",
        json={
            "upload_type": "single_csv",
            "reason": "missing_columns",
        },
    )

    assert response.status_code == 403


def test_csv_validation_failure_rejects_wrong_token(monkeypatch):
    monkeypatch.setenv("MONITORING_TOKEN", "test-secret")

    response = make_client().post(
        "/monitoring/validation-failure",
        json={
            "upload_type": "single_csv",
            "reason": "missing_columns",
        },
        headers={"X-Monitoring-Token": "wrong-secret"},
    )

    assert response.status_code == 403


def test_csv_validation_failure_rejects_invalid_category(monkeypatch):
    monkeypatch.setenv("MONITORING_TOKEN", "test-secret")

    response = make_client().post(
        "/monitoring/validation-failure",
        json={
            "upload_type": "single_csv",
            "reason": "unknown_error",
        },
        headers={"X-Monitoring-Token": "test-secret"},
    )

    assert response.status_code == 422


def test_csv_includes_model_run_id():
    import pandas as pd
    from io import StringIO

    client = make_client()

    frame = pd.DataFrame([
        valid_features(50),
        valid_features(150),
    ])

    response = csv_upload(
        client,
        frame.to_csv(index=False).encode(),
    )

    assert response.status_code == 200

    scored = pd.read_csv(StringIO(response.text))

    assert "model_run_id" in scored.columns
    assert len(scored) == 2
    assert scored["model_run_id"].tolist() == [
        "test-model",
        "test-model",
    ]
