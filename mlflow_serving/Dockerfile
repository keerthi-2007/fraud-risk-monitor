# Dedicated MLflow model-serving image (MLflow APIfication).
# Serves the registered model with MLflow's own scoring server:
#   POST /invocations   GET /ping   GET /health   GET /version
FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MLFLOW_MODEL_URI=models:/FraudRiskClassifier/1 \
    MLFLOW_SERVE_PORT=5001

WORKDIR /app

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

RUN groupadd --gid 1000 appgroup \
    && useradd --uid 1000 --gid appgroup \
       --create-home --home-dir /home/appuser appuser
ENV HOME=/home/appuser
USER appuser

EXPOSE 5001

HEALTHCHECK --interval=15s --timeout=5s --start-period=60s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:5001/ping')"

# --env-manager local: the image already holds the exact pinned dependencies.
CMD ["sh", "-c", "mlflow models serve --model-uri \"$MLFLOW_MODEL_URI\" --env-manager local --host 0.0.0.0 --port \"$MLFLOW_SERVE_PORT\""]
