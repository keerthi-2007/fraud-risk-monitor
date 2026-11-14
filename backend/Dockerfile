FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY backend ./backend
COPY monitoring/drift_baseline.json ./monitoring/drift_baseline.json

# Run the API as a non-root user matching mounted data ownership.
RUN groupadd --gid 1000 appgroup \
    && useradd --uid 1000 --gid appgroup \
       --create-home --home-dir /home/appuser appuser

ENV HOME=/home/appuser
USER appuser

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
