FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements-frontend.txt .
RUN pip install --no-cache-dir -r requirements-frontend.txt

COPY frontend ./frontend

# Run Streamlit as an unprivileged user.
RUN groupadd --gid 10001 appgroup \
    && useradd --uid 10001 --gid appgroup \
       --create-home --home-dir /home/appuser appuser

ENV HOME=/home/appuser
USER appuser

EXPOSE 8501

CMD ["streamlit", "run", "frontend/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
