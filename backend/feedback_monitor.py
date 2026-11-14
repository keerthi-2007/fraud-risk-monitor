"""Prometheus metrics for user-reported transaction feedback."""

from threading import Lock

from prometheus_client import Gauge

from backend.feedback_metrics import load_feedback_summary

FEEDBACK_CONFUSION = Gauge(
    "fraud_reported_feedback_confusion_count",
    "Unique feedback records by model and reported confusion category.",
    ["model_run_id", "category"],
)

FEEDBACK_DUPLICATES = Gauge(
    "fraud_reported_feedback_duplicates_count",
    "Repeated source references excluded from feedback summaries.",
    ["model_run_id"],
)

FEEDBACK_RECORDS = Gauge(
    "fraud_reported_feedback_records_count",
    "Unique user-reported feedback records by model version.",
    ["model_run_id"],
)

REFRESH_LOCK = Lock()


def refresh_feedback_metrics(path):
    """Refresh metrics from stored feedback without changing the file."""
    with REFRESH_LOCK:
        summaries = load_feedback_summary(path)

        # Read and validate before replacing exported measurements.
        measurements = []

        for model_id, result in summaries.items():
            measurements.append((model_id, result))

        FEEDBACK_CONFUSION.clear()
        FEEDBACK_DUPLICATES.clear()
        FEEDBACK_RECORDS.clear()

        categories = {
            "tp": "true_positive",
            "fp": "false_positive",
            "tn": "true_negative",
            "fn": "false_negative",
        }

        for model_id, result in measurements:
            FEEDBACK_RECORDS.labels(
                model_run_id=model_id
            ).set(result["reported_feedback_count"])

            FEEDBACK_DUPLICATES.labels(
                model_run_id=model_id
            ).set(result["duplicate_references_excluded"])

            for category, key in categories.items():
                FEEDBACK_CONFUSION.labels(
                    model_run_id=model_id,
                    category=category,
                ).set(result[key])

        return summaries
