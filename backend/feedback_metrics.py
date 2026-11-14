"""Summarise user-reported prediction outcomes.

Feedback labels are reported outcomes, not independently verified
ground truth. Metrics must be interpreted with this limitation.
"""

import json
from collections import defaultdict
from pathlib import Path

MIN_METRIC_SAMPLES = 30


def summarise_feedback(records):
    """Aggregate reported outcomes by source model run ID."""
    groups = defaultdict(list)

    for record in records:
        model_run_id = record.get("model_run_id") or "unknown"
        groups[model_run_id].append(record)

    summaries = {}

    for model_run_id, items in groups.items():
        tp = fp = tn = fn = 0
        seen = set()
        duplicates = 0

        for record in items:
            source_file = record.get("source_file")
            row_number = record.get("row_number")

            if source_file is not None and row_number is not None:
                reference = (source_file, row_number)

                if reference in seen:
                    duplicates += 1
                    continue

                seen.add(reference)

            predicted = record.get("predicted_decision")
            reported = record.get("corrected_label")

            if predicted not in {"review", "normal"}:
                raise ValueError("Invalid predicted decision in feedback.")

            if reported not in {"fraud", "legitimate"}:
                raise ValueError("Invalid reported outcome in feedback.")

            if predicted == "review" and reported == "fraud":
                tp += 1
            elif predicted == "review" and reported == "legitimate":
                fp += 1
            elif predicted == "normal" and reported == "legitimate":
                tn += 1
            else:
                fn += 1

        total = tp + fp + tn + fn

        # These are user-reported labels. A sufficiently large
        # sample permits descriptive rates, not an unbiased estimate
        # of production performance or independent verification.
        metrics_available = (
            total >= MIN_METRIC_SAMPLES
            and (tp + fn) > 0
            and (tp + fp) > 0
        )

        summaries[model_run_id] = {
            "reported_feedback_count": total,
            "duplicate_references_excluded": duplicates,
            "true_positive": tp,
            "false_positive": fp,
            "true_negative": tn,
            "false_negative": fn,
            "reported_precision": (
                tp / (tp + fp) if metrics_available else None
            ),
            "reported_recall": (
                tp / (tp + fn) if metrics_available else None
            ),
            "metrics_available": metrics_available,
            "label_source": "user_reported_unverified",
        }

    return summaries


def load_feedback_summary(path: str | Path):
    """Read stored feedback without modifying it."""
    path = Path(path)

    if not path.exists():
        return {}

    records = []

    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid feedback JSON at line {number}."
                ) from exc

    return summarise_feedback(records)
