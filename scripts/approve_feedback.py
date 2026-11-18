"""Manually approve independently verified feedback for model retraining."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FEEDBACK = ROOT / "data/feedback/feedback.jsonl"
APPROVALS = ROOT / "data/feedback/approved_feedback.jsonl"


def read_records(path):
    if not path.exists():
        return []
    records = []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if line.strip():
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path}, line {number}"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(f"Invalid record at line {number}")
            records.append(record)
    return records


def record_hash(record):
    raw = json.dumps(
        record, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="action", required=True)

    sub.add_parser("list", help="List feedback awaiting review")

    approve = sub.add_parser("approve")
    approve.add_argument("--feedback-id", required=True)
    approve.add_argument(
        "--confirmed-label",
        required=True,
        choices=("fraud", "legitimate"),
    )
    approve.add_argument("--verified-by", required=True)
    approve.add_argument("--evidence-ref", required=True)

    args = parser.parse_args()
    feedback = read_records(FEEDBACK)
    approvals = read_records(APPROVALS)
    approved_ids = {r["feedback_id"] for r in approvals}

    if args.action == "list":
        for record in feedback:
            feedback_id = record.get("feedback_id")
            if feedback_id not in approved_ids:
                print(
                    feedback_id,
                    "| submitted label:",
                    record.get("corrected_label"),
                    "| source:",
                    record.get("source_file"),
                )
        return

    matches = [
        r for r in feedback
        if r.get("feedback_id") == args.feedback_id
    ]
    if len(matches) != 1:
        raise ValueError("Feedback ID missing or not unique.")

    record = matches[0]

    if args.feedback_id in approved_ids:
        raise ValueError("Feedback already approved.")

    if record.get("corrected_label") != args.confirmed_label:
        raise ValueError(
            "Verified label differs from submitted label. "
            "Do not approve this record."
        )

    if not args.verified_by.strip() or not args.evidence_ref.strip():
        raise ValueError("Reviewer and evidence reference are required.")

    approval = {
        "feedback_id": args.feedback_id,
        "record_sha256": record_hash(record),
        "approved_label": args.confirmed_label,
        "verified_by": args.verified_by.strip(),
        "evidence_reference": args.evidence_ref.strip(),
        "approved_at": datetime.now(timezone.utc).isoformat(),
    }

    APPROVALS.parent.mkdir(parents=True, exist_ok=True)
    with APPROVALS.open("a", encoding="utf-8") as output:
        output.write(json.dumps(approval) + "\n")

    print("FEEDBACK_APPROVED:", args.feedback_id)


if __name__ == "__main__":
    main()
