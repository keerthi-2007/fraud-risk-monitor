import hashlib
import json

from scripts.auto_retrain import FEATURES, read_feedback, load_approvals


def make_record():
    return {
        "feedback_id": "test-feedback-001",
        "corrected_label": "fraud",
        "features": {name: 1.0 for name in FEATURES},
    }


def make_approval(record):
    digest = hashlib.sha256(
        json.dumps(
            record, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()

    return {
        "feedback_id": record["feedback_id"],
        "record_sha256": digest,
        "approved_label": "fraud",
        "verified_by": "test-reviewer",
        "evidence_reference": "test-evidence",
    }


def test_unapproved_feedback_excluded(tmp_path):
    path = tmp_path / "feedback.jsonl"
    path.write_text(json.dumps(make_record()) + "\n")

    frame, stats = read_feedback(path, set(), approved={})

    assert frame.empty
    assert stats["not_approved"] == 1
    assert stats["eligible"] == 0


def test_approved_feedback_accepted(tmp_path):
    record = make_record()
    path = tmp_path / "feedback.jsonl"
    path.write_text(json.dumps(record) + "\n")

    approvals = {record["feedback_id"]: make_approval(record)}
    frame, stats = read_feedback(path, set(), approved=approvals)

    assert len(frame) == 1
    assert frame.iloc[0]["Class"] == 1
    assert stats["eligible"] == 1


def test_modified_feedback_rejected(tmp_path):
    record = make_record()
    approval = make_approval(record)

    record["features"]["Amount"] = 999.0

    path = tmp_path / "feedback.jsonl"
    path.write_text(json.dumps(record) + "\n")

    frame, stats = read_feedback(
        path, set(), approved={record["feedback_id"]: approval}
    )

    assert frame.empty
    assert stats["invalid"] == 1


def test_duplicate_approvals_rejected(tmp_path):
    record = make_record()
    approval = make_approval(record)

    path = tmp_path / "approved_feedback.jsonl"
    path.write_text(
        json.dumps(approval) + "\n" + json.dumps(approval) + "\n"
    )

    try:
        load_approvals(path)
    except ValueError as exc:
        assert "Duplicate feedback approval" in str(exc)
    else:
        raise AssertionError("Duplicate approval was accepted")
