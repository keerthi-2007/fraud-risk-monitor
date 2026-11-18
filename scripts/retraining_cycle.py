"""Orchestrate automatic candidate retraining and MLflow evaluation."""

import argparse
import logging
import os
import subprocess
import sys
import time

from scripts.auto_retrain import ROOT, run_once

LOG = logging.getLogger("retraining_cycle")


def process_cycle(min_feedback=20, min_fraud=3, dry_run=False):
    result = run_once(
        min_feedback=min_feedback,
        min_fraud=min_fraud,
        dry_run=dry_run,
    )

    # Retry evaluation after an interrupted previous attempt.
    if not dry_run and result in ("trained_candidate", "already_processed"):
        LOG.info("Checking candidate MLflow evaluation")
        subprocess.run(
            [sys.executable, "-m", "scripts.prepare_candidate_evaluation"],
            cwd=ROOT,
            check=True,
        )
        LOG.info("Checking candidate promotion eligibility")
        promotion_command = [
            sys.executable, "-m", "scripts.auto_promotion"
        ]

        # Explicit opt-in only. Disabled by default.
        if os.getenv("FRAUD_AUTO_DEPLOY", "0") == "1":
            promotion_command.append("--deploy")

        subprocess.run(
            promotion_command,
            cwd=ROOT,
            check=True,
        )

    return result


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--once", action="store_true")
    mode.add_argument("--watch", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--min-feedback", type=int, default=20)
    parser.add_argument("--min-fraud", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if min(args.poll_seconds, args.min_feedback, args.min_fraud) < 1:
        parser.error("Numeric options must be positive")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if args.once:
        print(process_cycle(
            args.min_feedback, args.min_fraud, args.dry_run
        ))
        return

    LOG.info("Retraining and evaluation watcher started")

    while True:
        try:
            process_cycle(
                args.min_feedback, args.min_fraud, args.dry_run
            )
        except Exception:
            LOG.exception("Retraining/evaluation cycle failed")
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
