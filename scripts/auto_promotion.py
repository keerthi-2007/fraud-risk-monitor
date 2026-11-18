"""Bridge verified candidate evaluation to safe model deployment."""

import argparse
import subprocess
import sys

from scripts.deploy_candidate import ROOT, health
from scripts.promotion_controller import verify_candidate


def saved_model_uri():
    """Read only the active model URI from the local environment file."""
    path = ROOT / ".env"

    if not path.is_file():
        raise RuntimeError("Local .env file is missing")

    matches = []

    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("ACTIVE_MODEL_URI="):
            matches.append(line.split("=", 1)[1].strip())

    if len(matches) > 1:
        raise RuntimeError("Duplicate ACTIVE_MODEL_URI settings")

    return matches[0] if matches else None


def process_promotion(*, deploy=False):
    verified = verify_candidate()

    if verified is None:
        print("NO_VERIFIED_CANDIDATE")
        return "no_candidate"

    candidate_id, evaluation_id, threshold = verified
    target_uri = f"runs:/{candidate_id}/model"

    current = health()

    if current.get("status") != "ok":
        raise RuntimeError("Existing backend is unhealthy")

    if current.get("model_run_id") == candidate_id:
        if saved_model_uri() != target_uri:
            raise RuntimeError(
                "Live model and saved configuration disagree"
            )

        print("ALREADY_DEPLOYED:", candidate_id)
        return "already_deployed"

    if not deploy:
        print("VERIFIED_CANDIDATE_READY")
        print("Candidate:", candidate_id)
        print("Evaluation:", evaluation_id)
        print("Threshold:", threshold)
        print("Deployment: NOT PERFORMED")
        return "ready"

    subprocess.run(
        [sys.executable, "-m", "scripts.deploy_candidate", "--deploy"],
        cwd=ROOT,
        check=True,
    )

    updated = health()

    if (
        updated.get("status") != "ok"
        or updated.get("model_run_id") != candidate_id
    ):
        raise RuntimeError(
            "Post-deployment model verification failed"
        )

    if saved_model_uri() != target_uri:
        raise RuntimeError(
            "Promoted model configuration was not persisted"
        )

    print("AUTO_PROMOTION_SUCCESS:", candidate_id)
    return "deployed"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--deploy",
        action="store_true",
        help="Enable actual deployment of a verified candidate",
    )
    args = parser.parse_args()
    process_promotion(deploy=args.deploy)


if __name__ == "__main__":
    main()
