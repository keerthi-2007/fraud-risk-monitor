"""Deploy a verified MLflow candidate with health checks and rollback."""

import argparse
import json
import os
import stat
import subprocess
import time
import urllib.request
from pathlib import Path

from scripts.promotion_controller import verify_candidate
from scripts.deployment_env import save_active_model, restore_env_snapshot

ROOT = Path(__file__).resolve().parents[1]
COMPOSE = ["docker", "compose"]


def health():
    """Call the backend /health endpoint."""
    with urllib.request.urlopen(
        "http://127.0.0.1:8000/health", timeout=3
    ) as response:
        return json.load(response)


def wait_for_model(expected_id, timeout=90):
    """Wait until the backend reports the expected model run ID or time out."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            result = health()
            if (
                result.get("status") == "ok"
                and result.get("model_run_id") == expected_id
            ):
                return
        except (OSError, ValueError):
            pass
        time.sleep(2)
    raise RuntimeError(f"Backend failed health verification: {expected_id}")


def current_configuration():
    """Read the running backend container's model settings for rollback."""
    container_id = subprocess.check_output(
        ["docker", "compose", "ps", "-q", "backend"],
        cwd=ROOT, text=True
    ).strip()

    if not container_id:
        raise RuntimeError("Backend container is not running.")

    raw = subprocess.check_output(
        ["docker", "inspect", "--format", "{{json .Config.Env}}",
         container_id],
        text=True
    )

    environment = dict(item.split("=", 1) for item in json.loads(raw))
    required = ["MODEL_URI", "EVALUATION_RUN_ID", "DECISION_THRESHOLD"]

    if any(not environment.get(key) for key in required):
        raise RuntimeError("Existing deployment configuration incomplete.")

    return environment


def deploy(model_uri, evaluation_id, threshold):
    """Restart the backend with the given model URI, evaluation run and threshold."""
    environment = os.environ.copy()
    environment.update({
        "ACTIVE_MODEL_URI": str(model_uri),
        "ACTIVE_EVALUATION_RUN_ID": str(evaluation_id),
        "ACTIVE_DECISION_THRESHOLD": str(threshold),
    })

    subprocess.run(
        COMPOSE + [
            "up", "-d", "--no-deps", "--force-recreate", "backend"
        ],
        cwd=ROOT,
        env=environment,
        check=True,
    )


def main():
    """CLI: deploy a verified candidate (--deploy), rolling back if health checks fail."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--deploy", action="store_true",
        help="Explicitly authorize a verified candidate deployment"
    )
    args = parser.parse_args()

    verified = verify_candidate()
    if verified is None:
        print("DEPLOYMENT_BLOCKED: No verified candidate.")
        return

    candidate_id, evaluation_id, threshold = verified

    if not args.deploy:
        print("DRY_RUN_ONLY: No deployment performed.")
        return

    previous = current_configuration()
    previous_health = health()

    if previous_health.get("status") != "ok":
        raise RuntimeError("Existing backend is unhealthy.")

    previous_id = previous_health["model_run_id"]

    env_path = ROOT / ".env"
    if not env_path.is_file() or env_path.is_symlink():
        raise RuntimeError("Cannot safely snapshot the local .env file")

    previous_env = env_path.read_bytes()
    previous_mode = stat.S_IMODE(env_path.stat().st_mode)

    print("Previous model:", previous_id)
    print("Candidate model:", candidate_id)

    try:
        deploy(
            f"runs:/{candidate_id}/model",
            evaluation_id,
            threshold,
        )
        wait_for_model(candidate_id)
        save_active_model(
            ROOT / ".env",
            f"runs:/{candidate_id}/model",
            evaluation_id,
            threshold,
        )
        print("DEPLOYMENT_SUCCESS:", candidate_id)

    except BaseException:
        print("DEPLOYMENT_FAILED: Attempting rollback.")

        rollback_error = None
        try:
            deploy(
                previous["MODEL_URI"],
                previous["EVALUATION_RUN_ID"],
                previous["DECISION_THRESHOLD"],
            )
            wait_for_model(previous_id)
        except BaseException as exc:
            rollback_error = exc

        try:
            restore_env_snapshot(
                env_path, previous_env, previous_mode
            )
        except BaseException:
            print("CRITICAL: Saved .env restoration failed.")
            raise

        if rollback_error is not None:
            print("CRITICAL: ROLLBACK FAILED. Manual recovery required.")
            raise rollback_error

        print("ROLLBACK_SUCCESS:", previous_id)
        raise


if __name__ == "__main__":
    main()
