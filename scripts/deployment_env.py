"""Persist an approved model selection without altering unrelated .env values."""

import math
import os
import re
import stat
import tempfile
from pathlib import Path


def save_active_model(path: Path, model_uri: str,
                      evaluation_id: str, threshold: float) -> None:
    path = Path(path)

    if not path.is_file() or path.is_symlink():
        raise ValueError("Expected an existing, regular .env file")

    if not re.fullmatch(r"runs:/[0-9a-f]{32}/model", model_uri):
        raise ValueError("Invalid candidate MLflow model URI")

    if not re.fullmatch(r"[0-9a-f]{32}", evaluation_id):
        raise ValueError("Invalid MLflow evaluation run ID")

    threshold = float(threshold)
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Invalid decision threshold")

    settings = {
        "ACTIVE_MODEL_URI": model_uri,
        "ACTIVE_EVALUATION_RUN_ID": evaluation_id,
        "ACTIVE_DECISION_THRESHOLD": format(threshold, ".17g"),
    }

    original = path.read_text(encoding="utf-8")
    lines = original.splitlines(keepends=True)

    for key, value in settings.items():
        pattern = re.compile(rf"^\s*(?:export\s+)?{key}\s*=")
        matches = [
            i for i, line in enumerate(lines)
            if pattern.match(line)
        ]

        if len(matches) > 1:
            raise ValueError(f"Duplicate .env setting: {key}")

        if matches:
            lines[matches[0]] = f"{key}={value}\n"
        else:
            if lines and not lines[-1].endswith("\n"):
                lines[-1] += "\n"
            lines.append(f"{key}={value}\n")

    fd, temp_name = tempfile.mkstemp(
        prefix=".env-update-", dir=path.parent
    )

    try:
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            output.writelines(lines)
            output.flush()
            os.fsync(output.fileno())

        os.chmod(temp_name, stat.S_IMODE(path.stat().st_mode))
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def restore_env_snapshot(
    path: Path, original: bytes, original_mode: int
) -> None:
    """Atomically restore the exact pre-deployment .env contents."""
    path = Path(path)

    if path.is_symlink() or (
        path.exists() and not path.is_file()
    ):
        raise ValueError("Unsafe .env restoration target")

    if path.is_file():
        existing_mode = stat.S_IMODE(path.stat().st_mode)
        if (
            path.read_bytes() == original
            and existing_mode == original_mode
        ):
            return

    fd, temp_name = tempfile.mkstemp(
        prefix=".env-rollback-", dir=path.parent
    )

    try:
        with os.fdopen(fd, "wb") as output:
            output.write(original)
            output.flush()
            os.fsync(output.fileno())

        os.chmod(temp_name, original_mode)
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)
