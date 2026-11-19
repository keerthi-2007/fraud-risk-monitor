"""Encryption at rest for user-submitted feedback transactions.

Feedback records contain real transaction feature vectors. When the
``FEEDBACK_ENCRYPTION_KEY`` environment variable holds a Fernet key
(AES-128-CBC + HMAC-SHA256, from the ``cryptography`` package), the
feature vector is stored as an encrypted token instead of plaintext:

    {"...": "...", "features_encrypted": "<token>", "encryption": "fernet-v1"}

Readers call :func:`record_features`, which transparently handles both
encrypted and legacy plaintext records. Generate a key with::

    python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken

KEY_ENV = "FEEDBACK_ENCRYPTION_KEY"
SCHEME = "fernet-v1"

logger = logging.getLogger(__name__)


class FeedbackKeyMissing(RuntimeError):
    """Encrypted feedback exists but no decryption key is configured."""


ENV_FILE = Path(__file__).resolve().parents[1] / ".env"


def _configured_key() -> str:
    """Key from the environment, else from the project's git-ignored .env.

    The .env fallback lets host-side retraining scripts decrypt the same
    feedback the container encrypted, without exporting the key manually.
    """
    key = os.getenv(KEY_ENV, "").strip()
    if key or not ENV_FILE.is_file():
        return key
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        name, _, value = line.partition("=")
        if name.strip() == KEY_ENV:
            return value.strip().strip("'\"")
    return ""


def _fernet() -> Fernet | None:
    key = _configured_key()
    if not key:
        return None
    try:
        return Fernet(key.encode("ascii"))
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{KEY_ENV} is not a valid Fernet key.") from exc


def encryption_enabled() -> bool:
    return _fernet() is not None


def protect_features(features: dict[str, float]) -> dict:
    """Return the record fields that store ``features``.

    Encrypted when a key is configured, otherwise plaintext (with a warning).
    """
    fernet = _fernet()
    if fernet is None:
        logger.warning("%s not set; feedback features stored unencrypted.", KEY_ENV)
        return {"features": features}

    payload = json.dumps(features, sort_keys=True, allow_nan=False).encode("utf-8")
    return {
        "features_encrypted": fernet.encrypt(payload).decode("ascii"),
        "encryption": SCHEME,
    }


def record_features(record: dict) -> dict[str, float]:
    """Return the plaintext feature dict from a stored feedback record."""
    if "features" in record:
        return record["features"]

    if record.get("encryption") != SCHEME or "features_encrypted" not in record:
        raise ValueError("Feedback record has no readable features.")

    fernet = _fernet()
    if fernet is None:
        raise FeedbackKeyMissing(
            f"Feedback is encrypted; set {KEY_ENV} to read it."
        )
    try:
        payload = fernet.decrypt(record["features_encrypted"].encode("ascii"))
    except InvalidToken as exc:
        raise ValueError("Feedback record failed decryption/integrity check.") from exc
    return json.loads(payload)
