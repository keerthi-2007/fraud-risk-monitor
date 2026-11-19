"""Shared pytest fixtures."""

import pytest

from backend import feedback_crypto


@pytest.fixture(autouse=True)
def no_ambient_encryption_key(monkeypatch, tmp_path):
    """Keep tests independent of a developer's real .env encryption key."""
    monkeypatch.delenv(feedback_crypto.KEY_ENV, raising=False)
    monkeypatch.setattr(feedback_crypto, "ENV_FILE", tmp_path / "absent.env")
