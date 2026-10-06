"""Shared configuration for S3-compatible result storage."""

import os


DEFAULT_STORAGE_ENDPOINT_URL = "http://harbor-storage:9000"


def storage_endpoint_url() -> str:
    """Return the configured storage API endpoint, with an in-cluster default."""
    configured = os.environ.get("STORAGE_ENDPOINT_URL", "").strip().rstrip("/")
    return configured or DEFAULT_STORAGE_ENDPOINT_URL
