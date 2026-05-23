"""Application name and version (package metadata + optional ``APP_VERSION`` env)."""

from __future__ import annotations

import os

from app import __version__

APP_NAME = "layer-rag-query"


def get_app_version() -> str:
    """Release id: ``APP_VERSION`` from the environment, else installed package version."""
    override = (os.environ.get("APP_VERSION") or "").strip()
    return override if override else __version__
