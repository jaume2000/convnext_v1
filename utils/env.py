"""Minimal .env loader — no python-dotenv dependency."""

from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(path: str | Path | None = None, *, override: bool = False) -> Path | None:
    """Load KEY=VALUE pairs from a .env file into os.environ.

    Searches the repo root (.env next to this package's parent) by default.
    Existing environment variables win unless override=True.
    Returns the path that was loaded, or None if missing.
    """
    if path is None:
        path = Path(__file__).resolve().parents[1] / ".env"
    else:
        path = Path(path)
    if not path.is_file():
        return None
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if not key:
            continue
        if override or key not in os.environ:
            os.environ[key] = value
    return path


def experiment_name(default: str) -> str:
    """EXPERIMENT_NAME from the environment, after loading .env if present."""
    return os.environ.get("EXPERIMENT_NAME", default)
