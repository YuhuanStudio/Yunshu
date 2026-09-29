"""The Yunshu version.

One source: ``pyproject.toml``. An installed package reads it from its metadata; a source
checkout reads the pyproject next to ``python/`` (an editable install's metadata goes stale
until it is reinstalled, so the checkout wins when both exist).
"""

from __future__ import annotations

import re
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def _pyproject_version() -> str | None:
    path = Path(__file__).resolve().parents[2] / "pyproject.toml"
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    if not re.search(r'^name\s*=\s*"yunshu"\s*$', text, re.MULTILINE):
        return None
    m = re.search(r'^version\s*=\s*"([^"]+)"\s*$', text, re.MULTILINE)
    return m.group(1) if m else None


def yunshu_version() -> str:
    from_checkout = _pyproject_version()
    if from_checkout:
        return from_checkout
    try:
        return version("yunshu")
    except PackageNotFoundError:
        return "unknown"
