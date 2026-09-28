"""Where Yunshu keeps things on disk when nothing is configured.

One place for the per-user locations, so the CLI, the gateway and the
launchd service agree:

- models: ``YUNSHU_MODELS_DIR``, else ``~/.yunshu/models``
- service logs: ``~/Library/Logs/Yunshu``
- launchd agent: ``~/Library/LaunchAgents/<SERVICE_LABEL>.plist``
"""

from __future__ import annotations

from pathlib import Path

from . import settings

SERVICE_LABEL = "com.yuhuanstudio.yunshu"


def home() -> Path:
    return Path.home() / ".yunshu"


def models_dir() -> Path:
    configured = settings.get("YUNSHU_MODELS_DIR")
    if configured:
        return Path(configured).expanduser()
    return home() / "models"


def log_dir() -> Path:
    return Path.home() / "Library" / "Logs" / "Yunshu"


def launch_agent_plist() -> Path:
    return Path.home() / "Library" / "LaunchAgents" / f"{SERVICE_LABEL}.plist"
