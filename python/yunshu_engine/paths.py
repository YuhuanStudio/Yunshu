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


def media_dir() -> str:
    """Directory local media paths must live under (also where the gateway stages uploads)."""
    import os

    return settings.get("YUNSHU_MEDIA_DIR") or os.path.join(
        os.environ.get("TMPDIR", "/tmp"), "yunshu_media"
    )


def stage_media_file(suffix: str):
    """A NamedTemporaryFile(delete=False) inside the media dir, so the VLM path accepts it."""
    import os
    import tempfile

    d = media_dir()
    os.makedirs(d, exist_ok=True)
    return tempfile.NamedTemporaryFile(suffix=suffix, dir=d, delete=False)  # noqa: SIM115
