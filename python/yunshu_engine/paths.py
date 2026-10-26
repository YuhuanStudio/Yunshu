"""Where Yunshu keeps things on disk when nothing is configured.

One place for the per-user locations, so the CLI, the gateway and the
launchd service agree:

- models: ``YUNSHU_MODELS_DIR``, else ``~/.yunshu/models``
- APC SSD tier: ``YUNSHU_VLM_APC_DISK_DIR``, else ``~/.yunshu/cache/apc``
- service logs: ``~/Library/Logs/Yunshu``
- launchd agents: ``~/Library/LaunchAgents/<SERVICE_LABEL>.plist`` (the engine) and
  ``<CONSOLE_SERVICE_LABEL>.plist`` (the console process)
"""

from __future__ import annotations

from pathlib import Path

from . import settings

SERVICE_LABEL = "com.yuhuanstudio.yunshu"
# The console runs as its own launchd job, so it keeps recording while the engine restarts.
CONSOLE_SERVICE_LABEL = "com.yuhuanstudio.yunshu.console"


def home() -> Path:
    return Path.home() / ".yunshu"


def models_dir() -> Path:
    configured = settings.get("YUNSHU_MODELS_DIR")
    if configured:
        return Path(configured).expanduser()
    return home() / "models"


def apc_dir() -> Path | None:
    """The APC SSD tier directory, or None when the tier is switched off."""
    if not settings.get_bool("YUNSHU_VLM_APC_DISK"):
        return None
    configured = settings.get("YUNSHU_VLM_APC_DISK_DIR")
    if configured:
        return Path(configured).expanduser()
    return home() / "cache" / "apc"


def log_dir() -> Path:
    return Path.home() / "Library" / "Logs" / "Yunshu"


def console_launch_agent_plist() -> Path:
    return Path.home() / "Library" / "LaunchAgents" / f"{CONSOLE_SERVICE_LABEL}.plist"


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


def private_media_path() -> str:
    """Where decoded request media (base64 / downloaded images, audio, video) are written.
    A fixed per-user name under the system temp dir, so the sandbox deny list can name it
    without creating it. Shared system temp is readable by sandboxed code; this directory is not."""
    import os
    import tempfile

    return os.path.join(tempfile.gettempdir(), f"yunshu_decoded_{os.getuid()}")


def private_media_dir() -> str:
    """``private_media_path()``, created 0700 (a symlink or someone else's directory is refused)."""
    import contextlib
    import os
    import stat

    d = private_media_path()
    with contextlib.suppress(FileExistsError):
        os.mkdir(d, 0o700)
    st = os.lstat(d)
    if not stat.S_ISDIR(st.st_mode) or st.st_uid != os.getuid():
        raise RuntimeError(f"{d} is not a private directory owned by this user")
    if stat.S_IMODE(st.st_mode) != 0o700:
        os.chmod(d, 0o700)
    return d
