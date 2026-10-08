"""Log rotation and secret redaction for the launchd service log. Local only, no telemetry.

launchd opens ``StandardOutPath`` once and appends forever, so the service log grows without
bound. :func:`rotate` copies the live file to a gzip archive and truncates it in place (the
open descriptor keeps working; a few lines written during the copy can be lost, as with
``logrotate copytruncate``). Archives are redacted as they are written; they are what a
diagnostics bundle collects. Rotation is by size and by age of the last rotation; archives
are capped by count and by age.
"""

from __future__ import annotations

import contextlib
import gzip
import os
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from . import settings

_REDACTED = "[REDACTED]"
# (pattern, replacement). Order matters: header forms before bare tokens.
_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(r"(?i)(authorization\s*[:=]\s*)(bearer\s+)?[^\s,;'\"]+"),
        rf"\1\2{_REDACTED}",
    ),
    (
        re.compile(r"(?i)\b(bearer\s+)[A-Za-z0-9._~+/=-]{8,}"),
        rf"\1{_REDACTED}",
    ),
    (
        re.compile(
            r"(?i)\b([A-Z0-9_]*(?:TOKEN|SECRET|API[_-]?KEY|PASSWORD|PASSWD)(?![A-Z])"
            r"|x-api-key)(\s*[:=]\s*|\"\s*:\s*\")([^\s,;'\"}]+)"
        ),
        rf"\1\2{_REDACTED}",
    ),
    (re.compile(r"\b(sk|rk|pk)-[A-Za-z0-9_-]{16,}"), _REDACTED),
    (re.compile(r"\bek_[A-Za-z0-9_-]{16,}"), _REDACTED),
    (re.compile(r"\bhf_[A-Za-z0-9]{20,}"), _REDACTED),
    (re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}"), _REDACTED),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), _REDACTED),
    # key material in a URL query string
    (
        re.compile(r"(?i)([?&](?:api[_-]?key|token|key|access_token)=)[^&\s]+"),
        rf"\1{_REDACTED}",
    ),
]


def redact(text: str) -> str:
    """``text`` with credentials, tokens and API keys replaced by ``[REDACTED]``."""
    for pattern, repl in _PATTERNS:
        text = pattern.sub(repl, text)
    return text


@dataclass
class Policy:
    max_bytes: int
    keep: int
    retention_s: float
    interval_s: float

    @classmethod
    def from_settings(cls) -> Policy:
        return cls(
            max_bytes=int(settings.get("YUNSHU_LOG_MAX_MB") * 1024 * 1024),
            keep=int(settings.get("YUNSHU_LOG_KEEP")),
            retention_s=settings.get("YUNSHU_LOG_RETENTION_DAYS") * 86400.0,
            interval_s=settings.get("YUNSHU_LOG_ROTATE_HOURS") * 3600.0,
        )


def archives(log: Path) -> list[Path]:
    """Rotated archives of ``log``, newest first."""
    found = [p for p in log.parent.glob(f"{log.name}.*.gz") if p.is_file()]
    return sorted(found, key=lambda p: p.stat().st_mtime, reverse=True)


def _last_rotation(log: Path) -> float:
    arch = archives(log)
    if arch:
        return arch[0].stat().st_mtime
    st = log.stat()
    return getattr(st, "st_birthtime", st.st_mtime)


def rotate(
    log: Path, policy: Policy | None = None, *, now: float | None = None
) -> dict:
    """Rotate and prune ``log`` per ``policy``. Returns what happened."""
    policy = policy or Policy.from_settings()
    now = time.time() if now is None else now
    out: dict = {"rotated": False, "archive": None, "pruned": []}
    if log.is_file():
        size = log.stat().st_size
        due_size = policy.max_bytes > 0 and size >= policy.max_bytes
        due_time = (
            policy.interval_s > 0
            and size > 0
            and now - _last_rotation(log) >= policy.interval_s
        )
        if due_size or due_time:
            stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime(now))
            dest = log.with_name(f"{log.name}.{stamp}.gz")
            n = 1
            while dest.exists():
                dest = log.with_name(f"{log.name}.{stamp}-{n}.gz")
                n += 1
            with open(log, "rb") as src:
                data = src.read(size)
            text = redact(data.decode("utf-8", errors="replace"))
            tmp = dest.with_suffix(".tmp")
            with gzip.open(tmp, "wb") as gz:
                gz.write(text.encode("utf-8"))
            os.replace(tmp, dest)
            os.utime(dest, (now, now))
            with open(log, "r+b") as live:  # copy-truncate: the writer's fd stays valid
                live.truncate(0)
            out.update(rotated=True, archive=str(dest))
    arch = archives(log)
    for i, p in enumerate(arch):
        too_many = policy.keep >= 0 and i >= policy.keep
        too_old = (
            policy.retention_s > 0 and now - p.stat().st_mtime > policy.retention_s
        )
        if too_many or too_old:
            with contextlib.suppress(OSError):
                p.unlink()
                out["pruned"].append(str(p))
    return out


def stdout_is(log: Path) -> bool:
    """True when this process's stdout is ``log`` (it runs under the launchd agent)."""
    try:
        a, b = os.fstat(1), log.stat()
    except OSError:
        return False
    return (a.st_dev, a.st_ino) == (b.st_dev, b.st_ino)


def start_background(log: Path, every_s: float = 600.0) -> threading.Thread:
    """Rotate now, then every ``every_s`` seconds, on a daemon thread."""

    def loop() -> None:
        while True:
            with contextlib.suppress(Exception):
                rotate(log)
            time.sleep(every_s)

    t = threading.Thread(target=loop, name="yunshu-log-rotate", daemon=True)
    t.start()
    return t
