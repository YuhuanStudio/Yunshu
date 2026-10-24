"""Measurement pause: while a pause file says so, the console sends nothing to any engine.

A timing A/B must not be perturbed by a monitor that polls only one arm. ``gpuq`` writes
``{"pause": true, "until": <epoch or null>, "reason": str}`` to the file named by
``YUNSHU_CONSOLE_PAUSE_FILE`` while a timing job runs and removes it afterwards. The file counts
only while it exists, says ``pause: true`` and is not stale (past ``until``; two hours after its
mtime when ``until`` is null). The poller and the discovery probes check :meth:`Pause.state` before
every request; the proxy for user-initiated requests is not affected.
"""

from __future__ import annotations

import json
import os
import time

STALE_NO_UNTIL_S = 2 * 3600


class Pause:
    def __init__(self, path: str | os.PathLike | None, clock=time.time) -> None:
        self.path = os.fspath(path) if path else None
        self._clock = clock

    def state(self) -> dict | None:
        """``{"reason", "until"}`` while paused, else ``None`` (no file, unreadable, stale)."""
        if not self.path:
            return None
        try:
            mtime = os.stat(self.path).st_mtime
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError):
            return None
        if not isinstance(data, dict) or data.get("pause") is not True:
            return None
        now = self._clock()
        until = data.get("until")
        if isinstance(until, bool) or not isinstance(until, (int, float)):
            until = None
        if until is None:
            if now - mtime > STALE_NO_UNTIL_S:
                return None
        elif now >= until:
            return None
        reason = data.get("reason")
        return {"reason": reason if isinstance(reason, str) else "", "until": until}

    @property
    def paused(self) -> bool:
        return self.state() is not None
