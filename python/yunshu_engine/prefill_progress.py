"""Prefill progress tracker — lightweight per-request progress for dashboard.

Studied from oMLX's prefill_progress.py. Updated by scheduler during prefill,
read by admin API for live dashboard display. Thread-safe, O(1) per update.
"""
from __future__ import annotations

import threading
import time
from typing import Any


class PrefillProgressTracker:
    """Thread-safe tracker for per-request prefill progress.

    oMLX pattern: tracks (processed_tokens, total_tokens, speed, ETA) per request.
    Auto-removes entries when prefill completes. ~50ns per update.
    """

    def __init__(self) -> None:
        self._progress: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def update(self, request_id: str, processed: int, total: int, model_id: str) -> None:
        now = time.monotonic()
        with self._lock:
            if processed >= total:
                self._progress.pop(request_id, None)
            else:
                prev = self._progress.get(request_id)
                if prev is not None:
                    dt = now - prev["last_time"]
                    dtok = processed - prev["processed"]
                    speed = (dtok / dt) if dt > 0 and dtok > 0 else prev.get("speed", 0.0)
                else:
                    speed = 0.0

                self._progress[request_id] = {
                    "processed": processed,
                    "total": total,
                    "model_id": model_id,
                    "start_time": prev["start_time"] if prev else now,
                    "last_time": now,
                    "speed": speed,
                }

    def remove(self, request_id: str) -> None:
        with self._lock:
            self._progress.pop(request_id, None)

    def get_model_progress(self, model_id: str) -> list[dict]:
        with self._lock:
            results = []
            for rid, entry in self._progress.items():
                if entry["model_id"] != model_id:
                    continue
                remaining = entry["total"] - entry["processed"]
                speed = entry.get("speed", 0.0)
                eta = remaining / speed if speed > 0 else None
                results.append({
                    "request_id": rid,
                    "processed": entry["processed"],
                    "total": entry["total"],
                    "progress_pct": round(entry["processed"] / entry["total"] * 100, 1) if entry["total"] > 0 else 0,
                    "speed_tok_s": round(speed, 0),
                    "eta_s": round(eta, 1) if eta is not None else None,
                })
            return results

    def get_all_progress(self) -> dict[str, list[dict]]:
        with self._lock:
            models: dict[str, list[dict]] = {}
            for rid, entry in self._progress.items():
                mid = entry["model_id"]
                if mid not in models:
                    models[mid] = []
                remaining = entry["total"] - entry["processed"]
                speed = entry.get("speed", 0.0)
                eta = remaining / speed if speed > 0 else None
                models[mid].append({
                    "request_id": rid,
                    "processed": entry["processed"],
                    "total": entry["total"],
                    "progress_pct": round(entry["processed"] / entry["total"] * 100, 1) if entry["total"] > 0 else 0,
                    "speed_tok_s": round(speed, 0),
                    "eta_s": round(eta, 1) if eta is not None else None,
                })
            return models

    def clear(self) -> None:
        with self._lock:
            self._progress.clear()

    @property
    def active_count(self) -> int:
        with self._lock:
            return len(self._progress)


# Module-level singleton
_tracker: PrefillProgressTracker | None = None


def get_prefill_tracker() -> PrefillProgressTracker:
    global _tracker
    if _tracker is None:
        _tracker = PrefillProgressTracker()
    return _tracker
