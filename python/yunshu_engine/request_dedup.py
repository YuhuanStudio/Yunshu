from __future__ import annotations
"""Request deduplication — merge identical concurrent requests (SGLang pattern).

When multiple users submit identical requests simultaneously (same model,
same prompt, same sampling params), this module detects and merges them.
Only one actual inference is performed; all requesters share the output.

This is especially useful for:
  - Popular system prompts (chat completions with same prefix)
  - Cache warming bursts (multiple预热 requests)
  - Load testing / benchmark scenarios
"""

import hashlib
import logging
import os
import threading
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class DeduplicationEntry:
    """Tracks a deduplicated request group."""
    content_hash: str
    request_ids: list[str] = field(default_factory=list)
    primary_request_id: str = ""
    created_at: float = field(default_factory=time.monotonic)
    completed_at: float | None = None
    # The actual request parameters
    model: str = ""
    prompt_hash: str = ""
    sampling_hash: str = ""

    @property
    def fan_out(self) -> int:
        return len(self.request_ids)

    @property
    def age_ms(self) -> float:
        end = self.completed_at or time.monotonic()
        return (end - self.created_at) * 1000

    @property
    def is_completed(self) -> bool:
        return self.completed_at is not None


class RequestDeduplicator:
    """Detects and merges identical concurrent requests.

    Content hash = SHA-256(model + prompt + sampling_params)
    When a hash matches an in-flight request, the new request joins
    the existing group and receives the same output.

    Enable via YUNSHU_REQUEST_DEDUP=1.
    """

    def __init__(
        self,
        window_ms: float = 100.0,
        max_fan_out: int = 8,
        max_entries: int = 1000,
        ttl_seconds: float = 60.0,
    ) -> None:
        self._window_ms = window_ms
        self._max_fan_out = max_fan_out
        self._max_entries = max_entries
        self._ttl = ttl_seconds
        self._entries: dict[str, DeduplicationEntry] = {}
        self._lock = threading.Lock()
        self._total_deduplicated = 0
        self._total_saved_requests = 0
        self._total_inferences = 0

    @classmethod
    def from_env(cls) -> RequestDeduplicator:
        return cls(
            window_ms=float(os.environ.get("YUNSHU_DEDUP_WINDOW_MS", "100.0")),
            max_fan_out=int(os.environ.get("YUNSHU_DEDUP_MAX_FANOUT", "8")),
            max_entries=int(os.environ.get("YUNSHU_DEDUP_MAX_ENTRIES", "1000")),
            ttl_seconds=float(os.environ.get("YUNSHU_DEDUP_TTL", "60.0")),
        )

    @staticmethod
    def compute_hash(
        model: str,
        prompt: str | list[int],
        temperature: float = 0.7,
        top_p: float = 1.0,
        max_tokens: int = 512,
        **kwargs,
    ) -> str:
        """Compute content hash for deduplication."""
        import json
        # Deterministic serialization: json.dumps with sort_keys for
        # list[dict] (chat messages) to avoid key-ordering variance.
        if isinstance(prompt, list) and prompt and isinstance(prompt[0], dict):
            prompt_str = json.dumps(prompt, sort_keys=True, ensure_ascii=False)
        else:
            prompt_str = str(prompt) if isinstance(prompt, list) else prompt
        parts = [
            model,
            prompt_str,
            f"t={temperature}",
            f"p={top_p}",
            f"m={max_tokens}",
        ]
        # Add other sampling params deterministically
        for k in sorted(kwargs):
            if kwargs[k] is not None:
                parts.append(f"{k}={kwargs[k]}")
        content = "|".join(parts)
        return hashlib.sha256(content.encode()).hexdigest()[:32]

    def check(
        self,
        request_id: str,
        content_hash: str,
        model: str = "",
    ) -> DeduplicationEntry | None:
        """Check if this request can be deduplicated.

        Returns the existing entry if deduplication is possible,
        or None if this is a unique request.

        Thread-safe: uses internal lock for all state mutations.
        """
        with self._lock:
            # Prune expired entries
            self._prune_expired()

            entry = self._entries.get(content_hash)

            if entry is None or entry.is_completed:
                return None

            # Check if within deduplication window
            if entry.age_ms > self._window_ms:
                return None

            # Check fan-out limit
            if entry.fan_out >= self._max_fan_out:
                return None

            # Deduplicate: add to existing entry
            entry.request_ids.append(request_id)
            self._total_saved_requests += 1
            self._total_deduplicated += 1
            return entry

    def register(
        self,
        request_id: str,
        content_hash: str,
        model: str = "",
        prompt_hash: str = "",
    ) -> DeduplicationEntry:
        """Register a new request (not deduplicated).

        If an existing in-flight entry with the same hash exists but its age
        exceeded the dedup window (causing check() to return None), we do NOT
        overwrite it — the original primary is still in-flight and overwriting
        would orphan its shadow requests.  Instead, treat this as a new
        independent inference (no dedup).
        """
        with self._lock:
            self._prune_expired()

            # Guard: do not overwrite an in-flight entry whose age simply
            # exceeded the dedup window.  Only replace completed entries.
            existing = self._entries.get(content_hash)
            if existing is not None and not existing.is_completed:
                # The original primary is still in-flight.  Create a new
                # entry with a different hash (append a nonce) so both
                # inferences run independently.
                import hashlib
                for _ in range(3):
                    nonce = hashlib.sha256(
                        (content_hash + request_id).encode()
                    ).hexdigest()[:16]
                    if nonce not in self._entries:
                        content_hash = nonce
                        break
                    content_hash = nonce + request_id[:8]
                else:
                    content_hash = nonce + request_id[:8]

                # Final uniqueness guarantee: if the nonce loop exhausted
                # without finding a free key, keep appending a counter suffix
                # until we get one that doesn't collide with any in-flight entry.
                _suffix = 0
                while content_hash in self._entries and not self._entries[content_hash].is_completed:
                    content_hash = f"{nonce}{request_id[:8]}_{_suffix}"
                    _suffix += 1

            # Check capacity
            if len(self._entries) >= self._max_entries:
                self._evict_oldest()

            entry = DeduplicationEntry(
                content_hash=content_hash,
                request_ids=[request_id],
                primary_request_id=request_id,
                model=model,
                prompt_hash=prompt_hash,
            )
            self._entries[content_hash] = entry
            self._total_inferences += 1
            return entry

    def complete(self, content_hash: str) -> list[str]:
        """Mark a deduplication entry as completed.

        Returns all request IDs that should receive the output.
        """
        with self._lock:
            entry = self._entries.get(content_hash)
            if entry is None:
                return []

            entry.completed_at = time.monotonic()
            return list(entry.request_ids)

    def _prune_expired(self) -> None:
        now = time.monotonic()
        expired = [
            h for h, e in self._entries.items()
            if e.completed_at and (now - e.completed_at) > self._ttl
        ]
        # Also evict stuck in-flight entries that have been running for 10x TTL.
        # These are likely orphaned (primary crashed without calling complete()).
        stuck = [
            h for h, e in self._entries.items()
            if e.completed_at is None and (now - e.created_at) > self._ttl * 10
        ]
        for h in expired + stuck:
            del self._entries[h]

    def _evict_oldest(self) -> None:
        if not self._entries:
            return
        # Only evict completed entries. Evicting an in-flight primary
        # would orphan its shadow requests — they hold references to an
        # entry that no longer exists, breaking fan-out delivery.
        completed = [
            h for h, e in self._entries.items() if e.completed_at is not None
        ]
        if completed:
            oldest_hash = min(completed, key=lambda h: self._entries[h].created_at)
            del self._entries[oldest_hash]
            return
        # All entries are in-flight — cannot safely evict any of them.
        # Log a warning instead of orphaning shadows.
        logger.warning(
            "Dedup capacity reached with %d in-flight entries — "
            "skipping eviction to avoid orphaning shadow requests",
            len(self._entries),
        )

    def get_entry(self, content_hash: str) -> DeduplicationEntry | None:
        with self._lock:
            return self._entries.get(content_hash)

    def get_stats(self) -> dict:
        with self._lock:
            active = sum(1 for e in self._entries.values() if not e.is_completed)
            avg_fan_out = (
                sum(e.fan_out for e in self._entries.values())
                / len(self._entries)
                if self._entries
                else 0.0
            )
            return {
                "active_entries": active,
                "total_entries": len(self._entries),
                "total_deduplicated": self._total_deduplicated,
                "total_saved_requests": self._total_saved_requests,
                "total_inferences": self._total_inferences,
                "avg_fan_out": round(avg_fan_out, 2),
                "dedup_rate": (
                    self._total_deduplicated / max(self._total_inferences, 1)
                ),
                "max_fan_out": self._max_fan_out,
                "window_ms": self._window_ms,
            }
