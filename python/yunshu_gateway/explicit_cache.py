"""Explicit context cache (Gemini-style) — the third prompt-caching paradigm.

Three vendor caching concepts, all now supported by Yunshu:
  1. OpenAI  — AUTOMATIC implicit prefix caching; reported via
     usage.prompt_tokens_details.cached_tokens. (KVPrefixCache, always on.)
  2. Anthropic — EXPLICIT breakpoint HINTS (cache_control: ephemeral) +
     usage.cache_creation_input_tokens / cache_read_input_tokens.
  3. Google Gemini — EXPLICIT NAMED cache objects: the client WRITES a cache
     (POST content + TTL → a handle) and READS it by referencing the handle in
     later requests. This module implements that paradigm.

The named handle stores the content server-side with a TTL; on create we WARM
the underlying KV prefix cache (one prefill) so the first read is already hot.
On read, the stored content is prepended to the request and the automatic
KVPrefixCache serves the warmed prefix (cached_tokens reflects the reuse). If the
KV was LRU-evicted before a read, it transparently re-prefills once — the handle
governs *validity/TTL*, the KV tier governs *residency*.
"""
from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field


@dataclass
class CachedContent:
    name: str                      # "cachedContents/<id>"
    model: str
    messages: list[dict]           # content to prepend on read (system/context)
    token_count: int
    created_at: float
    expire_at: float
    display_name: str = ""
    last_used_at: float = field(default=0.0)
    read_count: int = 0
    owner: str = "anonymous"       # caller identity (resolve_actor) for tenant isolation

    def to_api(self, now: float | None = None) -> dict:
        now = now if now is not None else time.time()
        return {
            "name": self.name,
            "model": self.model,
            "displayName": self.display_name,
            "createTime": _rfc3339(self.created_at),
            "updateTime": _rfc3339(self.created_at),
            "expireTime": _rfc3339(self.expire_at),
            "ttl": f"{max(0.0, self.expire_at - now):.1f}s",
            "usageMetadata": {"totalTokenCount": self.token_count},
            "readCount": self.read_count,
        }


def _rfc3339(ts: float) -> str:
    import datetime
    return datetime.datetime.utcfromtimestamp(ts).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


class ExplicitContextCache:
    """Thread-safe registry of named, TTL'd context caches."""

    def __init__(self, default_ttl: float = 3600.0, max_entries: int = 256):
        self._entries: dict[str, CachedContent] = {}
        self._lock = threading.Lock()
        self._default_ttl = default_ttl
        self._max_entries = max_entries

    def _evict_expired(self, now: float) -> None:
        dead = [n for n, e in self._entries.items() if e.expire_at <= now]
        for n in dead:
            self._entries.pop(n, None)
        # Enforce the cap PER OWNER, not globally. The old global cap sorted ALL
        # entries by expire_at and dropped the soonest-to-expire regardless of owner, so one
        # tenant creating >max_entries handles could evict ANOTHER tenant's still-valid
        # handles (cross-tenant availability DoS). Evict only within an over-quota owner's own
        # bucket, so a tenant can never evict another's handles.
        by_owner: dict[str, list[str]] = {}
        for n, e in self._entries.items():
            by_owner.setdefault(e.owner, []).append(n)
        for names in by_owner.values():
            if len(names) > self._max_entries:
                for n in sorted(names, key=lambda k: self._entries[k].expire_at)[
                        :len(names) - self._max_entries]:
                    self._entries.pop(n, None)

    def create(self, model: str, messages: list[dict], token_count: int,
               ttl_seconds: float | None = None, display_name: str = "",
               owner: str = "anonymous") -> CachedContent:
        now = time.time()
        ttl = float(ttl_seconds) if ttl_seconds and ttl_seconds > 0 else self._default_ttl
        name = f"cachedContents/{uuid.uuid4().hex[:24]}"
        entry = CachedContent(
            name=name, model=model, messages=list(messages), token_count=int(token_count),
            created_at=now, expire_at=now + ttl, display_name=display_name, owner=owner)
        with self._lock:
            self._evict_expired(now)
            self._entries[name] = entry
        return entry

    def get(self, name: str) -> CachedContent | None:
        now = time.time()
        with self._lock:
            self._evict_expired(now)
            return self._entries.get(name)

    def use(self, name: str) -> CachedContent | None:
        """Fetch for a READ — bumps last_used/read_count. Returns None if missing/expired."""
        now = time.time()
        with self._lock:
            self._evict_expired(now)
            e = self._entries.get(name)
            if e is not None:
                e.last_used_at = now
                e.read_count += 1
            return e

    def delete(self, name: str) -> bool:
        with self._lock:
            return self._entries.pop(name, None) is not None

    def update_ttl(self, name: str, ttl_seconds: float) -> CachedContent | None:
        with self._lock:
            e = self._entries.get(name)
            if e is not None:
                e.expire_at = time.time() + max(0.0, float(ttl_seconds))
            return e

    def list(self) -> list[CachedContent]:
        now = time.time()
        with self._lock:
            self._evict_expired(now)
            return list(self._entries.values())


# Module-level singleton (one per gateway process).
_STORE: ExplicitContextCache | None = None
_STORE_LOCK = threading.Lock()


def get_store() -> ExplicitContextCache:
    global _STORE
    # Double-checked locking. The unguarded lazy-init raced on cold
    # start — two concurrent first requests both saw _STORE is None, each built
    # its own ExplicitContextCache, and the last assignment won, so a create()
    # the loser store already serviced was silently lost (the handle vanished).
    # The fast path (already-initialized) stays lock-free.
    if _STORE is None:
        with _STORE_LOCK:
            if _STORE is None:
                _STORE = ExplicitContextCache()
    return _STORE
