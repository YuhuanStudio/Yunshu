"""Local API-key store: named keys with scopes, expiry, per-key quotas and usage.

One file, ``~/.yunshu/keys.json`` (0600), holds the key *configuration*: only a SHA-256 of
the secret plus a short display prefix are stored; the secret is shown once, at creation or
rotation.  ``keys-usage.json`` next to it holds counters (per-day usage, the rolling quota
window, ``last_used``) and is flushed at most every ``FLUSH_S`` seconds and at exit.

``YUNSHU_AUTH_TOKEN`` stays the admin key and is checked by the callers before this store.
With no keys in the store nothing here changes behaviour.

Request path cost: one dict lookup by hash (lock-free: the table is replaced, never mutated),
then a short lock around integer counters.  Quota window: 288 five-minute buckets with running
sums, so memory is bounded and a check is O(1) amortised.
"""

from __future__ import annotations

import atexit
import contextlib
import hashlib
import json
import os
import secrets
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCOPES = ("infer", "admin")
QUOTA_FIELDS = ("requests_per_day", "tokens_per_day", "max_concurrent")
BUCKET_S = 300
N_BUCKETS = 288  # 24 h
FLUSH_S = 30.0
RELOAD_CHECK_S = 1.0
SECRET_PREFIX = "ysk-"


def hash_secret(secret: str) -> str:
    return hashlib.sha256(secret.encode("utf-8", "surrogatepass")).hexdigest()


def default_path() -> Path:
    return Path.home() / ".yunshu" / "keys.json"


@dataclass
class KeyRecord:
    id: str
    name: str
    prefix: str
    hash: str
    created: float
    enabled: bool = True
    scopes: tuple[str, ...] = ("infer",)
    expires: float | None = None
    quotas: dict[str, int | None] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "prefix": self.prefix,
            "hash": self.hash,
            "created": self.created,
            "enabled": self.enabled,
            "scopes": list(self.scopes),
            "expires": self.expires,
            "quotas": {k: self.quotas.get(k) for k in QUOTA_FIELDS},
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> KeyRecord:
        return cls(
            id=d["id"],
            name=d.get("name", ""),
            prefix=d.get("prefix", ""),
            hash=d["hash"],
            created=float(d.get("created", 0)),
            enabled=bool(d.get("enabled", True)),
            scopes=tuple(s for s in d.get("scopes", ["infer"]) if s in SCOPES),
            expires=d.get("expires"),
            quotas={k: (d.get("quotas") or {}).get(k) for k in QUOTA_FIELDS},
        )


class _State:
    """Mutable counters of one key (guarded by the store lock)."""

    __slots__ = ("buckets", "days", "inflight", "last_used", "req_sum", "tok_sum")

    def __init__(self) -> None:
        self.buckets: list[list[int]] = []  # [bucket_idx, requests, tokens], ascending
        self.req_sum = 0
        self.tok_sum = 0
        self.inflight = 0
        self.last_used: float | None = None
        self.days: dict[str, dict[str, int]] = {}

    def prune(self, now_idx: int) -> None:
        cutoff = now_idx - N_BUCKETS + 1
        while self.buckets and self.buckets[0][0] < cutoff:
            _, r, t = self.buckets.pop(0)
            self.req_sum -= r
            self.tok_sum -= t

    def bucket(self, now_idx: int) -> list[int]:
        if self.buckets and self.buckets[-1][0] == now_idx:
            return self.buckets[-1]
        b = [now_idx, 0, 0]
        self.buckets.append(b)
        return b


@dataclass
class Principal:
    """Who a request is: the admin token, or one key."""

    kind: str  # "token" | "key"
    scopes: frozenset[str]
    key_id: str | None = None
    name: str | None = None


ADMIN_TOKEN = Principal("token", frozenset(SCOPES), None, "admin-token")


class QuotaExceededError(Exception):
    def __init__(self, which: str, message: str, retry_after: int) -> None:
        super().__init__(message)
        self.which = which
        self.retry_after = max(1, int(retry_after))


class AuthFailureError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


def _day(t: float) -> str:
    return datetime.fromtimestamp(t, tz=UTC).strftime("%Y-%m-%d")


class KeyStore:
    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path) if path else default_path()
        self.usage_path = self.path.with_name("keys-usage.json")
        self._lock = threading.Lock()
        self._keys: dict[str, KeyRecord] = {}
        self._by_hash: dict[str, KeyRecord] = {}
        self._state: dict[str, _State] = {}
        self._mtime: float | None = None
        self._next_check = 0.0
        self._dirty = False
        self._last_flush = time.monotonic()
        self._load()
        self._load_usage()

    # ── persistence ────────────────────────────────────────────────────
    def _stat(self) -> float | None:
        try:
            return self.path.stat().st_mtime
        except OSError:
            return None

    def _load(self) -> None:
        self._mtime = self._stat()
        keys: dict[str, KeyRecord] = {}
        try:
            data = json.loads(self.path.read_text())
            for d in data.get("keys", []):
                rec = KeyRecord.from_dict(d)
                keys[rec.id] = rec
        except (OSError, ValueError, KeyError, TypeError):
            if self._mtime is not None:
                return  # unreadable right now (mid-write, bad edit): keep what we had
            keys = {}
        self._install(keys)

    def _install(self, keys: dict[str, KeyRecord]) -> None:
        self._keys = keys
        self._by_hash = {
            k.hash: k for k in keys.values()
        }  # replaced whole: lock-free reads
        for kid in keys:
            self._state.setdefault(kid, _State())

    def _load_usage(self) -> None:
        try:
            data = json.loads(self.usage_path.read_text())
        except (OSError, ValueError):
            return
        for kid, d in (data.get("keys") or {}).items():
            st = self._state.setdefault(kid, _State())
            st.last_used = d.get("last_used")
            st.days = {
                k: {m: int(v) for m, v in row.items()}
                for k, row in (d.get("days") or {}).items()
            }
            st.buckets = [list(map(int, b)) for b in d.get("buckets") or []]
            st.req_sum = sum(b[1] for b in st.buckets)
            st.tok_sum = sum(b[2] for b in st.buckets)
            st.prune(int(time.time() // BUCKET_S))

    def _write_private(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex[:8]}.tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(payload, f, indent=1)
            os.replace(tmp, path)
        finally:
            with contextlib.suppress(OSError):
                os.unlink(tmp)

    def _save_keys(self) -> None:
        self._write_private(
            self.path,
            {"version": 1, "keys": [k.to_dict() for k in self._keys.values()]},
        )
        self._mtime = self._stat()

    def flush(self, force: bool = False) -> None:
        with self._lock:
            if not (self._dirty or force):
                return
            payload = {
                "version": 1,
                "keys": {
                    kid: {
                        "last_used": st.last_used,
                        "days": st.days,
                        "buckets": st.buckets,
                    }
                    for kid, st in self._state.items()
                    if kid in self._keys
                },
            }
            self._dirty = False
            self._last_flush = time.monotonic()
        with contextlib.suppress(OSError):
            self._write_private(self.usage_path, payload)

    def _maybe_flush(self) -> None:
        if self._dirty and time.monotonic() - self._last_flush >= FLUSH_S:
            self.flush()

    def _refresh(self) -> None:
        """Pick up edits made by another process (at most one stat per second)."""
        now = time.monotonic()
        if now < self._next_check:
            return
        self._next_check = now + RELOAD_CHECK_S
        if self._stat() != self._mtime:
            with self._lock:
                self._load()

    # ── queries ────────────────────────────────────────────────────────
    def has_keys(self) -> bool:
        self._refresh()
        return bool(self._keys)

    def lookup(self, secret: str) -> KeyRecord | None:
        self._refresh()
        return self._by_hash.get(hash_secret(secret))

    # ── request path ───────────────────────────────────────────────────
    def authenticate(self, secret: str, now: float | None = None) -> Principal:
        """Resolve a presented secret to a principal or raise ``AuthFailureError`` (no counting)."""
        rec = self.lookup(secret)
        if rec is None:
            raise AuthFailureError(401, "Invalid or missing API key")
        now = time.time() if now is None else now
        if not rec.enabled:
            raise AuthFailureError(401, "This API key is disabled")
        if rec.expires is not None and now >= rec.expires:
            raise AuthFailureError(401, "This API key has expired")
        return Principal("key", frozenset(rec.scopes), rec.id, rec.name)

    def admit(self, principal: Principal, now: float | None = None) -> None:
        """Count a request against the key's quotas; raise ``QuotaExceededError`` (nothing counted).
        On success the key's concurrency slot is held until ``release``."""
        rec = self._keys.get(principal.key_id or "")
        if rec is None:
            raise AuthFailureError(401, "Invalid or missing API key")
        now = time.time() if now is None else now
        idx = int(now // BUCKET_S)
        q = rec.quotas
        with self._lock:
            st = self._state.setdefault(rec.id, _State())
            st.prune(idx)
            mc = q.get("max_concurrent")
            if mc and st.inflight >= mc:
                raise QuotaExceededError(
                    "max_concurrent",
                    f"Too many concurrent requests for this API key (limit {mc}).",
                    1,
                )
            rpd = q.get("requests_per_day")
            if rpd and st.req_sum >= rpd:
                raise QuotaExceededError(
                    "requests_per_day",
                    f"Daily request quota exceeded for this API key (limit {rpd}).",
                    self._retry(st, idx, now, 1),
                )
            tpd = q.get("tokens_per_day")
            if tpd and st.tok_sum >= tpd:
                raise QuotaExceededError(
                    "tokens_per_day",
                    f"Daily token quota exceeded for this API key (limit {tpd}).",
                    self._retry(st, idx, now, 2),
                )
            st.bucket(idx)[1] += 1
            st.req_sum += 1
            st.inflight += 1
            st.last_used = now
            day = st.days.setdefault(_day(now), {})
            day["requests"] = day.get("requests", 0) + 1
            self._dirty = True
        self._maybe_flush()

    @staticmethod
    def _retry(st: _State, idx: int, now: float, col: int) -> int:
        """Seconds until the oldest bucket holding the counted resource leaves the window."""
        for b in st.buckets:
            if b[col]:
                return int((b[0] + N_BUCKETS) * BUCKET_S - now) + 1
        return 1

    def release(self, key_id: str | None) -> None:
        if not key_id:
            return
        with self._lock:
            st = self._state.get(key_id)
            if st is not None and st.inflight > 0:
                st.inflight -= 1

    def account(
        self,
        key_id: str | None,
        prompt: int = 0,
        completion: int = 0,
        cached: int = 0,
        error: bool = False,
        now: float | None = None,
    ) -> None:
        """Add a finished request's tokens (called once, from ``x_yunshu.record_done``) or an error."""
        if not key_id:
            return
        now = time.time() if now is None else now
        idx = int(now // BUCKET_S)
        with self._lock:
            st = self._state.get(key_id)
            if st is None:
                return
            day = st.days.setdefault(_day(now), {})
            if error:
                day["errors"] = day.get("errors", 0) + 1
            else:
                day["prompt_tokens"] = day.get("prompt_tokens", 0) + prompt
                day["completion_tokens"] = day.get("completion_tokens", 0) + completion
                day["cached_tokens"] = day.get("cached_tokens", 0) + cached
                n = prompt + completion
                st.prune(idx)
                st.bucket(idx)[2] += n
                st.tok_sum += n
            self._dirty = True
        self._maybe_flush()

    # ── admin ──────────────────────────────────────────────────────────
    @staticmethod
    def _new_secret() -> str:
        return SECRET_PREFIX + secrets.token_urlsafe(32)

    @staticmethod
    def _clean(
        scopes: Any, quotas: Any, expires: Any
    ) -> tuple[tuple[str, ...] | None, dict[str, int | None] | None, Any]:
        if scopes is not None:
            if not isinstance(scopes, (list, tuple)) or not scopes:
                raise ValueError("scopes must be a non-empty list")
            bad = [s for s in scopes if s not in SCOPES]
            if bad:
                raise ValueError(f"unknown scope {bad[0]!r}; use {', '.join(SCOPES)}")
            scopes = tuple(dict.fromkeys(scopes))
        if quotas is not None:
            if not isinstance(quotas, dict):
                raise ValueError("quotas must be an object")
            out: dict[str, int | None] = {}
            for k, v in quotas.items():
                if k not in QUOTA_FIELDS:
                    raise ValueError(
                        f"unknown quota {k!r}; use {', '.join(QUOTA_FIELDS)}"
                    )
                if v is not None and (
                    isinstance(v, bool) or not isinstance(v, int) or v < 0
                ):
                    raise ValueError(
                        f"quota {k} must be a non-negative integer or null"
                    )
                out[k] = v or None  # 0 / null = unlimited
            quotas = out
        if expires is not None and (
            isinstance(expires, bool) or not isinstance(expires, (int, float))
        ):
            raise ValueError("expires must be an epoch-seconds number or null")
        return scopes, quotas, expires

    def create(
        self,
        name: str,
        scopes: list[str] | None = None,
        quotas: dict | None = None,
        expires: float | None = None,
    ) -> tuple[KeyRecord, str]:
        if not name or not name.strip():
            raise ValueError("name is required")
        sc, qu, ex = self._clean(scopes or ["infer"], quotas or {}, expires)
        secret = self._new_secret()
        rec = KeyRecord(
            id="key_" + uuid.uuid4().hex[:12],
            name=name.strip()[:80],
            prefix=secret[: len(SECRET_PREFIX) + 4],
            hash=hash_secret(secret),
            created=time.time(),
            scopes=sc or ("infer",),
            expires=ex,
            quotas={k: (qu or {}).get(k) for k in QUOTA_FIELDS},
        )
        with self._lock:
            keys = dict(self._keys)
            keys[rec.id] = rec
            self._install(keys)
            self._save_keys()
        return rec, secret

    def update(self, key_id: str, patch: dict) -> KeyRecord:
        with self._lock:
            rec = self._keys.get(key_id)
            if rec is None:
                raise KeyError(key_id)
            sc, qu, ex = self._clean(
                patch.get("scopes"), patch.get("quotas"), patch.get("expires")
            )
            new = KeyRecord.from_dict(rec.to_dict())
            if "name" in patch:
                if not str(patch["name"] or "").strip():
                    raise ValueError("name must not be empty")
                new.name = str(patch["name"]).strip()[:80]
            if "enabled" in patch:
                if not isinstance(patch["enabled"], bool):
                    raise ValueError("enabled must be a boolean")
                new.enabled = patch["enabled"]
            if sc is not None:
                new.scopes = sc
            if qu is not None:
                new.quotas = {**rec.quotas, **qu}
            if "expires" in patch:
                new.expires = ex
            keys = dict(self._keys)
            keys[key_id] = new
            self._install(keys)
            self._save_keys()
            return new

    def rotate(self, key_id: str) -> tuple[KeyRecord, str]:
        with self._lock:
            rec = self._keys.get(key_id)
            if rec is None:
                raise KeyError(key_id)
            secret = self._new_secret()
            new = KeyRecord.from_dict(rec.to_dict())
            new.hash = hash_secret(secret)
            new.prefix = secret[: len(SECRET_PREFIX) + 4]
            keys = dict(self._keys)
            keys[key_id] = new
            self._install(keys)
            self._save_keys()
            return new, secret

    def delete(self, key_id: str) -> None:
        with self._lock:
            if key_id not in self._keys:
                raise KeyError(key_id)
            keys = {k: v for k, v in self._keys.items() if k != key_id}
            self._install(keys)
            self._state.pop(key_id, None)
            self._save_keys()
            self._dirty = True

    def public(self, rec: KeyRecord) -> dict[str, Any]:
        d = rec.to_dict()
        d.pop("hash", None)
        with self._lock:
            st = self._state.get(rec.id)
            now = time.time()
            if st is not None:
                st.prune(int(now // BUCKET_S))
            d["last_used"] = st.last_used if st else None
            d["window"] = {
                "requests": st.req_sum if st else 0,
                "tokens": st.tok_sum if st else 0,
                "inflight": st.inflight if st else 0,
            }
        d["expired"] = rec.expires is not None and now >= rec.expires
        return d

    def list_keys(self) -> list[dict[str, Any]]:
        self._refresh()
        return [self.public(k) for k in self._keys.values()]

    def usage(
        self, key_id: str | None = None, since: str | None = None
    ) -> list[dict[str, Any]]:
        """Per-day rows ``{key, day, requests, prompt_tokens, completion_tokens, cached_tokens, errors}``."""
        rows: list[dict[str, Any]] = []
        with self._lock:
            for kid, st in self._state.items():
                if kid not in self._keys or (key_id and kid != key_id):
                    continue
                for day, d in sorted(st.days.items()):
                    if since and day < since:
                        continue
                    rows.append(
                        {
                            "key": kid,
                            "name": self._keys[kid].name,
                            "day": day,
                            "requests": d.get("requests", 0),
                            "prompt_tokens": d.get("prompt_tokens", 0),
                            "completion_tokens": d.get("completion_tokens", 0),
                            "cached_tokens": d.get("cached_tokens", 0),
                            "errors": d.get("errors", 0),
                        }
                    )
        return rows


_store: KeyStore | None = None
_store_lock = threading.Lock()


def get_store() -> KeyStore:
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                _store = KeyStore()
                atexit.register(_store.flush)
    return _store


def configure(path: Path | str | None) -> KeyStore:
    """Point the process-wide store at ``path`` (tests, embedding); ``None`` re-reads the default."""
    global _store
    with _store_lock:
        if _store is not None:
            _store.flush()
        _store = KeyStore(Path(path) if path else None)
        return _store


def principal_for(presented: str | None, static_token: str | None) -> Principal | None:
    """Principal for a presented secret: the admin token first, then the key store.
    Returns None when nothing matches (the caller answers 401)."""
    if not presented:
        return None
    if static_token:
        from yunshu_gateway.token_compare import tokens_equal

        if tokens_equal(presented, static_token):
            return ADMIN_TOKEN
    store = get_store()
    if not store.has_keys():
        return None
    try:
        return store.authenticate(presented)
    except AuthFailureError:
        return None
