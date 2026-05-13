"""Yunshu Gateway — Rate limiting middleware.

Token bucket rate limiter per client IP with RBAC per-key override.
oMLX pattern: configurable RPM + token limits.

Memory safety: LRU eviction + TTL expiry prevents unbounded growth from
unique-IP DoS. Configurable via environment:
  YUNSHU_RATE_LIMIT_MAX_BUCKETS — max per-IP buckets (default: 10000)
  YUNSHU_RATE_LIMIT_TTL_SECONDS — bucket TTL in seconds (default: 600)
"""

from __future__ import annotations

import os
import time
from collections import OrderedDict
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse


class _TokenBucket:
    """Simple token bucket rate limiter with TTL tracking."""

    def __init__(self, rate: float, capacity: int):
        self.rate = rate  # Tokens per second
        self.capacity = capacity
        self.tokens = float(capacity)
        self.last_refill = time.monotonic()
        self.last_used = time.monotonic()

    def consume(self, tokens: int = 1) -> bool:
        now = time.monotonic()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
        self.last_refill = now
        self.last_used = now

        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False

    def is_expired(self, ttl: float) -> bool:
        return (time.monotonic() - self.last_used) > ttl


class _LRUBucketCache:
    """OrderedDict-backed LRU cache for IP token buckets.

    Evicts oldest/unused entries when max capacity is reached.
    Also supports TTL-based expiry for inactive buckets.
    """

    def __init__(self, rate: float, capacity: int, max_buckets: int = 10000, ttl: float = 600.0):
        self._rate = rate
        self._capacity = capacity
        self._max_buckets = max_buckets
        self._ttl = ttl
        self._buckets: OrderedDict[str, _TokenBucket] = OrderedDict()

    def get_or_create(self, key: str) -> _TokenBucket:
        if key in self._buckets:
            # Move to end (most recently used)
            self._buckets.move_to_end(key)
            return self._buckets[key]

        # Evict if at capacity: remove expired first, then LRU
        if len(self._buckets) >= self._max_buckets:
            self._evict()

        bucket = _TokenBucket(rate=self._rate, capacity=self._capacity)
        self._buckets[key] = bucket
        return bucket

    def _evict(self) -> None:
        """Evict expired buckets first, then LRU until under max."""
        # First pass: remove expired
        expired = [
            k for k, b in self._buckets.items()
            if b.is_expired(self._ttl)
        ]
        for k in expired:
            del self._buckets[k]

        # Second pass: remove LRU until under limit
        while len(self._buckets) >= self._max_buckets:
            self._buckets.popitem(last=False)

    def cleanup_expired(self) -> int:
        """Remove all expired buckets. Returns count removed."""
        expired = [
            k for k, b in self._buckets.items()
            if b.is_expired(self._ttl)
        ]
        for k in expired:
            del self._buckets[k]
        return len(expired)

    def __len__(self) -> int:
        return len(self._buckets)

    def __contains__(self, key: str) -> bool:
        return key in self._buckets


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Per-IP rate limiting with RBAC per-key override.

    If an RBAC API key has `requests_per_minute` set, use that instead
    of the global default. Falls back to per-IP buckets otherwise.

    Memory-safe: LRU eviction prevents unbounded bucket growth from unique-IP
    DoS. Buckets also expire after a configurable TTL of inactivity.

    Configured via environment:
      YUNSHU_RATE_LIMIT_RPM — requests per minute (default: 120)
      YUNSHU_RATE_LIMIT_MAX_BUCKETS — max per-IP buckets (default: 10000)
      YUNSHU_RATE_LIMIT_TTL_SECONDS — bucket TTL in seconds (default: 600)
    """

    PUBLIC_PATHS = {"/health", "/health/ready", "/health/live", "/docs", "/openapi.json", "/redoc", "/metrics"}

    def __init__(self, app, rpm: int | None = None):
        super().__init__(app)
        rpm = rpm or int(os.environ.get("YUNSHU_RATE_LIMIT_RPM", "120"))
        self._rpm = rpm
        max_buckets = int(os.environ.get("YUNSHU_RATE_LIMIT_MAX_BUCKETS", "10000"))
        ttl = float(os.environ.get("YUNSHU_RATE_LIMIT_TTL_SECONDS", "600"))
        self._bucket_cache = _LRUBucketCache(
            rate=rpm / 60.0, capacity=rpm,
            max_buckets=max_buckets, ttl=ttl,
        )
        self._key_buckets: OrderedDict[str, _TokenBucket] = OrderedDict()
        self._max_key_buckets = max_buckets

    def _get_key_bucket(self, key_name: str, rpm: int) -> _TokenBucket:
        if key_name in self._key_buckets:
            if self._key_buckets[key_name].capacity != rpm:
                self._key_buckets[key_name] = _TokenBucket(rate=rpm / 60.0, capacity=rpm)
            else:
                self._key_buckets.move_to_end(key_name)
            return self._key_buckets[key_name]
        if len(self._key_buckets) >= self._max_key_buckets:
            self._key_buckets.popitem(last=False)
        self._key_buckets[key_name] = _TokenBucket(rate=rpm / 60.0, capacity=rpm)
        return self._key_buckets[key_name]

    async def dispatch(self, request: Request, call_next):
        if request.url.path in self.PUBLIC_PATHS:
            return await call_next(request)

        # Check RBAC key-level rate limit first
        rbac_key = getattr(request.state, "rbac_key", None)
        if rbac_key is not None and rbac_key.requests_per_minute is not None:
            bucket = self._get_key_bucket(rbac_key.name, rbac_key.requests_per_minute)
            if not bucket.consume():
                return JSONResponse(
                    status_code=429,
                    content={
                        "detail": "API key rate limit exceeded",
                        "retry_after": int(60 / rbac_key.requests_per_minute) + 1,
                    },
                    headers={"Retry-After": str(int(60 / rbac_key.requests_per_minute) + 1)},
                )
            return await call_next(request)

        # Fall back to per-IP rate limiting (LRU + TTL safe)
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            client_ip = forwarded.split(",")[0].strip()
        else:
            client_ip = request.client.host if request.client else "unknown"
        bucket = self._bucket_cache.get_or_create(client_ip)

        if not bucket.consume():
            return JSONResponse(
                status_code=429,
                content={
                    "detail": "Rate limit exceeded",
                    "retry_after": int(60 / self._rpm) + 1,
                },
                headers={"Retry-After": str(int(60 / self._rpm) + 1)},
            )

        return await call_next(request)
