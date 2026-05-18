from __future__ import annotations
"""Yunshu Gateway — Rate limiting middleware.

Token bucket rate limiter per client IP with RBAC per-key override.
oMLX pattern: configurable RPM + token limits.

Memory safety: LRU eviction + TTL expiry prevents unbounded growth from
unique-IP DoS. Configurable via environment:
  YUNSHU_RATE_LIMIT_MAX_BUCKETS — max per-IP buckets (default: 10000)
  YUNSHU_RATE_LIMIT_TTL_SECONDS — bucket TTL in seconds (default: 600)
"""


import os
import time
from collections import OrderedDict
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response


# Paths served by the Anthropic router — must use Anthropic error format
_ANTHROPIC_PATHS = ("/v1/messages", "/messages")


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
        self.last_used = now

        if self.tokens >= tokens:
            self.tokens -= tokens
            self.last_refill = now
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

    IP source security:
    - By default, uses the direct client IP (ignores X-Forwarded-For).
    - Set YUNSHU_TRUSTED_PROXIES to a comma-separated list of trusted proxy IPs.
      Only when the direct client is a trusted proxy will X-Forwarded-For be used.
    - Without trusted proxies configured, X-Forwarded-For is ignored to prevent
      spoofing attacks where clients inject arbitrary IPs to bypass rate limits.

    Configured via environment:
      YUNSHU_RATE_LIMIT_RPM — requests per minute (default: 120)
      YUNSHU_RATE_LIMIT_MAX_BUCKETS — max per-IP buckets (default: 10000)
      YUNSHU_RATE_LIMIT_TTL_SECONDS — bucket TTL in seconds (default: 600)
      YUNSHU_TRUSTED_PROXIES — comma-separated trusted proxy IPs (default: none)
    """

    PUBLIC_PATHS = {"/health", "/health/ready", "/health/live", "/docs", "/openapi.json", "/redoc", "/metrics"}
    # Prefixes that are exempt from rate limiting (monitoring, admin health)
    PUBLIC_PREFIXES = ("/api/v1/gw/monitoring/", "/api/v1/admin/hardware", "/api/v1/admin/memory", "/api/v1/admin/metrics")

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
        # Parse trusted proxies for X-Forwarded-For validation
        trusted_raw = os.environ.get("YUNSHU_TRUSTED_PROXIES", "").strip()
        self._trusted_proxies: set[str] = (
            {ip.strip() for ip in trusted_raw.split(",") if ip.strip()}
            if trusted_raw else set()
        )

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
        if any(request.url.path.startswith(p) for p in self.PUBLIC_PREFIXES):
            return await call_next(request)

        # CORS preflight (OPTIONS) must pass through without rate limiting —
        # browsers send OPTIONS without Authorization headers, and the inner
        # CORSMiddleware needs to respond first.
        if request.method == "OPTIONS":
            return await call_next(request)

        # Apply rate limiting to WebSocket upgrade requests as well
        is_websocket = (
            request.headers.get("upgrade", "").lower() == "websocket"
        )

        # Check RBAC key-level rate limit first
        rbac_key = getattr(request.state, "rbac_key", None)
        if rbac_key is not None and rbac_key.requests_per_minute is not None:
            bucket = self._get_key_bucket(rbac_key.name, rbac_key.requests_per_minute)
            if not bucket.consume():
                retry_after = int(60 / rbac_key.requests_per_minute) + 1
                if is_websocket:
                    # WebSocket upgrades can't return JSON bodies; return HTTP 429
                    return Response(status_code=429, content="Rate limit exceeded")
                # Anthropic endpoints: return Anthropic error format
                if request.url.path.endswith(_ANTHROPIC_PATHS):
                    return JSONResponse(
                        status_code=429,
                        content={
                            "type": "error",
                            "error": {
                                "type": "rate_limit_error",
                                "message": "API key rate limit exceeded",
                            },
                        },
                        headers={"Retry-After": str(retry_after)},
                    )
                # Default: OpenAI error format
                return JSONResponse(
                    status_code=429,
                    content={
                        "error": {
                            "message": "API key rate limit exceeded",
                            "type": "rate_limit_error",
                            "code": "rate_limit_exceeded",
                        }
                    },
                    headers={"Retry-After": str(retry_after)},
                )
            # Key-level rate limit passed — still apply IP-level rate limit
            # for defense-in-depth (prevents single key from unlimited IPs).

        # Per-IP rate limiting (LRU + TTL safe) — always applied
        # Security: only trust X-Forwarded-For when the direct client is a
        # configured trusted proxy. This prevents header spoofing attacks.
        direct_ip = request.client.host if request.client else "unknown"
        client_ip = direct_ip
        if self._trusted_proxies and direct_ip in self._trusted_proxies:
            forwarded = request.headers.get("x-forwarded-for")
            if forwarded:
                client_ip = forwarded.split(",")[0].strip()
        bucket = self._bucket_cache.get_or_create(client_ip)

        if not bucket.consume():
            retry_after = int(60 / self._rpm) + 1
            if is_websocket:
                return Response(status_code=429, content="Rate limit exceeded")
            # Anthropic endpoints: return Anthropic error format
            if request.url.path.endswith(_ANTHROPIC_PATHS):
                return JSONResponse(
                    status_code=429,
                    content={
                        "type": "error",
                        "error": {
                            "type": "rate_limit_error",
                            "message": "Rate limit exceeded",
                        },
                    },
                    headers={"Retry-After": str(retry_after)},
                )
            # Default: OpenAI error format
            return JSONResponse(
                status_code=429,
                content={
                    "error": {
                        "message": "Rate limit exceeded",
                        "type": "rate_limit_error",
                        "code": "rate_limit_exceeded",
                    }
                },
                headers={"Retry-After": str(retry_after)},
            )

        return await call_next(request)
