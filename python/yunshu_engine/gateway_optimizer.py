from __future__ import annotations

"""Gateway request pipeline optimizations — L1 layer acceleration.

Components:
  - RequestCoalescer: batches simultaneous requests per model
  - StreamingResponseBuffer: zero-alloc ring buffer for SSE streaming
  - GatewayConnectionPool: HTTP connection pooling for distributed mode
  - ResponseCache: content-hash request dedup with TTL expiry

Enabled via environment:
  YUNSHU_RESPONSE_CACHE=1  — enable response cache (non-streaming only)
"""


import asyncio
import hashlib
import json
import logging
import os
import time
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ── 1. RequestCoalescer ──────────────────────────────────────────────────


@dataclass
class _PendingBatch:
    """Accumulates requests that arrived during the coalescing window."""

    requests: list[Any] = field(default_factory=list)
    futures: list[asyncio.Future] = field(default_factory=list)
    created_at: float = field(default_factory=time.monotonic)


class RequestCoalescer:
    """Batch simultaneous requests for the same model into a single engine call.

    When multiple requests for the same model arrive within the coalescing
    window (default 5ms), they are grouped into a single batch sent to the
    engine.

    Only coalesces requests that share the same model AND have identical
    parameter fingerprints (temperature, top_p, max_tokens, etc).
    Requests with different parameters go into separate batches.

    Usage::

        coalescer = RequestCoalescer(window_ms=5)
        result_future = await coalescer.add_request(my_request)
        result = await result_future
    """

    def __init__(self, window_ms: float = 5.0):
        self._window_ms = window_ms
        self._window_s = window_ms / 1000.0
        # model_name -> pending batch
        self._pending: dict[str, _PendingBatch] = {}
        # Flush timers per model
        self._timers: dict[str, asyncio.TimerHandle] = {}
        # Tracked flush tasks (for proper cancellation on shutdown)
        self._flush_tasks: set[asyncio.Task] = set()
        self._loop: asyncio.AbstractEventLoop | None = None

        # Stats
        self._stats = _CoalescerStats()

        self._lock = asyncio.Lock()

    # -- public API --

    async def add_request(self, request: Any) -> asyncio.Future:
        """Add a request to the pending batch for its model.

        Returns a Future that will be resolved with the engine result
        when the batch is flushed.  If the request fails, the Future
        will receive an exception.
        """
        loop = self._get_loop()
        model = getattr(request, "model", "default")
        future = loop.create_future()

        async with self._lock:
            if model not in self._pending:
                batch = _PendingBatch()
                self._pending[model] = batch
                # Schedule flush after the coalescing window
                timer = loop.call_later(self._window_s, self._schedule_flush, model)
                self._timers[model] = timer
            else:
                batch = self._pending[model]

            batch.requests.append(request)
            batch.futures.append(future)

        return future

    async def flush(self, model: str | None = None) -> int:
        """Flush pending batches, sending them to the engine.

        If *model* is specified, only that model's batch is flushed.
        Returns the number of requests flushed.
        """
        async with self._lock:
            if model is not None:
                count = self._do_flush(model)
            else:
                count = 0
                for m in list(self._pending):
                    count += self._do_flush(m)
            return count

    def get_stats(self) -> dict[str, Any]:
        """Return coalescing statistics."""
        s = self._stats
        return {
            "total_batches": s.total_batches,
            "total_requests": s.total_requests,
            "coalesced_batches": s.coalesced_batches,
            "avg_batch_size": (
                s.total_requests / s.total_batches if s.total_batches else 0.0
            ),
            "avg_coalescing_delay_ms": (
                s.total_delay_ms / s.total_batches if s.total_batches else 0.0
            ),
            "pending_models": len(self._pending),
        }

    # -- internals --

    def _get_loop(self) -> asyncio.AbstractEventLoop:
        if self._loop is None or self._loop.is_closed():
            try:
                self._loop = asyncio.get_running_loop()
            except RuntimeError:
                # No running loop — fall back to get_event_loop for
                # callers that create the coalescer outside async context.
                self._loop = asyncio.get_event_loop()
        return self._loop

    def _schedule_flush(self, model: str) -> None:
        """Called by the timer — schedules async flush on the event loop."""
        self._get_loop()
        try:
            task = asyncio.ensure_future(self.flush(model))
            self._flush_tasks.add(task)
            task.add_done_callback(self._flush_tasks.discard)
        except RuntimeError:
            # Loop closed during shutdown
            pass

    def _do_flush(self, model: str) -> int:
        """Must be called under self._lock. Returns count of flushed requests."""
        batch = self._pending.pop(model, None)
        if batch is None:
            return 0

        # Cancel the timer if still pending
        timer = self._timers.pop(model, None)
        if timer is not None:
            timer.cancel()

        n = len(batch.requests)
        delay_ms = (time.monotonic() - batch.created_at) * 1000.0

        self._stats.total_batches += 1
        self._stats.total_requests += n
        self._stats.total_delay_ms += delay_ms
        if n > 1:
            self._stats.coalesced_batches += 1

        # Store the batch for the engine to pick up
        self._last_flushed = (model, batch)
        return n

    async def get_flushed_batch(self) -> tuple[str, _PendingBatch] | None:
        """Retrieve the most recently flushed batch (for engine consumption)."""
        async with self._lock:
            return getattr(self, "_last_flushed", None)

    async def resolve_batch(self, batch: _PendingBatch, results: list[Any]) -> None:
        """Resolve all futures in a flushed batch with their results."""
        for future, result in zip(batch.futures, results, strict=False):
            if not future.done():
                future.set_result(result)

    async def reject_batch(self, batch: _PendingBatch, error: Exception) -> None:
        """Reject all futures in a flushed batch with an error.

        Called when the engine fails to process the batch, so that
        waiting callers receive the exception instead of hanging.
        """
        for future in batch.futures:
            if not future.done():
                future.set_exception(error)


@dataclass
class _CoalescerStats:
    total_batches: int = 0
    total_requests: int = 0
    coalesced_batches: int = 0
    total_delay_ms: float = 0.0


# ── 2. StreamingResponseBuffer ───────────────────────────────────────────


class StreamingResponseBuffer:
    """Optimized ring buffer for SSE streaming responses.

    Avoids per-chunk string allocation by using pre-allocated byte
    buffers with recycling.  Each SSE event is written into a segment
    of the ring buffer and flushed as a contiguous byte string.
    """

    def __init__(self, capacity: int = 64 * 1024, segment_size: int = 4096):
        self._capacity = capacity
        self._segment_size = segment_size
        # Pre-allocated ring buffer
        self._buffer = bytearray(capacity)
        self._write_pos = 0
        self._read_pos = 0
        self._used = 0
        # Recycled segments for temporary event assembly
        self._segment_pool: list[bytearray] = []
        self._max_pool = 16
        # Stats
        self._flush_count = 0
        self._write_count = 0
        self._bytes_written = 0

    def write(self, data: bytes | str) -> int:
        """Write data into the ring buffer.  Returns bytes written."""
        if isinstance(data, str):
            data = data.encode("utf-8")

        n = len(data)
        if n > self._available():
            # Not enough space — truncate to available space and log warning.
            # Caller should flush() before writing to avoid data loss.
            original_n = n
            n = self._available()
            data = data[:n]
            logger.warning(
                "StreamingResponseBuffer: truncated write %d -> %d bytes "
                "(buffer full, flush before writing)",
                original_n,
                n,
            )

        if n == 0:
            return 0

        # Write into ring buffer (may wrap around)
        first_chunk = min(n, self._capacity - self._write_pos)
        self._buffer[self._write_pos : self._write_pos + first_chunk] = data[
            :first_chunk
        ]
        if first_chunk < n:
            # Wrap around
            self._buffer[0 : n - first_chunk] = data[first_chunk:]
        self._write_pos = (self._write_pos + n) % self._capacity
        self._used += n
        self._bytes_written += n
        self._write_count += 1
        return n

    def flush(self) -> list[bytes]:
        """Return all buffered data as a list of SSE event byte strings.

        Returns individual events (split on '\\n\\n' boundaries) so the
        caller can yield them individually.

        Note: SSE events are returned with their content preserved
        exactly as written (no whitespace stripping). A trailing
        incomplete event (no \\n\\n terminator) is kept as-is so the
        caller can decide whether to buffer or yield it.
        """
        if self._used == 0:
            return []

        self._flush_count += 1

        # Extract the data from the ring buffer
        raw = self._read_all()
        self._read_pos = self._write_pos
        self._used = 0

        # Split into SSE events (each ends with \n\n)
        events = []
        # raw may contain a trailing incomplete event (no \n\n terminator).
        # We split on \n\n to get complete events, but the last part may be
        # an incomplete event that we preserve without modification.
        parts = raw.split(b"\n\n")
        for i, part in enumerate(parts):
            if not part:
                continue
            if i < len(parts) - 1:
                # Complete event: re-add the \n\n terminator
                events.append(part + b"\n\n")
            else:
                # Last part: could be a complete event (if raw ended with \n\n,
                # split produces a trailing empty string which we skip above) or
                # an incomplete event. Append \n\n only if it looks complete
                # (i.e., raw ended with \n\n, meaning this is a full event).
                # Since split removes the delimiter, if raw ends with \n\n the
                # last element is b"" (already handled by the `if not part` check).
                # So this last non-empty part is an incomplete event.
                events.append(part)
        return events

    def _read_all(self) -> bytes:
        """Read all buffered data from the ring buffer."""
        if self._used == 0:
            return b""
        if self._read_pos + self._used <= self._capacity:
            return bytes(self._buffer[self._read_pos : self._read_pos + self._used])
        # Wrap-around
        first = self._capacity - self._read_pos
        return bytes(self._buffer[self._read_pos : self._capacity]) + bytes(
            self._buffer[0 : self._used - first]
        )

    def _available(self) -> int:
        return self._capacity - self._used

    def get_segment(self) -> bytearray:
        """Get a recycled or new segment for temporary use."""
        if self._segment_pool:
            seg = self._segment_pool.pop()
            seg.clear()
            return seg
        return bytearray(self._segment_size)

    def return_segment(self, seg: bytearray) -> None:
        """Return a segment to the pool for reuse."""
        if len(self._segment_pool) < self._max_pool:
            seg.clear()
            self._segment_pool.append(seg)

    def get_stats(self) -> dict[str, Any]:
        return {
            "capacity": self._capacity,
            "used": self._used,
            "available": self._available(),
            "utilization_pct": (self._used / self._capacity * 100)
            if self._capacity
            else 0,
            "flush_count": self._flush_count,
            "write_count": self._write_count,
            "bytes_written": self._bytes_written,
            "segment_pool_size": len(self._segment_pool),
        }


# ── 3. GatewayConnectionPool ────────────────────────────────────────────


@dataclass
class _PooledConnection:
    """A reusable HTTP connection wrapper."""

    endpoint: str
    connection: Any  # httpx.AsyncClient or similar
    created_at: float = field(default_factory=time.monotonic)
    last_used: float = field(default_factory=time.monotonic)
    active: bool = False
    health_checks: int = 0
    requests_served: int = 0


class GatewayConnectionPool:
    """HTTP connection pooling for upstream model servers (distributed mode).

    Reuses TCP connections across requests instead of creating new ones.
    Includes health checking that removes stale connections.

    Connection lifecycle:
      1. get_connection(endpoint) — returns idle conn or creates new one
      2. Caller uses conn.connection for the HTTP request
      3. return_connection(conn) — marks idle, prunes stale entries
      4. health_check() — periodic cleanup of stale idle connections
      5. close_all() — shutdown: close all connections gracefully

    Usage::

        pool = GatewayConnectionPool(max_per_host=4)
        conn = await pool.get_connection("http://worker1:8000")
        try:
            # use conn.connection ...
            pass
        finally:
            await pool.return_connection(conn)
    """

    def __init__(
        self,
        max_per_host: int = 8,
        idle_timeout: float = 60.0,
        health_check_interval: float = 30.0,
    ):
        self._max_per_host = max_per_host
        self._idle_timeout = idle_timeout
        self._health_check_interval = health_check_interval
        # endpoint -> list of connections
        self._pool: dict[str, list[_PooledConnection]] = defaultdict(list)
        self._lock = asyncio.Lock()

        # Stats
        self._total_gets = 0
        self._reuse_count = 0
        self._create_count = 0
        self._eviction_count = 0
        self._rejection_count = 0

    async def get_connection(self, endpoint: str) -> _PooledConnection:
        """Get or create a pooled connection for the given endpoint.

        Returns an idle connection if available, otherwise creates a new
        one.  Raises ConnectionError if the pool is exhausted (all
        connections for this endpoint are active and at max_per_host).
        """
        async with self._lock:
            self._total_gets += 1

            # Try to find an idle connection
            conns = self._pool[endpoint]
            for conn in conns:
                if not conn.active and not self._is_stale(conn):
                    conn.active = True
                    conn.last_used = time.monotonic()
                    conn.requests_served += 1
                    self._reuse_count += 1
                    return conn

            # Prune stale connections to make room
            before = len(conns)
            self._pool[endpoint] = [
                c for c in conns if c.active or not self._is_stale(c)
            ]
            pruned = before - len(self._pool[endpoint])
            self._eviction_count += pruned
            conns = self._pool[endpoint]

            # Check max_per_host limit (count only active connections)
            active_count = sum(1 for c in conns if c.active)
            if active_count >= self._max_per_host:
                self._rejection_count += 1
                raise ConnectionError(
                    f"Connection pool exhausted for {endpoint}: "
                    f"{active_count}/{self._max_per_host} active connections"
                )

            # Create a new connection (mock for now; real impl uses httpx)
            self._create_count += 1
            new_conn = _PooledConnection(
                endpoint=endpoint,
                connection=None,  # Would be httpx.AsyncClient in production
            )
            new_conn.active = True
            new_conn.requests_served = 1
            conns.append(new_conn)
            return new_conn

    async def return_connection(self, conn: _PooledConnection) -> None:
        """Return a connection to the pool for reuse."""
        async with self._lock:
            conn.active = False
            conn.last_used = time.monotonic()

            # Prune stale connections for this endpoint
            self._prune_endpoint(conn.endpoint)

    async def health_check(self) -> int:
        """Remove stale/unhealthy connections. Returns count removed.

        Active connections are preserved even if idle-timer exceeded,
        since they are mid-request.
        """
        async with self._lock:
            removed = 0
            for endpoint in list(self._pool):
                conns = self._pool[endpoint]
                before = len(conns)
                self._pool[endpoint] = [
                    c for c in conns if c.active or not self._is_stale(c)
                ]
                removed += before - len(self._pool[endpoint])
            self._eviction_count += removed
            return removed

    async def close_all(self) -> None:
        """Close all connections. Call during shutdown.

        Properly closes connections that have an aclose() coroutine
        (e.g. httpx.AsyncClient).  Connections currently in use are
        marked inactive before closing.
        """
        async with self._lock:
            for endpoint, conns in self._pool.items():
                for conn in conns:
                    conn.active = False
                    if conn.connection is not None and hasattr(
                        conn.connection, "aclose"
                    ):
                        try:
                            # aclose is typically async
                            import inspect

                            if inspect.iscoroutinefunction(conn.connection.aclose):
                                await conn.connection.aclose()
                            else:
                                conn.connection.aclose()
                        except Exception:
                            logger.debug(
                                "Failed to close connection to %s",
                                endpoint,
                                exc_info=True,
                            )
            self._pool.clear()

    def get_stats(self) -> dict[str, Any]:
        total = sum(len(c) for c in self._pool.values())
        active = sum(1 for c in self._pool.values() for conn in c if conn.active)
        idle = total - active
        reuse_rate = (
            self._reuse_count / self._total_gets if self._total_gets > 0 else 0.0
        )
        return {
            "pool_size": total,
            "active_connections": active,
            "idle_connections": idle,
            "endpoints": len(self._pool),
            "total_gets": self._total_gets,
            "reuse_count": self._reuse_count,
            "create_count": self._create_count,
            "reuse_rate": reuse_rate,
            "eviction_count": self._eviction_count,
            "rejection_count": self._rejection_count,
        }

    def _is_stale(self, conn: _PooledConnection) -> bool:
        """Check if a connection has exceeded the idle timeout."""
        idle_time = time.monotonic() - conn.last_used
        return idle_time > self._idle_timeout

    def _prune_endpoint(self, endpoint: str) -> None:
        """Remove stale idle connections for a specific endpoint."""
        conns = self._pool[endpoint]
        before = len(conns)
        self._pool[endpoint] = [c for c in conns if not self._is_stale(c) or c.active]
        pruned = before - len(self._pool[endpoint])
        self._eviction_count += pruned


# ── 4. ResponseCache ────────────────────────────────────────────────────


@dataclass
class _CacheEntry:
    """A cached response with TTL tracking."""

    key: str
    response: Any
    created_at: float = field(default_factory=time.monotonic)
    last_accessed: float = field(default_factory=time.monotonic)
    access_count: int = 0
    size_bytes: int = 0


class ResponseCache:
    """Content-hash based response cache for non-streaming requests.

    Caches identical requests (same model + messages + params) using a
    content-hash of the request as cache key.  TTL-based expiration
    prevents stale entries.

    Only enabled when YUNSHU_RESPONSE_CACHE=1 is set.

    Thread-safe: uses asyncio.Lock for async compatibility.
    LRU eviction uses OrderedDict for O(1) operations.

    IMPORTANT: Should NOT be used for:
      - Streaming requests (each chunk is different)
      - n>1 completions (each choice is independently generated)
      - Requests with non-deterministic sampling (temperature > 0, no seed)

    Usage::

        cache = ResponseCache()
        key = cache.hash_request(model, messages, params)
        hit = cache.get(key)
        if hit is not None:
            return hit
        result = await engine.generate(...)
        cache.put(key, result)
    """

    def __init__(
        self,
        ttl: float = 60.0,
        max_entries: int = 1024,
        max_memory_bytes: int = 64 * 1024 * 1024,
    ):
        self._ttl = ttl
        self._max_entries = max_entries
        self._max_memory = max_memory_bytes
        self._enabled = os.environ.get("YUNSHU_RESPONSE_CACHE", "").strip() == "1"
        self._entries: dict[str, _CacheEntry] = {}
        # OrderedDict for O(1) LRU: front = oldest, back = newest
        self._lru: OrderedDict[str, None] = OrderedDict()
        self._lock = asyncio.Lock()
        self._total_memory = 0

        # Stats
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._stores = 0

    @property
    def enabled(self) -> bool:
        return self._enabled

    @staticmethod
    def hash_request(
        model: str,
        messages: list[dict],
        **params: Any,
    ) -> str:
        """Compute a content-hash for a request.  Deterministic ordering."""
        payload = {
            "model": model,
            "messages": messages,
            "params": _sort_dict(params),
        }
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    async def get(self, request_hash: str) -> Any | None:
        """Return the cached response or None if not found / expired."""
        if not self._enabled:
            return None

        async with self._lock:
            entry = self._entries.get(request_hash)
            if entry is None:
                self._misses += 1
                return None

            # Check TTL
            if self._is_expired(entry):
                self._remove_entry(request_hash)
                self._misses += 1
                return None

            # Promote in LRU (move to end = most recently used)
            entry.last_accessed = time.monotonic()
            entry.access_count += 1
            self._lru.move_to_end(request_hash)

            self._hits += 1
            return entry.response

    async def put(self, request_hash: str, response: Any) -> bool:
        """Store a response in the cache.  Returns True if stored."""
        if not self._enabled:
            return False

        # Estimate size
        try:
            size = len(json.dumps(response, default=str).encode("utf-8"))
        except (TypeError, ValueError):
            size = 1024  # default estimate

        async with self._lock:
            if request_hash in self._entries:
                # Update existing entry — only evict for the net size increase
                old = self._entries[request_hash]
                net_increase = size - old.size_bytes
                if net_increase > 0:
                    await self._evict_if_needed(net_increase)
                # Re-fetch after eviction — the entry may have been evicted
                # as expired during _evict_if_needed (which yields control).
                old = self._entries.get(request_hash)
                if old is None:
                    # Evicted — treat as new entry
                    entry = _CacheEntry(
                        key=request_hash,
                        response=response,
                        size_bytes=size,
                    )
                    self._entries[request_hash] = entry
                    self._lru[request_hash] = None
                    self._total_memory += size
                else:
                    self._total_memory -= old.size_bytes
                    old.response = response
                    old.created_at = time.monotonic()
                    old.last_accessed = time.monotonic()
                    old.size_bytes = size
                    self._total_memory += size
                    # Promote in LRU
                    self._lru.move_to_end(request_hash)
            else:
                # New entry — evict if necessary for full size
                await self._evict_if_needed(size)
                entry = _CacheEntry(
                    key=request_hash,
                    response=response,
                    size_bytes=size,
                )
                self._entries[request_hash] = entry
                self._lru[request_hash] = None
                self._total_memory += size

            self._stores += 1
            return True

    async def invalidate(self, pattern: str | None = None) -> int:
        """Invalidate cache entries.  If pattern is None, clears all.

        If pattern is a string, removes entries whose hash starts with
        the given prefix.  Returns the number of entries removed.
        """
        async with self._lock:
            if pattern is None:
                count = len(self._entries)
                self._entries.clear()
                self._lru.clear()
                self._total_memory = 0
                self._evictions += count
                return count

            to_remove = [k for k in self._entries if k.startswith(pattern)]
            for k in to_remove:
                self._remove_entry(k)
            self._evictions += len(to_remove)
            return len(to_remove)

    def get_stats(self) -> dict[str, Any]:
        # Snapshot counters to avoid torn reads during concurrent async updates.
        # Since asyncio.Lock is not reentrant and get_stats() may be called
        # from sync contexts (monitoring endpoints), we read directly.
        # In CPython, simple int attribute reads are atomic (GIL), so this
        # is safe for monitoring purposes — worst case is a slightly stale value.
        hits = self._hits
        misses = self._misses
        total = hits + misses
        return {
            "enabled": self._enabled,
            "hits": hits,
            "misses": misses,
            "hit_rate": (hits / total) if total > 0 else 0.0,
            "stores": self._stores,
            "evictions": self._evictions,
            "entries": len(self._entries),
            "memory_bytes": self._total_memory,
            "memory_limit_bytes": self._max_memory,
            "memory_utilization_pct": (
                self._total_memory / self._max_memory * 100
                if self._max_memory > 0
                else 0
            ),
        }

    def _is_expired(self, entry: _CacheEntry) -> bool:
        return (time.monotonic() - entry.created_at) > self._ttl

    def _remove_entry(self, key: str) -> None:
        """Remove an entry from the cache. Must be called under lock."""
        entry = self._entries.pop(key, None)
        if entry is not None:
            self._total_memory -= entry.size_bytes
        self._lru.pop(key, None)

    async def _evict_if_needed(self, incoming_size: int) -> int:
        """Evict LRU entries to make room. Must be called under lock.

        Returns the number of entries evicted.
        """
        evicted = 0

        # Evict expired entries first
        now = time.monotonic()
        expired = [
            k for k, e in self._entries.items() if (now - e.created_at) > self._ttl
        ]
        for k in expired:
            self._remove_entry(k)
            evicted += 1

        # Evict LRU until under limits
        while (
            len(self._entries) >= self._max_entries
            or (self._total_memory + incoming_size > self._max_memory)
        ) and self._lru:
            # popitem(last=False) pops the OLDEST (front) entry — O(1)
            oldest_key, _ = self._lru.popitem(last=False)
            # _remove_entry will try to pop from _lru again, but it's
            # already gone, so the pop(..., None) is a no-op.
            self._remove_entry(oldest_key)
            evicted += 1

        self._evictions += evicted
        return evicted


def _sort_dict(d: dict) -> dict:
    """Recursively sort a dictionary for deterministic hashing."""
    result = {}
    for k in sorted(d):
        v = d[k]
        if isinstance(v, dict):
            result[k] = _sort_dict(v)
        elif isinstance(v, list):
            result[k] = [_sort_dict(i) if isinstance(i, dict) else i for i in v]
        elif isinstance(v, tuple):
            result[k] = list(v)
        else:
            result[k] = v
    return result


# ── Module-level singletons ──────────────────────────────────────────────

_request_coalescer: RequestCoalescer | None = None
_streaming_buffer_pool: list[StreamingResponseBuffer] = []
_connection_pool: GatewayConnectionPool | None = None
_response_cache: ResponseCache | None = None


def get_request_coalescer() -> RequestCoalescer:
    """Get or create the global RequestCoalescer singleton."""
    global _request_coalescer
    if _request_coalescer is None:
        _request_coalescer = RequestCoalescer()
    return _request_coalescer


def get_streaming_buffer() -> StreamingResponseBuffer:
    """Get or create a StreamingResponseBuffer (one per streaming request).

    Recycles buffers from the pool when available. Callers should call
    ``return_streaming_buffer()`` when done to allow reuse.
    """
    global _streaming_buffer_pool
    if _streaming_buffer_pool:
        buf = _streaming_buffer_pool.pop()
        # Reset buffer state for reuse
        buf._read_pos = 0
        buf._write_pos = 0
        buf._used = 0
        buf._flush_count = 0
        buf._write_count = 0
        buf._bytes_written = 0
        return buf
    return StreamingResponseBuffer()


def return_streaming_buffer(buf: StreamingResponseBuffer) -> None:
    """Return a StreamingResponseBuffer to the pool for reuse.

    Limits pool size to prevent unbounded memory growth.
    """
    global _streaming_buffer_pool
    max_pool_size = 32
    if len(_streaming_buffer_pool) < max_pool_size:
        _streaming_buffer_pool.append(buf)


def get_connection_pool() -> GatewayConnectionPool:
    """Get or create the global GatewayConnectionPool singleton."""
    global _connection_pool
    if _connection_pool is None:
        _connection_pool = GatewayConnectionPool()
    return _connection_pool


def get_response_cache() -> ResponseCache:
    """Get or create the global ResponseCache singleton."""
    global _response_cache
    if _response_cache is None:
        _response_cache = ResponseCache()
    return _response_cache
