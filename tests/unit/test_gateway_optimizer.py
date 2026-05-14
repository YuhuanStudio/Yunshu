"""Unit tests for gateway_optimizer.py — L1 pipeline optimizations."""

import asyncio
import os
import time
from unittest.mock import MagicMock

import pytest

from yunshu_engine.gateway_optimizer import (
    GatewayConnectionPool,
    RequestCoalescer,
    ResponseCache,
    StreamingResponseBuffer,
    _PooledConnection,
    get_connection_pool,
    get_request_coalescer,
    get_response_cache,
    get_streaming_buffer,
)


# ── RequestCoalescer Tests ──────────────────────────────────────────────


class TestRequestCoalescer:
    """Tests for RequestCoalescer — request batching by model."""

    def test_create_default_window(self):
        c = RequestCoalescer()
        assert c._window_ms == 5.0
        assert c._window_s == 0.005

    def test_create_custom_window(self):
        c = RequestCoalescer(window_ms=10.0)
        assert c._window_ms == 10.0
        assert c._window_s == 0.01

    def test_add_request_returns_future(self):
        c = RequestCoalescer()
        req = MagicMock(model="test-model")
        future = asyncio.get_event_loop().run_until_complete(c.add_request(req))
        assert isinstance(future, asyncio.Future)
        assert not future.done()
        # Cleanup
        c._pending.clear()

    def test_single_request_batch(self):
        """A single request should form a batch of size 1."""
        c = RequestCoalescer()
        req = MagicMock(model="gpt-4")

        async def _run():
            future = await c.add_request(req)
            count = await c.flush("gpt-4")
            assert count == 1
            stats = c.get_stats()
            assert stats["total_batches"] == 1
            assert stats["total_requests"] == 1
            assert stats["avg_batch_size"] == 1.0
            assert stats["coalesced_batches"] == 0

        asyncio.get_event_loop().run_until_complete(_run())

    def test_multiple_requests_same_model_coalesced(self):
        """Multiple requests for the same model should be batched together."""
        c = RequestCoalescer()
        reqs = [MagicMock(model="llama-3") for _ in range(5)]

        async def _run():
            futures = []
            for req in reqs:
                f = await c.add_request(req)
                futures.append(f)
            count = await c.flush("llama-3")
            assert count == 5
            stats = c.get_stats()
            assert stats["total_batches"] == 1
            assert stats["total_requests"] == 5
            assert stats["avg_batch_size"] == 5.0
            assert stats["coalesced_batches"] == 1

        asyncio.get_event_loop().run_until_complete(_run())

    def test_different_models_separate_batches(self):
        """Requests for different models should be in separate batches."""
        c = RequestCoalescer()

        async def _run():
            await c.add_request(MagicMock(model="model-a"))
            await c.add_request(MagicMock(model="model-b"))
            await c.add_request(MagicMock(model="model-a"))

            count_a = await c.flush("model-a")
            count_b = await c.flush("model-b")
            assert count_a == 2
            assert count_b == 1
            stats = c.get_stats()
            assert stats["total_batches"] == 2
            assert stats["coalesced_batches"] == 1  # Only model-a batch was coalesced

        asyncio.get_event_loop().run_until_complete(_run())

    def test_flush_all_models(self):
        """flush() without model arg flushes all pending batches."""
        c = RequestCoalescer()

        async def _run():
            await c.add_request(MagicMock(model="m1"))
            await c.add_request(MagicMock(model="m2"))
            await c.add_request(MagicMock(model="m3"))

            total = await c.flush()
            assert total == 3
            assert c.get_stats()["pending_models"] == 0

        asyncio.get_event_loop().run_until_complete(_run())

    def test_flush_empty_batch(self):
        """Flushing a model with no pending requests returns 0."""
        c = RequestCoalescer()

        async def _run():
            count = await c.flush("nonexistent")
            assert count == 0

        asyncio.get_event_loop().run_until_complete(_run())

    def test_resolve_batch_futures(self):
        """Resolve batch futures with results."""
        c = RequestCoalescer()

        async def _run():
            f1 = await c.add_request(MagicMock(model="m"))
            f2 = await c.add_request(MagicMock(model="m"))
            await c.flush("m")
            batch_info = await c.get_flushed_batch()
            assert batch_info is not None
            model, batch = batch_info
            assert model == "m"
            assert len(batch.requests) == 2

            await c.resolve_batch(batch, ["result1", "result2"])
            assert f1.result() == "result1"
            assert f2.result() == "result2"

        asyncio.get_event_loop().run_until_complete(_run())

    def test_stats_avg_batch_size(self):
        """Track average batch size across multiple flushes."""
        c = RequestCoalescer()

        async def _run():
            # Batch 1: 3 requests
            for _ in range(3):
                await c.add_request(MagicMock(model="m"))
            await c.flush("m")
            # Batch 2: 1 request
            await c.add_request(MagicMock(model="m"))
            await c.flush("m")

            stats = c.get_stats()
            assert stats["total_batches"] == 2
            assert stats["total_requests"] == 4
            assert stats["avg_batch_size"] == 2.0

        asyncio.get_event_loop().run_until_complete(_run())

    def test_coalescing_delay_tracked(self):
        """Coalescing delay is tracked in stats."""
        c = RequestCoalescer(window_ms=50)

        async def _run():
            await c.add_request(MagicMock(model="m"))
            await c.flush("m")
            stats = c.get_stats()
            assert stats["avg_coalescing_delay_ms"] >= 0.0

        asyncio.get_event_loop().run_until_complete(_run())

    def test_default_model_name(self):
        """Requests without a .model attribute use 'default'."""
        c = RequestCoalescer()

        async def _run():
            await c.add_request(MagicMock(spec=[]))  # no .model attribute
            count = await c.flush("default")
            assert count == 1

        asyncio.get_event_loop().run_until_complete(_run())


# ── StreamingResponseBuffer Tests ───────────────────────────────────────


class TestStreamingResponseBuffer:
    """Tests for StreamingResponseBuffer — zero-alloc ring buffer."""

    def test_create_default(self):
        buf = StreamingResponseBuffer()
        assert buf._capacity == 64 * 1024
        assert buf._segment_size == 4096

    def test_write_and_read_string(self):
        buf = StreamingResponseBuffer(capacity=1024)
        written = buf.write("data: hello\n\n")
        assert written == 13  # "data: hello" (11) + "\n\n" (2)

    def test_write_and_read_bytes(self):
        buf = StreamingResponseBuffer(capacity=1024)
        written = buf.write(b"data: world\n\n")
        assert written == 13

    def test_flush_returns_events(self):
        buf = StreamingResponseBuffer(capacity=1024)
        buf.write(b"data: event1\n\ndata: event2\n\n")
        events = buf.flush()
        assert len(events) == 2
        assert events[0] == b"data: event1\n\n"
        assert events[1] == b"data: event2\n\n"

    def test_flush_empty_buffer(self):
        buf = StreamingResponseBuffer()
        events = buf.flush()
        assert events == []

    def test_ring_buffer_wraparound(self):
        """Test that the ring buffer correctly wraps around."""
        buf = StreamingResponseBuffer(capacity=32)
        # Fill almost full
        buf.write(b"A" * 28)
        assert buf._used == 28
        # Write past capacity to trigger wrap
        written = buf.write(b"BCDEFGHIJ")
        assert written == 4  # Only 4 bytes available (32 - 28)
        assert buf._used == 32
        events = buf.flush()
        assert len(events) > 0

    def test_stats_tracking(self):
        buf = StreamingResponseBuffer(capacity=1024)
        buf.write(b"data: test\n\n")
        buf.flush()
        stats = buf.get_stats()
        assert stats["capacity"] == 1024
        assert stats["flush_count"] == 1
        assert stats["write_count"] == 1
        assert stats["bytes_written"] == 12

    def test_utilization_pct(self):
        buf = StreamingResponseBuffer(capacity=100)
        buf.write(b"X" * 50)
        stats = buf.get_stats()
        assert stats["utilization_pct"] == 50.0
        assert stats["used"] == 50
        assert stats["available"] == 50

    def test_segment_pooling(self):
        buf = StreamingResponseBuffer()
        seg1 = buf.get_segment()
        assert isinstance(seg1, bytearray)
        buf.return_segment(seg1)
        assert len(buf._segment_pool) == 1

        seg2 = buf.get_segment()
        assert seg2 is seg1  # Reused
        assert len(buf._segment_pool) == 0

    def test_segment_pool_max_size(self):
        buf = StreamingResponseBuffer(segment_size=16)
        buf._max_pool = 3
        segs = [buf.get_segment() for _ in range(5)]
        for s in segs:
            buf.return_segment(s)
        assert len(buf._segment_pool) == 3  # Capped

    def test_multiple_writes_and_flush(self):
        buf = StreamingResponseBuffer(capacity=1024)
        buf.write(b"data: a\n\n")
        buf.write(b"data: b\n\n")
        buf.write(b"data: c\n\n")
        events = buf.flush()
        assert len(events) == 3

    def test_empty_string_write(self):
        buf = StreamingResponseBuffer(capacity=1024)
        written = buf.write("")
        assert written == 0

    def test_utf8_encoding(self):
        buf = StreamingResponseBuffer(capacity=1024)
        buf.write("data: 你好\n\n")
        events = buf.flush()
        assert len(events) == 1
        assert "你好" in events[0].decode("utf-8")


# ── GatewayConnectionPool Tests ─────────────────────────────────────────


class TestGatewayConnectionPool:
    """Tests for GatewayConnectionPool — connection reuse."""

    @pytest.mark.asyncio
    async def test_create_connection_on_first_get(self):
        pool = GatewayConnectionPool()
        conn = await pool.get_connection("http://worker1:8000")
        assert isinstance(conn, _PooledConnection)
        assert conn.endpoint == "http://worker1:8000"
        assert conn.active is True
        await pool.close_all()

    @pytest.mark.asyncio
    async def test_return_and_reuse_connection(self):
        pool = GatewayConnectionPool()
        conn1 = await pool.get_connection("http://w1:8000")
        await pool.return_connection(conn1)
        assert conn1.active is False

        conn2 = await pool.get_connection("http://w1:8000")
        assert conn2 is conn1  # Reused
        assert conn2.active is True
        assert conn2.requests_served == 2
        await pool.close_all()

    @pytest.mark.asyncio
    async def test_different_endpoints_separate_connections(self):
        pool = GatewayConnectionPool()
        c1 = await pool.get_connection("http://w1:8000")
        c2 = await pool.get_connection("http://w2:8000")
        assert c1 is not c2
        assert c1.endpoint != c2.endpoint
        await pool.close_all()

    @pytest.mark.asyncio
    async def test_max_per_host_respected(self):
        pool = GatewayConnectionPool(max_per_host=2)
        conns = [await pool.get_connection("http://w1:8000") for _ in range(5)]
        assert len(conns) == 5  # All returned, but pool grew
        # Only 2 can be idle at most for reuse
        for c in conns:
            await pool.return_connection(c)
        stats = pool.get_stats()
        assert stats["pool_size"] == 5  # All created
        await pool.close_all()

    @pytest.mark.asyncio
    async def test_stats_reuse_rate(self):
        pool = GatewayConnectionPool()
        c1 = await pool.get_connection("http://w1:8000")
        await pool.return_connection(c1)
        c2 = await pool.get_connection("http://w1:8000")  # Reuse
        await pool.return_connection(c2)

        stats = pool.get_stats()
        assert stats["total_gets"] == 2
        assert stats["reuse_count"] == 1
        assert stats["create_count"] == 1
        assert stats["reuse_rate"] == 0.5
        await pool.close_all()

    @pytest.mark.asyncio
    async def test_health_check_removes_stale(self):
        pool = GatewayConnectionPool(idle_timeout=0.01)  # 10ms timeout
        c1 = await pool.get_connection("http://w1:8000")
        await pool.return_connection(c1)
        assert pool.get_stats()["pool_size"] == 1

        # Wait for idle timeout
        await asyncio.sleep(0.05)
        removed = await pool.health_check()
        assert removed >= 1
        assert pool.get_stats()["pool_size"] == 0

    @pytest.mark.asyncio
    async def test_health_check_keeps_active(self):
        pool = GatewayConnectionPool(idle_timeout=0.01)
        c1 = await pool.get_connection("http://w1:8000")
        # c1 is still active
        await asyncio.sleep(0.05)
        removed = await pool.health_check()
        assert removed == 0
        assert pool.get_stats()["pool_size"] == 1
        await pool.return_connection(c1)
        await pool.close_all()

    @pytest.mark.asyncio
    async def test_close_all(self):
        pool = GatewayConnectionPool()
        await pool.get_connection("http://w1:8000")
        await pool.get_connection("http://w2:8000")
        assert pool.get_stats()["pool_size"] == 2
        await pool.close_all()
        assert pool.get_stats()["pool_size"] == 0

    @pytest.mark.asyncio
    async def test_stats_initial(self):
        pool = GatewayConnectionPool()
        stats = pool.get_stats()
        assert stats["pool_size"] == 0
        assert stats["active_connections"] == 0
        assert stats["idle_connections"] == 0
        assert stats["reuse_rate"] == 0.0


# ── ResponseCache Tests ─────────────────────────────────────────────────


class TestResponseCache:
    """Tests for ResponseCache — content-hash request dedup."""

    def test_hash_request_deterministic(self):
        """Same input produces same hash."""
        h1 = ResponseCache.hash_request("gpt-4", [{"role": "user", "content": "hi"}])
        h2 = ResponseCache.hash_request("gpt-4", [{"role": "user", "content": "hi"}])
        assert h1 == h2
        assert isinstance(h1, str)
        assert len(h1) == 64  # SHA-256 hex

    def test_hash_request_different_content(self):
        """Different messages produce different hashes."""
        h1 = ResponseCache.hash_request("gpt-4", [{"role": "user", "content": "hi"}])
        h2 = ResponseCache.hash_request("gpt-4", [{"role": "user", "content": "bye"}])
        assert h1 != h2

    def test_hash_request_different_model(self):
        """Different model produces different hash."""
        h1 = ResponseCache.hash_request("gpt-4", [{"role": "user", "content": "hi"}])
        h2 = ResponseCache.hash_request("llama-3", [{"role": "user", "content": "hi"}])
        assert h1 != h2

    def test_hash_request_with_params(self):
        """Extra params are included in the hash."""
        h1 = ResponseCache.hash_request("gpt-4", [{"role": "user", "content": "hi"}], temperature=0.7)
        h2 = ResponseCache.hash_request("gpt-4", [{"role": "user", "content": "hi"}], temperature=0.9)
        assert h1 != h2

    def test_hash_request_param_order_invariant(self):
        """Parameter order does not affect the hash."""
        h1 = ResponseCache.hash_request("m", [], a=1, b=2)
        h2 = ResponseCache.hash_request("m", [], b=2, a=1)
        assert h1 == h2

    def test_cache_disabled_by_default(self):
        cache = ResponseCache()
        assert cache.enabled is False

    def test_cache_enabled_with_env(self):
        old = os.environ.get("YUNSHU_RESPONSE_CACHE")
        try:
            os.environ["YUNSHU_RESPONSE_CACHE"] = "1"
            cache = ResponseCache()
            assert cache.enabled is True
        finally:
            if old is None:
                os.environ.pop("YUNSHU_RESPONSE_CACHE", None)
            else:
                os.environ["YUNSHU_RESPONSE_CACHE"] = old

    def test_get_returns_none_when_disabled(self):
        cache = ResponseCache()
        h = cache.hash_request("m", [])
        assert cache.get(h) is None

    def test_put_returns_false_when_disabled(self):
        cache = ResponseCache()
        h = cache.hash_request("m", [])
        assert cache.put(h, {"text": "hi"}) is False

    def test_cache_hit_and_miss(self):
        old = os.environ.get("YUNSHU_RESPONSE_CACHE")
        try:
            os.environ["YUNSHU_RESPONSE_CACHE"] = "1"
            cache = ResponseCache()
            h = cache.hash_request("gpt-4", [{"role": "user", "content": "hi"}])

            # Miss
            result = cache.get(h)
            assert result is None
            stats = cache.get_stats()
            assert stats["misses"] == 1

            # Store
            cache.put(h, {"choices": [{"text": "hello"}]})
            stats = cache.get_stats()
            assert stats["stores"] == 1

            # Hit
            result = cache.get(h)
            assert result == {"choices": [{"text": "hello"}]}
            stats = cache.get_stats()
            assert stats["hits"] == 1
            assert stats["hit_rate"] == 0.5  # 1 hit / 2 total
        finally:
            if old is None:
                os.environ.pop("YUNSHU_RESPONSE_CACHE", None)
            else:
                os.environ["YUNSHU_RESPONSE_CACHE"] = old

    def test_cache_ttl_expiry(self):
        old = os.environ.get("YUNSHU_RESPONSE_CACHE")
        try:
            os.environ["YUNSHU_RESPONSE_CACHE"] = "1"
            cache = ResponseCache(ttl=0.05)  # 50ms TTL
            h = cache.hash_request("m", [])

            cache.put(h, "cached_data")
            assert cache.get(h) == "cached_data"

            time.sleep(0.1)  # Wait for TTL to expire
            assert cache.get(h) is None
            stats = cache.get_stats()
            assert stats["misses"] == 1  # The second get was a miss
        finally:
            if old is None:
                os.environ.pop("YUNSHU_RESPONSE_CACHE", None)
            else:
                os.environ["YUNSHU_RESPONSE_CACHE"] = old

    def test_invalidate_all(self):
        old = os.environ.get("YUNSHU_RESPONSE_CACHE")
        try:
            os.environ["YUNSHU_RESPONSE_CACHE"] = "1"
            cache = ResponseCache()
            for i in range(10):
                h = cache.hash_request("m", [{"i": i}])
                cache.put(h, f"result-{i}")

            assert cache.get_stats()["entries"] == 10
            removed = cache.invalidate()
            assert removed == 10
            assert cache.get_stats()["entries"] == 0
        finally:
            if old is None:
                os.environ.pop("YUNSHU_RESPONSE_CACHE", None)
            else:
                os.environ["YUNSHU_RESPONSE_CACHE"] = old

    def test_invalidate_by_prefix(self):
        old = os.environ.get("YUNSHU_RESPONSE_CACHE")
        try:
            os.environ["YUNSHU_RESPONSE_CACHE"] = "1"
            cache = ResponseCache()
            h1 = cache.hash_request("m", [{"i": 1}])
            h2 = cache.hash_request("m", [{"i": 2}])
            cache.put(h1, "r1")
            cache.put(h2, "r2")

            # Invalidate by first 4 chars of h1
            prefix = h1[:4]
            removed = cache.invalidate(prefix)
            assert removed >= 1  # At least h1 matches
        finally:
            if old is None:
                os.environ.pop("YUNSHU_RESPONSE_CACHE", None)
            else:
                os.environ["YUNSHU_RESPONSE_CACHE"] = old

    def test_lru_eviction_on_max_entries(self):
        old = os.environ.get("YUNSHU_RESPONSE_CACHE")
        try:
            os.environ["YUNSHU_RESPONSE_CACHE"] = "1"
            cache = ResponseCache(max_entries=3)
            hashes = []
            for i in range(5):
                h = cache.hash_request("m", [{"i": i}])
                hashes.append(h)
                cache.put(h, f"result-{i}")

            # Only 3 entries should remain
            stats = cache.get_stats()
            assert stats["entries"] <= 3
            assert stats["evictions"] >= 2
        finally:
            if old is None:
                os.environ.pop("YUNSHU_RESPONSE_CACHE", None)
            else:
                os.environ["YUNSHU_RESPONSE_CACHE"] = old

    def test_stats_reflect_operations(self):
        old = os.environ.get("YUNSHU_RESPONSE_CACHE")
        try:
            os.environ["YUNSHU_RESPONSE_CACHE"] = "1"
            cache = ResponseCache()
            h = cache.hash_request("m", [])

            cache.get(h)  # miss
            cache.put(h, "r")  # store
            cache.get(h)  # hit
            cache.invalidate()  # eviction

            stats = cache.get_stats()
            assert stats["enabled"] is True
            assert stats["hits"] == 1
            assert stats["misses"] == 1
            assert stats["stores"] == 1
            assert stats["evictions"] == 1
        finally:
            if old is None:
                os.environ.pop("YUNSHU_RESPONSE_CACHE", None)
            else:
                os.environ["YUNSHU_RESPONSE_CACHE"] = old

    def test_put_updates_existing_entry(self):
        old = os.environ.get("YUNSHU_RESPONSE_CACHE")
        try:
            os.environ["YUNSHU_RESPONSE_CACHE"] = "1"
            cache = ResponseCache()
            h = cache.hash_request("m", [])

            cache.put(h, "old_value")
            cache.put(h, "new_value")

            assert cache.get(h) == "new_value"
            stats = cache.get_stats()
            assert stats["entries"] == 1
        finally:
            if old is None:
                os.environ.pop("YUNSHU_RESPONSE_CACHE", None)
            else:
                os.environ["YUNSHU_RESPONSE_CACHE"] = old


# ── Module Singleton Tests ──────────────────────────────────────────────


class TestModuleSingletons:
    """Tests for module-level singleton accessors."""

    def test_get_request_coalescer(self):
        c = get_request_coalescer()
        assert isinstance(c, RequestCoalescer)
        # Second call returns same instance
        assert get_request_coalescer() is c

    def test_get_streaming_buffer(self):
        buf = get_streaming_buffer()
        assert isinstance(buf, StreamingResponseBuffer)

    def test_get_connection_pool(self):
        pool = get_connection_pool()
        assert isinstance(pool, GatewayConnectionPool)
        assert get_connection_pool() is pool

    def test_get_response_cache(self):
        cache = get_response_cache()
        assert isinstance(cache, ResponseCache)
        assert get_response_cache() is cache
