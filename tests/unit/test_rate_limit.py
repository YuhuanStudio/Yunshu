"""Tests for Yunshu Rate Limiting — LRU eviction, TTL expiry, and memory safety."""

import os
import time
from unittest.mock import MagicMock, patch

from yunshu_gateway.middleware.rate_limit import (
    RateLimitMiddleware,
    _LRUBucketCache,
    _TokenBucket,
)


class TestTokenBucket:
    """Test _TokenBucket TTL tracking."""

    def test_consume_updates_last_used(self):
        bucket = _TokenBucket(rate=10.0, capacity=100)
        t0 = bucket.last_used
        time.sleep(0.01)
        bucket.consume()
        assert bucket.last_used > t0

    def test_is_expired_when_stale(self):
        bucket = _TokenBucket(rate=10.0, capacity=100)
        bucket.last_used = time.monotonic() - 100
        assert bucket.is_expired(ttl=60)

    def test_not_expired_when_fresh(self):
        bucket = _TokenBucket(rate=10.0, capacity=100)
        assert not bucket.is_expired(ttl=600)

    def test_consume_within_capacity(self):
        bucket = _TokenBucket(rate=2.0, capacity=5)
        # Should be able to consume 5 tokens
        for _ in range(5):
            assert bucket.consume()
        # 6th should fail
        assert not bucket.consume()


class TestLRUBucketCache:
    """Test _LRUBucketCache LRU eviction and TTL expiry."""

    def test_get_or_create_returns_bucket(self):
        cache = _LRUBucketCache(rate=2.0, capacity=10, max_buckets=5, ttl=60)
        bucket = cache.get_or_create("1.2.3.4")
        assert isinstance(bucket, _TokenBucket)
        assert bucket.capacity == 10

    def test_same_key_returns_same_bucket(self):
        cache = _LRUBucketCache(rate=2.0, capacity=10, max_buckets=5, ttl=60)
        b1 = cache.get_or_create("1.2.3.4")
        b2 = cache.get_or_create("1.2.3.4")
        assert b1 is b2

    def test_lru_eviction_at_max(self):
        """When max_buckets reached, LRU entries are evicted."""
        cache = _LRUBucketCache(rate=2.0, capacity=10, max_buckets=3, ttl=600)

        # Fill to max
        cache.get_or_create("ip1")
        cache.get_or_create("ip2")
        cache.get_or_create("ip3")
        assert len(cache) == 3

        # Adding 4th should evict LRU (ip1)
        cache.get_or_create("ip4")
        assert len(cache) == 3
        assert "ip1" not in cache
        assert "ip4" in cache

    def test_access_refreshes_lru(self):
        """Accessing a key moves it to end, preventing eviction."""
        cache = _LRUBucketCache(rate=2.0, capacity=10, max_buckets=3, ttl=600)

        cache.get_or_create("ip1")
        cache.get_or_create("ip2")
        cache.get_or_create("ip3")

        # Touch ip1 to refresh it
        cache.get_or_create("ip1")

        # Adding ip4 should now evict ip2 (LRU), not ip1
        cache.get_or_create("ip4")
        assert len(cache) == 3
        assert "ip1" in cache
        assert "ip2" not in cache
        assert "ip4" in cache

    def test_ttl_expiry_during_eviction(self):
        """Expired buckets are evicted first when at capacity."""
        cache = _LRUBucketCache(rate=2.0, capacity=10, max_buckets=3, ttl=0.05)

        cache.get_or_create("expired1")
        cache.get_or_create("expired2")

        # Wait for TTL to expire on first two
        time.sleep(0.06)

        # Create 'active' after expiry — it's fresh
        cache.get_or_create("active")
        # Touch it to ensure it's most recently used
        cache.get_or_create("active")

        # Adding new entry should evict expired first, keep 'active'
        cache.get_or_create("new")
        assert "active" in cache
        assert "new" in cache

    def test_cleanup_expired(self):
        """cleanup_expired removes all stale buckets."""
        cache = _LRUBucketCache(rate=2.0, capacity=10, max_buckets=100, ttl=0.05)

        # Create buckets
        for i in range(10):
            cache.get_or_create(f"ip{i}")
        assert len(cache) == 10

        # Wait for expiry
        time.sleep(0.06)

        removed = cache.cleanup_expired()
        assert removed == 10
        assert len(cache) == 0

    def test_cleanup_expired_preserves_active(self):
        """cleanup_expired keeps non-expired buckets."""
        cache = _LRUBucketCache(rate=2.0, capacity=10, max_buckets=100, ttl=0.1)

        # Create old buckets
        for i in range(5):
            cache.get_or_create(f"old_{i}")
        assert len(cache) == 5

        # Wait for them to expire
        time.sleep(0.12)

        # Create fresh buckets
        for i in range(5):
            cache.get_or_create(f"fresh_{i}")

        removed = cache.cleanup_expired()
        assert removed == 5
        assert len(cache) == 5
        for i in range(5):
            assert f"fresh_{i}" in cache

    def test_max_buckets_zero_eviction(self):
        """Very small max_buckets still works."""
        cache = _LRUBucketCache(rate=2.0, capacity=10, max_buckets=1, ttl=600)
        cache.get_or_create("ip1")
        assert len(cache) == 1
        cache.get_or_create("ip2")
        assert len(cache) == 1
        assert "ip1" not in cache
        assert "ip2" in cache


class TestRateLimitMiddlewareInit:
    """Test middleware configuration from environment."""

    def test_default_config(self):
        app = MagicMock()
        mw = RateLimitMiddleware(app)
        assert mw._rpm == 120
        assert mw._bucket_cache._max_buckets == 10000
        assert mw._bucket_cache._ttl == 600.0

    def test_custom_rpm(self):
        app = MagicMock()
        mw = RateLimitMiddleware(app, rpm=60)
        assert mw._rpm == 60

    def test_env_override_max_buckets(self):
        app = MagicMock()
        with patch.dict(os.environ, {"YUNSHU_RATE_LIMIT_MAX_BUCKETS": "500"}):
            mw = RateLimitMiddleware(app)
        assert mw._bucket_cache._max_buckets == 500

    def test_env_override_ttl(self):
        app = MagicMock()
        with patch.dict(os.environ, {"YUNSHU_RATE_LIMIT_TTL_SECONDS": "300"}):
            mw = RateLimitMiddleware(app)
        assert mw._bucket_cache._ttl == 300.0


class TestRateLimitMiddlewareNoGrowth:
    """Test that the middleware does not grow unboundedly under unique-IP traffic."""

    def test_unique_ips_capped(self):
        """Simulate many unique IPs and verify bucket count stays bounded."""
        app = MagicMock()
        mw = RateLimitMiddleware(app, rpm=120)
        # Set small limit for test
        mw._bucket_cache._max_buckets = 100

        # Simulate 1000 unique IPs
        for i in range(1000):
            bucket = mw._bucket_cache.get_or_create(f"10.0.{i // 256}.{i % 256}")
            bucket.consume()

        # Should not exceed max_buckets
        assert len(mw._bucket_cache) <= 100


# ── refund() coverage ──


class TestTokenBucketRefund:
    """validate the _TokenBucket.refund() method added in that closes the leaked-key DoS vector (refund on 401/403 response)."""

    def test_refund_increments_tokens(self):
        b = _TokenBucket(rate=1.0, capacity=10)
        assert b.tokens == 10.0
        assert b.consume() is True
        # After consume, tokens is approximately 9 (minus rate*tiny_elapsed)
        b.refund()
        # Now should be capped at 10 (capacity)
        assert b.tokens <= 10.0
        assert b.tokens > 9.0

    def test_refund_caps_at_capacity(self):
        """Refunding when bucket is already full should NOT exceed capacity."""
        b = _TokenBucket(rate=1.0, capacity=5)
        # Bucket starts full
        b.refund()
        b.refund()
        b.refund()
        assert b.tokens <= 5.0

    def test_refund_multiple_tokens(self):
        b = _TokenBucket(rate=1.0, capacity=10)
        b.consume(3)
        # tokens ≈ 7
        b.refund(3)
        # back near 10
        assert b.tokens >= 9.5

    def test_consume_then_refund_then_consume(self):
        """Verify the DoS-guard semantics: consume, refund on auth-fail,
        then a legitimate next consume still succeeds."""
        b = _TokenBucket(rate=0.1, capacity=2)  # 2 tokens, slow regen
        # Burn both tokens
        assert b.consume()
        assert b.consume()
        # Bucket is empty — next consume would fail
        assert not b.consume()
        # Simulate the middleware refund on 401
        b.refund()
        b.refund()
        # Now legitimate consumes work again
        assert b.consume()
        assert b.consume()
