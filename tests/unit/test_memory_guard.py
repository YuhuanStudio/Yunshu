"""Tests for MemoryGuard — proactive memory admission control.

Tests cover:
- preflight_check() accepts small requests
- preflight_check() rejects huge requests that would OOM
- preflight_check() with zero available memory
- get_recommended_max_tokens() returns sensible limits
- generation_guard() respects concurrent request limit
- get_stats() tracks rejections
- MemoryGuard integration with mocked MemoryMonitor
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from yunshu_engine.memory_guard import MemoryGuard
from yunshu_engine.memory_monitor import MemoryInfo, MemoryMonitor

# ── Helpers ──


def _make_monitor(
    available_bytes: int = 4 * 1024 * 1024 * 1024,  # 4 GB
    total_bytes: int = 8 * 1024 * 1024 * 1024,  # 8 GB
    active_bytes: int = 4 * 1024 * 1024 * 1024,
    num_layers: int = 24,
    num_kv_heads: int = 8,
    head_dim: int = 128,
    num_attention_heads: int = 32,
) -> MemoryMonitor:
    """Create a MemoryMonitor with real model info and mocked MLX calls."""
    with patch('python.yunshu_engine.memory_monitor.HAS_MLX_METAL', False):
        monitor = MemoryMonitor()
    monitor.set_model_info(
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        num_attention_heads=num_attention_heads,
    )
    # Mock get_memory_info to return controlled values
    info = MemoryInfo(
        total_bytes=total_bytes,
        active_bytes=active_bytes,
        peak_bytes=active_bytes,
        cache_bytes=0,
        available_bytes=available_bytes,
        utilization_pct=(active_bytes / total_bytes * 100) if total_bytes > 0 else 0.0,
    )
    monitor.get_memory_info = MagicMock(return_value=info)  # type: ignore
    return monitor


def _make_guard(
    available_bytes: int = 4 * 1024 * 1024 * 1024,
    max_concurrent: int = 64,
    safety_margin_pct: float = 0.10,
    **monitor_kwargs,
) -> MemoryGuard:
    """Create a MemoryGuard with a configured MemoryMonitor."""
    monitor = _make_monitor(available_bytes=available_bytes, **monitor_kwargs)
    return MemoryGuard(
        memory_monitor=monitor,
        max_concurrent_requests=max_concurrent,
        safety_margin_pct=safety_margin_pct,
    )


# ── Preflight Check Tests ──


class TestPreflightCheck:
    """Tests for MemoryGuard.preflight_check()."""

    def test_accepts_small_requests(self):
        """Small requests should pass preflight check."""
        guard = _make_guard(available_bytes=4 * 1024 ** 3)
        ok, reason = guard.preflight_check(
            num_prompt_tokens=64,
            max_tokens=128,
        )
        assert ok is True
        assert reason == ""

    def test_accepts_medium_requests(self):
        """Reasonable requests should pass with 4 GB available."""
        guard = _make_guard(available_bytes=4 * 1024 ** 3)
        ok, reason = guard.preflight_check(
            num_prompt_tokens=1024,
            max_tokens=512,
        )
        assert ok is True
        assert reason == ""

    def test_rejects_huge_requests(self):
        """Requests that would exceed available memory should be rejected."""
        guard = _make_guard(available_bytes=1 * 1024 ** 2)  # 1 MB available
        ok, reason = guard.preflight_check(
            num_prompt_tokens=100_000,
            max_tokens=10_000,
        )
        assert ok is False
        assert "Insufficient memory" in reason

    def test_rejects_with_zero_available_memory(self):
        """Zero available memory should reject all requests."""
        guard = _make_guard(available_bytes=0)
        ok, reason = guard.preflight_check(
            num_prompt_tokens=10,
            max_tokens=10,
        )
        assert ok is False
        assert "No usable memory" in reason

    def test_rejects_with_tiny_available_memory(self):
        """Very small available memory (less than safety margin) should reject."""
        guard = _make_guard(available_bytes=100, safety_margin_pct=0.50)
        ok, reason = guard.preflight_check(
            num_prompt_tokens=10,
            max_tokens=10,
        )
        assert ok is False

    def test_returns_tuple(self):
        """preflight_check always returns a (bool, str) tuple."""
        guard = _make_guard()
        result = guard.preflight_check(num_prompt_tokens=100, max_tokens=100)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bool)
        assert isinstance(result[1], str)

    def test_increments_total_checks(self):
        """Each call should increment total_checks counter."""
        guard = _make_guard()
        assert guard.get_stats()["total_checks"] == 0
        guard.preflight_check(num_prompt_tokens=100, max_tokens=100)
        assert guard.get_stats()["total_checks"] == 1
        guard.preflight_check(num_prompt_tokens=100, max_tokens=100)
        assert guard.get_stats()["total_checks"] == 2

    def test_increments_rejections_on_failure(self):
        """Failed checks should increment rejection counters."""
        guard = _make_guard(available_bytes=0)
        guard.preflight_check(num_prompt_tokens=100, max_tokens=100)
        stats = guard.get_stats()
        assert stats["preflight_rejections"] == 1
        assert stats["total_rejections"] == 1

    def test_does_not_increment_rejections_on_success(self):
        """Successful checks should not increment rejection counters."""
        guard = _make_guard(available_bytes=4 * 1024 ** 3)
        guard.preflight_check(num_prompt_tokens=100, max_tokens=100)
        stats = guard.get_stats()
        assert stats["preflight_rejections"] == 0
        assert stats["total_rejections"] == 0


# ── Generation Guard Tests ──


class TestGenerationGuard:
    """Tests for MemoryGuard.generation_guard()."""

    def test_allows_under_limit(self):
        """Should allow when active requests < max_concurrent."""
        guard = _make_guard(max_concurrent=64)
        assert guard.generation_guard(num_active_requests=10) is True

    def test_rejects_at_limit(self):
        """Should reject when active requests >= max_concurrent."""
        guard = _make_guard(max_concurrent=4)
        assert guard.generation_guard(num_active_requests=4) is False

    def test_rejects_over_limit(self):
        """Should reject when active requests > max_concurrent."""
        guard = _make_guard(max_concurrent=4)
        assert guard.generation_guard(num_active_requests=10) is False

    def test_rejects_under_memory_pressure(self):
        """Should reject when memory is under pressure (>90%)."""
        monitor = _make_monitor(available_bytes=100)
        # Simulate 95% utilization
        info = MemoryInfo(
            total_bytes=8 * 1024 ** 3,
            active_bytes=int(7.6 * 1024 ** 3),
            peak_bytes=int(7.6 * 1024 ** 3),
            cache_bytes=0,
            available_bytes=100,
            utilization_pct=95.0,
        )
        monitor.get_memory_info = MagicMock(return_value=info)  # type: ignore
        guard = MemoryGuard(memory_monitor=monitor, max_concurrent_requests=64)
        assert guard.generation_guard(num_active_requests=1) is False

    def test_increments_concurrent_rejections(self):
        """Failed generation guard checks should track concurrent_rejections."""
        guard = _make_guard(max_concurrent=2)
        guard.generation_guard(num_active_requests=5)
        stats = guard.get_stats()
        assert stats["concurrent_rejections"] == 1

    def test_zero_active_requests_always_allowed(self):
        """Zero active requests should be allowed (unless memory pressure)."""
        guard = _make_guard(max_concurrent=1)
        assert guard.generation_guard(num_active_requests=0) is True


# ── Recommended Max Tokens Tests ──


class TestRecommendedMaxTokens:
    """Tests for MemoryGuard.get_recommended_max_tokens()."""

    def test_returns_positive_for_available_memory(self):
        """Should return a positive number when memory is available."""
        guard = _make_guard(available_bytes=4 * 1024 ** 3)
        result = guard.get_recommended_max_tokens(num_prompt_tokens=512)
        assert result > 0

    def test_returns_zero_for_no_memory(self):
        """Should return 0 when no memory is available."""
        guard = _make_guard(available_bytes=0)
        result = guard.get_recommended_max_tokens(num_prompt_tokens=512)
        assert result == 0

    def test_returns_sensible_limit(self):
        """Recommended max tokens should be reasonable (not absurdly large)."""
        guard = _make_guard(available_bytes=2 * 1024 ** 3)
        result = guard.get_recommended_max_tokens(num_prompt_tokens=512)
        assert 0 < result <= 32768  # capped at 32K

    def test_longer_prompt_fewer_max_tokens(self):
        """Longer prompts should allow fewer decode tokens."""
        guard = _make_guard(available_bytes=2 * 1024 ** 3)
        short = guard.get_recommended_max_tokens(num_prompt_tokens=128)
        long = guard.get_recommended_max_tokens(num_prompt_tokens=8192)
        assert short >= long

    def test_returns_default_without_model_info(self):
        """Without model info (per-token cost = 0), should return default 256."""
        with patch('python.yunshu_engine.memory_monitor.HAS_MLX_METAL', False):
            monitor = MemoryMonitor()
        # No set_model_info called — estimate_prompt_kv_bytes returns 0
        info = MemoryInfo(
            total_bytes=8 * 1024 ** 3,
            active_bytes=4 * 1024 ** 3,
            peak_bytes=4 * 1024 ** 3,
            cache_bytes=0,
            available_bytes=4 * 1024 ** 3,
            utilization_pct=50.0,
        )
        monitor.get_memory_info = MagicMock(return_value=info)  # type: ignore
        guard = MemoryGuard(memory_monitor=monitor)
        result = guard.get_recommended_max_tokens(num_prompt_tokens=100)
        assert result == 256

    def test_returns_zero_for_prompt_too_large(self):
        """If prompt alone exceeds available memory, max_tokens should be 0."""
        guard = _make_guard(available_bytes=1024)  # 1 KB available
        result = guard.get_recommended_max_tokens(num_prompt_tokens=100_000)
        assert result == 0


# ── Stats Tests ──


class TestGetStats:
    """Tests for MemoryGuard.get_stats()."""

    def test_initial_stats(self):
        """Fresh guard should have zero stats."""
        guard = _make_guard()
        stats = guard.get_stats()
        assert stats["total_checks"] == 0
        assert stats["total_rejections"] == 0
        assert stats["preflight_rejections"] == 0
        assert stats["concurrent_rejections"] == 0
        assert stats["rejection_rate"] == 0.0
        assert stats["max_concurrent_requests"] == 64

    def test_rejection_rate_calculation(self):
        """Rejection rate should be calculated correctly."""
        guard = _make_guard(available_bytes=0)
        guard.preflight_check(num_prompt_tokens=100, max_tokens=100)
        guard.preflight_check(num_prompt_tokens=100, max_tokens=100)
        stats = guard.get_stats()
        assert stats["total_checks"] == 2
        assert stats["total_rejections"] == 2
        assert stats["rejection_rate"] == 100.0

    def test_mixed_rejection_rate(self):
        """Rejection rate with mixed pass/fail should be correct."""
        guard = _make_guard(available_bytes=4 * 1024 ** 3)
        # One pass
        guard.preflight_check(num_prompt_tokens=10, max_tokens=10)
        # Switch to no memory for rejection
        info = MemoryInfo(
            total_bytes=8 * 1024 ** 3,
            active_bytes=8 * 1024 ** 3,
            peak_bytes=8 * 1024 ** 3,
            cache_bytes=0,
            available_bytes=0,
            utilization_pct=100.0,
        )
        guard._monitor.get_memory_info = MagicMock(return_value=info)  # type: ignore
        guard.preflight_check(num_prompt_tokens=10, max_tokens=10)

        stats = guard.get_stats()
        assert stats["total_checks"] == 2
        assert stats["total_rejections"] == 1
        assert stats["rejection_rate"] == 50.0

    def test_includes_memory_stats(self):
        """Stats should include memory monitor stats."""
        guard = _make_guard()
        stats = guard.get_stats()
        assert "memory" in stats
        assert "available_bytes" in stats["memory"]

    def test_tracks_guard_config(self):
        """Stats should include guard configuration."""
        guard = _make_guard(max_concurrent=32, safety_margin_pct=0.20)
        stats = guard.get_stats()
        assert stats["max_concurrent_requests"] == 32
        assert stats["safety_margin_pct"] == 0.20


# ── Integration Tests (Mocked MemoryMonitor) ──


class TestMemoryGuardIntegration:
    """Integration tests with mocked MemoryMonitor."""

    def test_preflight_with_custom_model_arch(self):
        """Test preflight with different model architectures."""
        # Small model: 12 layers, 4 KV heads, 64 head_dim
        guard = _make_guard(
            available_bytes=1 * 1024 ** 3,  # 1 GB
            num_layers=12,
            num_kv_heads=4,
            head_dim=64,
            num_attention_heads=16,
        )
        ok, _ = guard.preflight_check(
            num_prompt_tokens=2048,
            max_tokens=1024,
        )
        # Small model should fit easily in 1 GB
        assert ok is True

    def test_preflight_with_large_model_limited_memory(self):
        """Large model with tight memory should reject moderate requests."""
        # Large model: 64 layers, 8 KV heads, 128 head_dim
        guard = _make_guard(
            available_bytes=10 * 1024 ** 2,  # 10 MB — very tight
            num_layers=64,
            num_kv_heads=8,
            head_dim=128,
            num_attention_heads=64,
        )
        ok, reason = guard.preflight_check(
            num_prompt_tokens=4096,
            max_tokens=2048,
        )
        assert ok is False

    def test_guard_prevents_oom_scenario(self):
        """Simulate a scenario where guard prevents OOM.

        Start with plenty of memory, then simulate memory filling up.
        """
        guard = _make_guard(available_bytes=4 * 1024 ** 3)

        # First request should pass
        ok1, _ = guard.preflight_check(num_prompt_tokens=100, max_tokens=100)
        assert ok1 is True

        # Simulate memory filling up
        low_memory_info = MemoryInfo(
            total_bytes=8 * 1024 ** 3,
            active_bytes=int(7.99 * 1024 ** 3),
            peak_bytes=8 * 1024 ** 3,
            cache_bytes=0,
            available_bytes=10 * 1024 ** 2,  # 10 MB left
            utilization_pct=99.9,
        )
        guard._monitor.get_memory_info = MagicMock(return_value=low_memory_info)  # type: ignore

        # Same request should now fail
        ok2, reason2 = guard.preflight_check(num_prompt_tokens=100, max_tokens=100)
        assert ok2 is False

    def test_generation_guard_with_prefill_tracking(self):
        """Test that generation_guard and preflight_check counters are independent."""
        guard = _make_guard(max_concurrent=2)

        # Preflight pass
        guard.preflight_check(num_prompt_tokens=10, max_tokens=10)
        # Generation guard rejection
        guard.generation_guard(num_active_requests=5)

        stats = guard.get_stats()
        assert stats["preflight_rejections"] == 0
        assert stats["concurrent_rejections"] == 1
        assert stats["total_rejections"] == 1
        assert stats["total_checks"] == 2

    def test_safety_margin_reduces_usable_memory(self):
        """Higher safety margin should reduce usable memory and cause rejections."""
        # With 10% margin, 100 MB should be enough for a tiny request
        guard_low_margin = _make_guard(
            available_bytes=100 * 1024 ** 2,  # 100 MB
            safety_margin_pct=0.10,
        )

        # With 99% margin, almost nothing is usable
        guard_high_margin = _make_guard(
            available_bytes=100 * 1024 ** 2,  # 100 MB
            safety_margin_pct=0.99,
        )

        # Tiny request — might pass with low margin but fail with high margin
        # (depends on per-token cost, but the high margin leaves only 1 MB usable)
        ok_low, _ = guard_low_margin.preflight_check(num_prompt_tokens=10, max_tokens=10)
        ok_high, _ = guard_high_margin.preflight_check(num_prompt_tokens=10, max_tokens=10)

        # Low margin should pass (90 MB usable for a 10-token request)
        assert ok_low is True
        # High margin has only 1 MB usable — likely rejects
        assert ok_high is False
