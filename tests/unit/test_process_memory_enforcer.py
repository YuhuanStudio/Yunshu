"""Tests for ProcessMemoryEnforcer (model-free, mlx memory mocked).

Covers:
- start()/stop() lifecycle and idempotency
- max_bytes getter/setter
- is_running / get_status
- _check_and_enforce no-op when under limit
- _check_and_enforce over-limit branch with no models to evict
"""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from yunshu_engine.process_memory_enforcer import ProcessMemoryEnforcer


def _make_manager():
    """A minimal fake ModelManager with the attributes the enforcer touches."""
    mgr = MagicMock()
    mgr._entries = {}
    mgr._find_lru_victim = MagicMock(return_value=None)

    async def _check_ttl():
        return None

    mgr.check_ttl = MagicMock(side_effect=_check_ttl)
    return mgr


class TestLifecycle:
    @pytest.mark.asyncio
    async def test_start_sets_running(self):
        enf = ProcessMemoryEnforcer(
            _make_manager(), max_bytes=10 * 1024**3, poll_interval=10.0
        )
        assert enf.is_running is False
        enf.start()
        try:
            assert enf.is_running is True
            assert enf._task is not None
        finally:
            await enf.stop()

    @pytest.mark.asyncio
    async def test_start_idempotent(self):
        enf = ProcessMemoryEnforcer(
            _make_manager(), max_bytes=10 * 1024**3, poll_interval=10.0
        )
        enf.start()
        first_task = enf._task
        enf.start()  # second call should be a no-op
        try:
            assert enf._task is first_task
        finally:
            await enf.stop()

    @pytest.mark.asyncio
    async def test_stop_clears_task(self):
        enf = ProcessMemoryEnforcer(
            _make_manager(), max_bytes=10 * 1024**3, poll_interval=10.0
        )
        enf.start()
        await enf.stop()
        assert enf.is_running is False
        assert enf._task is None

    @pytest.mark.asyncio
    async def test_stop_when_not_started(self):
        enf = ProcessMemoryEnforcer(_make_manager(), max_bytes=10 * 1024**3)
        await enf.stop()  # should not raise
        assert enf.is_running is False


class TestMaxBytes:
    def test_getter(self):
        enf = ProcessMemoryEnforcer(_make_manager(), max_bytes=42)
        assert enf.max_bytes == 42

    def test_setter(self):
        enf = ProcessMemoryEnforcer(_make_manager(), max_bytes=42)
        enf.max_bytes = 100
        assert enf.max_bytes == 100


class TestGetStatus:
    def test_status_not_running(self):
        enf = ProcessMemoryEnforcer(_make_manager(), max_bytes=8 * 1024**3)
        status = enf.get_status()
        assert status["enabled"] is False
        assert status["current_bytes"] == 0
        assert status["max_bytes"] == 8 * 1024**3
        assert status["utilization"] == 0.0

    def test_status_zero_limit_no_div_zero(self):
        enf = ProcessMemoryEnforcer(_make_manager(), max_bytes=0)
        status = enf.get_status()
        assert status["utilization"] == 0.0


class TestCheckAndEnforce:
    @pytest.mark.asyncio
    async def test_noop_when_max_bytes_zero(self):
        enf = ProcessMemoryEnforcer(_make_manager(), max_bytes=0)
        # max_bytes <= 0 returns immediately; no mlx call needed
        await enf._check_and_enforce()

    @pytest.mark.asyncio
    async def test_noop_when_under_limit(self):
        mgr = _make_manager()
        enf = ProcessMemoryEnforcer(mgr, max_bytes=100 * 1024**3)
        with patch("yunshu_engine.process_memory_enforcer.mx") as mx_mock:
            mx_mock.get_active_memory.return_value = 1 * 1024**3  # well under
            await enf._check_and_enforce()
        # No eviction attempted
        mgr._find_lru_victim.assert_not_called()

    @pytest.mark.asyncio
    async def test_over_limit_no_victim(self):
        """Huge RSS over limit, but no evictable model -> graceful no-op."""
        mgr = _make_manager()
        mgr._lock = asyncio.Lock()
        mgr._find_lru_victim = MagicMock(return_value=None)
        mgr._get_mlx_executor = MagicMock(return_value=None)
        enf = ProcessMemoryEnforcer(mgr, max_bytes=1 * 1024**3)
        with patch("yunshu_engine.process_memory_enforcer.mx") as mx_mock:
            mx_mock.get_active_memory.return_value = 50 * 1024**3  # way over
            await enf._check_and_enforce()
        # Tried to find a victim, found none, exited loop without unloading
        mgr._find_lru_victim.assert_called()
        mgr.unload_model.assert_not_called()
