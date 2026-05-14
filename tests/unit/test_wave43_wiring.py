"""Wave 43 integration tests — verify additional modules wired into production paths.

Tests:
1. EngineCore wires forward_batch, memory_aware_scheduler, context_window,
   kv_prefix_compression, batch_sampler
2. MeshManager wires RTT-aware routing
3. BatchedEngine wires context window truncation
4. Stats chain complete for all new modules
"""

import pytest
from unittest.mock import MagicMock, patch
from contextlib import ExitStack
import asyncio
import os


def _make_mock_model():
    model = MagicMock()
    model.parameters.return_value = []
    model.layers = []
    return model


def _make_mock_tokenizer():
    tok = MagicMock()
    tok.encode.return_value = [1, 2, 3, 4, 5]
    tok.eos_token_id = 2
    tok.eos_token_ids = [2]
    return tok


@pytest.fixture(autouse=True)
def _patch_engine():
    from concurrent.futures import ThreadPoolExecutor
    real_executor = ThreadPoolExecutor(max_workers=1)
    with ExitStack() as stack:
        stack.enter_context(
            patch('yunshu_engine.mlx_executor.get_mlx_executor', return_value=real_executor)
        )
        stack.enter_context(patch('yunshu_engine.scheduler.Scheduler'))
        yield


def _make_core(**env_overrides):
    from yunshu_engine.engine_core import EngineCore, EngineCoreConfig
    patches = []
    for k, v in env_overrides.items():
        p = patch.dict(os.environ, {k: v})
        p.__enter__()
        patches.append(p)
    try:
        core = EngineCore(
            _make_mock_model(), _make_mock_tokenizer(),
            config=EngineCoreConfig(),
        )
    finally:
        for p in patches:
            p.__exit__(None, None, None)
    return core


class TestEngineCoreWave43Wiring:
    """Test Wave 43 modules wired into EngineCore."""

    def test_batch_composer_created(self):
        core = _make_core()
        assert core._batch_composer is not None

    def test_memory_aware_scheduler_created(self):
        core = _make_core()
        assert core._memory_aware_scheduler is not None

    def test_context_window_mgr_created(self):
        core = _make_core()
        assert core._context_window_mgr is not None

    def test_kv_compressor_created(self):
        core = _make_core()
        assert core._kv_compressor is not None

    def test_batch_sampler_created(self):
        core = _make_core()
        assert core._batch_sampler is not None

    def test_batch_stop_checker_created(self):
        core = _make_core()
        assert core._batch_stop_checker is not None

    def test_sliding_window_mgr_none_by_default(self):
        core = _make_core()
        assert core._sliding_window_mgr is None


class TestEngineCoreWave43Stats:
    """Test Wave 43 stats in get_stats()."""

    def test_stats_include_wave43_modules(self):
        core = _make_core()
        core._start_time = None
        stats = core.get_stats()
        expected_keys = [
            "forward_batch", "memory_aware_scheduler", "context_window",
            "kv_prefix_compression", "batch_sampler",
        ]
        for key in expected_keys:
            assert key in stats, f"Missing stats key: {key}"
            assert isinstance(stats[key], dict), f"{key} stats not a dict"


class TestMeshManagerRttRouting:
    """Test MeshManager wires RTT-aware routing."""

    def test_rtt_router_created(self):
        from yunshu_mesh.manager import MeshManager
        mgr = MeshManager()
        assert mgr._rtt_router is not None

    def test_rtt_router_in_stats(self):
        from yunshu_mesh.manager import MeshManager
        mgr = MeshManager()
        stats = mgr.get_stats()
        assert "rtt_routing" in stats
        assert isinstance(stats["rtt_routing"], dict)


class TestBatchedEngineContextWindow:
    """Test BatchedEngine context window truncation wiring."""

    def test_preprocessor_and_context_window_available(self):
        from yunshu_engine.batched_engine import BatchedEngine
        engine = BatchedEngine(model_name="test-model")
        assert engine._preprocessor_registry is not None


class TestCrossModuleWave43:
    """Test cross-module integration for Wave 43."""

    def test_all_wired_modules_produce_stats(self):
        core = _make_core()
        core._start_time = None
        stats = core.get_stats()

        # All Wave 42 + 43 modules should produce valid stats
        all_keys = [
            # Wave 42
            "lifecycle", "budget", "kv_lifecycle", "token_scheduler",
            "auto_tuner", "fairness", "profiler", "slo",
            # Wave 43
            "forward_batch", "memory_aware_scheduler", "context_window",
            "kv_prefix_compression", "batch_sampler",
        ]
        for key in all_keys:
            assert key in stats, f"Missing: {key}"
            assert isinstance(stats[key], dict), f"{key} not dict"

    def test_batch_sampler_functional(self):
        from yunshu_engine.batch_sampler import BatchSampler
        sampler = BatchSampler()
        stats = sampler.get_stats()
        assert isinstance(stats, dict)

    def test_context_window_functional(self):
        from yunshu_engine.context_window import ContextWindowManager
        mgr = ContextWindowManager()
        result = mgr.compute_truncation(
            messages=[{"role": "user", "content": "Hello world"}],
            max_tokens=100,
            strategy="truncate_oldest",
        )
        assert result is not None
        stats = mgr.get_stats()
        assert isinstance(stats, dict)

    def test_memory_aware_scheduler_functional(self):
        from yunshu_engine.memory_aware_scheduler import MemoryAwareScheduler
        sched = MemoryAwareScheduler()
        stats = sched.get_stats()
        # SchedulerStats is a dataclass, not a dict
        assert hasattr(stats, 'total_admissions')
        assert hasattr(stats, 'is_paused')

    def test_kv_compressor_functional(self):
        from yunshu_engine.kv_prefix_compression import KVPrefixCompressor
        comp = KVPrefixCompressor()
        stats = comp.get_stats()
        assert isinstance(stats, dict)
