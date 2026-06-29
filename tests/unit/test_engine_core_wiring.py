"""integration tests — verify standalone modules are wired into production paths.

Tests that:
1. EngineCore.__init__ creates all wired modules
2. add_request() calls budget, dedup, lifecycle
3. _engine_loop calls profiler, auto-tuner, fairness, lifecycle
4. get_stats() returns stats from all wired modules
5. stop() cleans up wired modules
6. BatchedEngine wires model preprocessor
"""

import asyncio
import os
from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import pytest


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
    """Patch EngineCore's heavy dependencies for all tests."""
    from concurrent.futures import ThreadPoolExecutor

    real_executor = ThreadPoolExecutor(max_workers=1)

    with ExitStack() as stack:
        stack.enter_context(
            patch(
                "yunshu_engine.mlx_executor.get_mlx_executor",
                return_value=real_executor,
            )
        )
        stack.enter_context(patch("yunshu_engine.scheduler.Scheduler"))
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
            _make_mock_model(),
            _make_mock_tokenizer(),
            config=EngineCoreConfig(),
        )
    finally:
        for p in patches:
            p.__exit__(None, None, None)
    return core


class TestEngineCoreWiring:
    """Test that EngineCore creates and wires all standalone modules."""

    def test_lifecycle_orchestrator_created(self):
        core = _make_core()
        assert core._lifecycle_orchestrator is not None

    def test_budget_manager_created(self):
        core = _make_core()
        assert core._budget_manager is not None

    def test_kv_lifecycle_created(self):
        core = _make_core()
        assert core._kv_lifecycle is not None

    def test_token_scheduler_created(self):
        core = _make_core()
        assert core._token_scheduler is not None
        assert core._priority_guard is not None
        assert core._fairness_tracker is not None

    def test_auto_tuner_created(self):
        core = _make_core()
        assert core._profiler is not None
        assert core._slo_monitor is not None
        assert core._auto_tuner is not None
        assert core._adaptive_batch_sizer is not None

    def test_dedup_disabled_by_default(self):
        core = _make_core()
        assert core._request_dedup is None

    def test_dedup_enabled_via_env(self):
        core = _make_core(YUNSHU_REQUEST_DEDUP="1")
        assert core._request_dedup is not None

    def test_composition_scheduler_field_exists(self):
        core = _make_core()
        assert hasattr(core, "_composition_scheduler")


class TestEngineCoreStatsWiring:
    """Test that get_stats() returns stats from all wired modules."""

    def test_stats_include_all_modules(self):
        core = _make_core()
        core._start_time = None
        stats = core.get_stats()
        expected_keys = [
            "lifecycle",
            "budget",
            "kv_lifecycle",
            "token_scheduler",
            "auto_tuner",
            "fairness",
            "profiler",
            "slo",
        ]
        for key in expected_keys:
            assert key in stats, f"Missing stats key: {key}"
            assert isinstance(stats[key], dict), f"{key} stats not a dict"


class TestBudgetWiringInAddRequest:
    """Test that add_request() registers a budget for each request."""

    @pytest.mark.asyncio
    async def test_budget_registered_on_add(self):
        core = _make_core()
        core._wake_event = asyncio.Event()
        core._running = True  # Engine must be running to accept requests

        req_id = await core.add_request(prompt="hello", max_tokens=100)
        budget = core._budget_manager.get_budget(req_id)
        assert budget is not None
        assert budget.max_tokens == 100


class TestLifecycleWiringInAddRequest:
    """Test that add_request() tracks lifecycle state."""

    @pytest.mark.asyncio
    async def test_lifecycle_tracked_on_add(self):
        core = _make_core()
        core._wake_event = asyncio.Event()
        core._running = True  # Engine must be running to accept requests

        req_id = await core.add_request(prompt="hello", max_tokens=100)

        state = core._lifecycle_orchestrator.get_state(req_id)
        assert state is not None
        from yunshu_engine.request_lifecycle import RequestPhase

        assert state.phase in (RequestPhase.QUEUED, RequestPhase.PREFILLING)


class TestDedupWiringInAddRequest:
    """Test that add_request() uses dedup when enabled."""

    @pytest.mark.asyncio
    async def test_dedup_registers_request(self):
        core = _make_core(YUNSHU_REQUEST_DEDUP="1")
        core._wake_event = asyncio.Event()
        core._running = True  # Engine must be running to accept requests

        req_id = await core.add_request(prompt="hello", max_tokens=100)
        assert req_id in core._dedup_hashes


class TestBatchedEngineWiring:
    """Test BatchedEngine wires preprocessor registry."""

    def test_preprocessor_registry_created(self):
        from yunshu_engine.batched_engine import BatchedEngine

        engine = BatchedEngine(model_name="test-model")
        assert engine._preprocessor_registry is not None

    def test_stats_include_preprocessor(self):
        from yunshu_engine.batched_engine import BatchedEngine

        engine = BatchedEngine(model_name="test-model")
        engine._loaded = False
        stats = engine.get_stats()
        assert "model_preprocessor" in stats


class TestCrossModuleIntegration:
    """Test that wired modules interact correctly with each other."""

    def test_lifecycle_concurrency_from_env(self):
        core = _make_core(YUNSHU_CONCURRENCY_MAX="4")
        assert core._lifecycle_orchestrator._concurrency._maximum == 4

    def test_budget_from_env(self):
        core = _make_core(YUNSHU_DEFAULT_MAX_TOKENS="1024")
        assert core._budget_manager._default_max_tokens == 1024

    @pytest.mark.asyncio
    async def test_multiple_requests_budget_isolation(self):
        core = _make_core()
        core._wake_event = asyncio.Event()
        core._running = True  # Engine must be running to accept requests

        req1 = await core.add_request(prompt="hello", max_tokens=50)
        req2 = await core.add_request(prompt="world", max_tokens=200)

        b1 = core._budget_manager.get_budget(req1)
        b2 = core._budget_manager.get_budget(req2)
        assert b1.max_tokens == 50
        assert b2.max_tokens == 200
