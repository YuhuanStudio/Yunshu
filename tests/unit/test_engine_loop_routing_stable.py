"""Fast path vs EngineCore is configuration, not load (YUNSHU_ENGINE_LOOP)."""

from types import SimpleNamespace

from yunshu_engine.batched_engine import BatchedEngine


def _engine(loop_default=False, fast_inflight=0, core_busy=True):
    eng = object.__new__(BatchedEngine)
    eng._engine_loop_default = loop_default
    eng._active_fast_path_count = fast_inflight
    eng._engine_core = SimpleNamespace(has_active_requests=core_busy)
    return eng


def test_default_is_fast_path_even_when_core_is_busy():
    assert _engine(core_busy=True)._should_use_engine_loop(None) is False


def test_engine_loop_mode_is_config():
    assert _engine(loop_default=True)._should_use_engine_loop(None) is True
    assert _engine()._should_use_engine_loop(True) is True
    assert _engine(loop_default=True)._should_use_engine_loop(False) is False


def test_never_enters_loop_while_fast_path_runs():
    eng = _engine(loop_default=True, fast_inflight=1)
    assert eng._should_use_engine_loop(None) is False
    assert eng._should_use_engine_loop(True) is False
