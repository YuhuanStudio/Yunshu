from __future__ import annotations

import pytest

from yunshu_gateway import engine as engine_mod
from yunshu_gateway.main import create_app

from .harness import FakeBatchedEngine, LeakAudit


@pytest.fixture(autouse=True)
def _isolate():
    engine_mod._engine = None
    yield
    engine_mod._engine = None


@pytest.fixture
def make_app(monkeypatch):
    """``make_app(engine)`` -> app with that engine as the single served model."""

    def _make(engine: FakeBatchedEngine):
        # Single-model mode whatever an earlier test in this worker left behind.
        monkeypatch.setattr(engine_mod, "_model_manager", None)
        monkeypatch.setattr(engine_mod, "_engine", engine)
        return create_app()

    return _make


@pytest.fixture
def audit():
    def _make(*engines):
        return LeakAudit(*engines)

    return _make


@pytest.fixture(autouse=True)
def _record_who_registers(monkeypatch):
    """A leaked tracker entry names the thread, task and stack that registered it (an entry that is not this
    test's own request came from another test's late work in the same worker)."""
    import asyncio
    import threading
    import traceback

    from yunshu_engine.request_tracker import RequestTracker

    original = RequestTracker.register

    def register(self, request_id, model="", owner=None):
        gen = original(self, request_id, model, owner)
        try:
            task = asyncio.current_task()
        except RuntimeError:
            task = None
        gen._registered_by = (
            threading.current_thread().name,
            task.get_name() if task else None,
            [
                f"{f.filename.rsplit('/', 1)[-1]}:{f.lineno}"
                for f in traceback.extract_stack()[-8:-1]
            ],
        )
        return gen

    monkeypatch.setattr(RequestTracker, "register", register)
