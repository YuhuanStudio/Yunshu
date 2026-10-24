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
