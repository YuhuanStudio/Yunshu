"""unit coverage for sleep router (previously zero coverage).

Tests:
- /sleep/status RBAC gate
- is_sleeping() / get_sleep_state() helpers
- POST /v1/sleep level boundary validation (0/1/2 allowed, 3 rejected)
- Auth-disabled bypass path
"""

from __future__ import annotations

import os

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


def _make_app_with_sleep_router():
    """Mount the sleep router on a bare FastAPI app for isolated testing."""
    from yunshu_gateway.routers.sleep import router as sleep_router

    app = FastAPI()
    # Mirror the gateway's /v1 prefix used in production
    app.include_router(sleep_router, prefix="/v1")
    return app


def _reset_sleep_state():
    """Clear module-global sleep state between tests."""
    import yunshu_gateway.routers.sleep as sleep_mod

    sleep_mod._sleeping = False
    sleep_mod._sleep_level = -1
    sleep_mod._sleep_transitioning = False
    sleep_mod._saved_model_name = None
    os.environ.pop("YUNSHU_SLEEPING", None)


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch):
    """Always start each test from a known sleep-state."""
    _reset_sleep_state()
    # Don't leak previous env between tests
    monkeypatch.delenv("YUNSHU_AUTH_DISABLED", raising=False)
    monkeypatch.delenv("YUNSHU_AUTH_TOKEN", raising=False)
    yield
    _reset_sleep_state()


class TestSleepStatus:
    """/sleep/status returns the current sleep snapshot."""

    def test_status_default_state(self, monkeypatch):
        """With auth disabled, /sleep/status should return the awake snapshot."""
        monkeypatch.setenv("YUNSHU_AUTH_DISABLED", "true")
        app = _make_app_with_sleep_router()
        client = TestClient(app)
        r = client.get("/v1/sleep/status")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["sleeping"] is False
        assert body["level"] == -1
        assert body["transitioning"] is False

    def test_status_reflects_sleeping_flag(self, monkeypatch):
        """After flipping the module-global _sleeping, status mirrors it."""
        monkeypatch.setenv("YUNSHU_AUTH_DISABLED", "true")
        import yunshu_gateway.routers.sleep as sleep_mod

        sleep_mod._sleeping = True
        sleep_mod._sleep_level = 1
        app = _make_app_with_sleep_router()
        client = TestClient(app)
        r = client.get("/v1/sleep/status")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["sleeping"] is True
        assert body["level"] == 1


class TestSleepHelpers:
    """The lightweight, monkey-patch-friendly helpers."""

    def test_is_sleeping_default_false(self):
        from yunshu_gateway.routers.sleep import is_sleeping

        assert is_sleeping() is False

    def test_is_sleeping_after_flag_set(self):
        import yunshu_gateway.routers.sleep as sleep_mod

        sleep_mod._sleeping = True
        assert sleep_mod.is_sleeping() is True

    def test_get_sleep_state_awake(self):
        from yunshu_gateway.routers.sleep import get_sleep_state

        snap = get_sleep_state()
        assert snap["sleeping"] is False
        assert snap["level"] == -1
        assert snap["transitioning"] is False

    def test_get_sleep_state_sleeping(self):
        import yunshu_gateway.routers.sleep as sleep_mod

        sleep_mod._sleeping = True
        sleep_mod._sleep_level = 2
        sleep_mod._sleep_transitioning = True
        snap = sleep_mod.get_sleep_state()
        assert snap["sleeping"] is True
        assert snap["level"] == 2
        assert snap["transitioning"] is True


class TestSleepPostBoundary:
    """Validate the pydantic level boundary added on SleepRequest."""

    def test_level_3_rejected_422(self, monkeypatch):
        """Note: level >= 3 must hit pydantic validation, not silent clamp."""
        monkeypatch.setenv("YUNSHU_AUTH_DISABLED", "true")
        app = _make_app_with_sleep_router()
        client = TestClient(app)
        r = client.post("/v1/sleep", json={"level": 3})
        assert r.status_code == 422, r.text

    def test_level_negative_rejected_422(self, monkeypatch):
        monkeypatch.setenv("YUNSHU_AUTH_DISABLED", "true")
        app = _make_app_with_sleep_router()
        client = TestClient(app)
        r = client.post("/v1/sleep", json={"level": -1})
        assert r.status_code == 422, r.text

    def test_level_0_happy_path(self, monkeypatch):
        """Level 0 (pause) should not touch engine and return sleeping snapshot."""
        monkeypatch.setenv("YUNSHU_AUTH_DISABLED", "true")
        app = _make_app_with_sleep_router()
        client = TestClient(app)
        r = client.post("/v1/sleep", json={"level": 0})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["status"] == "sleeping"
        assert body["level"] == 0


class TestSleepInflightGuardW804:
    """level>=1 sleep must refuse when the ENGINE reports active requests
    (the HTTP counter missed /v1/score, /ocr, /audio/translations, etc. — those hit
    the same default engine L1/L2 tears down). Trust engine.has_active_requests."""

    def _reset(self):
        import yunshu_gateway.routers.sleep as sleep_mod

        sleep_mod._sleeping = False
        sleep_mod._sleep_level = -1
        sleep_mod._sleep_transitioning = False
        import yunshu_gateway.main as _main

        _main._active_requests = 0

    def test_sleep_refused_when_engine_busy(self, monkeypatch):
        monkeypatch.setenv("YUNSHU_AUTH_DISABLED", "true")
        self._reset()

        class _BusyEngine:
            def has_active_requests(self):
                return True

        monkeypatch.setattr("yunshu_gateway.engine.get_engine", lambda: _BusyEngine())
        app = _make_app_with_sleep_router()
        client = TestClient(app)
        r = client.post("/v1/sleep", json={"level": 1})
        assert r.status_code == 409, r.text
        assert "active requests" in r.json().get("detail", "").lower()

    def test_sleep_allowed_when_engine_idle(self, monkeypatch):
        monkeypatch.setenv("YUNSHU_AUTH_DISABLED", "true")
        self._reset()

        class _IdleEngine:
            def has_active_requests(self):
                return False

            def __init__(self):
                self._model = object()

        monkeypatch.setattr("yunshu_gateway.engine.get_engine", lambda: _IdleEngine())
        app = _make_app_with_sleep_router()
        client = TestClient(app)
        r = client.post("/v1/sleep", json={"level": 1})
        assert r.status_code == 200, r.text
