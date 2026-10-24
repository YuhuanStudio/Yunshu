"""YUNSHU_OBSERVE: one switch for every observability setting nobody set explicitly."""

from __future__ import annotations

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from yunshu_engine import settings
from yunshu_gateway.routers import monitoring


@pytest.fixture(autouse=True)
def _clean(monkeypatch, tmp_path):
    for name in settings.REGISTRY:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    settings.clear_overrides()
    yield
    settings.clear_overrides()


def test_off_by_default():
    assert settings.get_bool("YUNSHU_DEBUG_ROUTES") is False
    assert settings.raw("YUNSHU_DEBUG_ROUTES")[1] == "default"


def test_observe_turns_everything_on(monkeypatch, tmp_path):
    monkeypatch.setenv("YUNSHU_OBSERVE", "1")
    assert settings.get_bool("YUNSHU_DEBUG_ROUTES")
    assert settings.get("YUNSHU_TELEMETRY") == "on"
    assert settings.get_bool("YUNSHU_SERVE_LOG")
    assert settings.get_bool("YUNSHU_REPLAY_TOKEN_IDS")
    assert settings.get("YUNSHU_AUDIT_LOG_FILE") == str(
        tmp_path / ".yunshu/logs/audit.log"
    )
    assert settings.get("YUNSHU_DEBUG_STREAM_CAPTURE").endswith("stream_capture.jsonl")
    for name in settings.OBSERVE_SETTINGS:
        assert settings.raw(name)[1] == "observe"
    rows = {r["name"]: r for r in settings.effective()}
    assert rows["YUNSHU_SERVE_LOG"]["source"] == "observe"


def test_explicit_beats_observe(monkeypatch):
    monkeypatch.setenv("YUNSHU_OBSERVE", "1")
    monkeypatch.setenv("YUNSHU_TELEMETRY", "off")
    monkeypatch.setenv("YUNSHU_SERVE_LOG", "0")
    settings.set_override("YUNSHU_DEBUG_ROUTES", "0")
    assert settings.get("YUNSHU_TELEMETRY") == "off"
    assert settings.get_bool("YUNSHU_SERVE_LOG") is False
    assert settings.get_bool("YUNSHU_DEBUG_ROUTES") is False
    assert settings.raw("YUNSHU_TELEMETRY")[1] == "env"
    assert settings.raw("YUNSHU_DEBUG_ROUTES")[1] == "cli"
    assert settings.get_bool("YUNSHU_REPLAY_TOKEN_IDS")  # still observed


def _request(host: str, headers: dict | None = None) -> Request:
    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/debug/engine",
            "headers": [(k.encode(), v.encode()) for k, v in (headers or {}).items()],
            "client": (host, 5000),
        }
    )


def test_debug_denied_without_token_and_without_observe():
    with pytest.raises(HTTPException) as e:
        monitoring._check_permission(_request("127.0.0.1"))
    assert e.value.status_code == 401


def test_observe_allows_loopback_only(monkeypatch):
    monkeypatch.setenv("YUNSHU_OBSERVE", "1")
    monitoring._check_permission(_request("127.0.0.1"))
    monitoring._check_permission(_request("::1"))
    for host in ("192.168.1.20", "10.0.0.2", "testclient"):
        with pytest.raises(HTTPException):
            monitoring._check_permission(_request(host))
    with pytest.raises(HTTPException):  # behind a proxy the peer is not the user
        monitoring._check_permission(
            _request("127.0.0.1", {"x-forwarded-for": "8.8.8.8"})
        )


def test_observe_with_token_still_needs_it(monkeypatch):
    monkeypatch.setenv("YUNSHU_OBSERVE", "1")
    monkeypatch.setenv("YUNSHU_AUTH_TOKEN", "secret")
    with pytest.raises(HTTPException):
        monitoring._check_permission(_request("127.0.0.1"))
    monitoring._check_permission(
        _request("127.0.0.1", {"authorization": "Bearer secret"})
    )


def test_serve_flag_registered():
    import inspect

    from yunshu_cli import serve

    assert "observe" in inspect.signature(serve.serve).parameters


def test_status_reports_observe(monkeypatch):
    from yunshu_gateway.routers import yunshu as yr

    assert yr._observe_on() is False
    monkeypatch.setenv("YUNSHU_OBSERVE", "1")
    assert yr._observe_on() is True


def test_debug_requests_lists_in_flight_with_engine_counters():
    import time
    from types import SimpleNamespace

    from yunshu_gateway.x_yunshu import RequestInfo, registry

    assert monitoring._in_flight_raw() == []
    t0 = time.perf_counter()
    info = RequestInfo(request_id="r-obs", method="POST", path="/v1/chat/completions")
    info.gen = SimpleNamespace(
        stats=SimpleNamespace(
            phase="decode",
            prompt_tokens=100,
            cached_tokens=64,
            prefill_done=36,
            prefill_total=36,
            prefill_known=True,
            generated=7,
            t_admit=t0 + 0.01,
            t_first=t0 + 0.5,
            t_last=t0 + 0.6,
        )
    )
    info.arrived = t0
    registry.add(info)
    try:
        (row,) = monitoring._in_flight_raw()
    finally:
        registry.remove(info)
    assert row["request_id"] == "r-obs"
    assert row["phase"] == "decode" and row["generated"] == 7
    assert row["prefill_done"] == 36 and row["cached_tokens"] == 64
    assert 9 <= row["t_admit_ms"] <= 11 and 499 <= row["t_first_ms"] <= 501
