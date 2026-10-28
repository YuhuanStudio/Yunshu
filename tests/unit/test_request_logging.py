"""RequestLoggingMiddleware: request ids, log levels per status, slow-request warning."""

from __future__ import annotations

import logging

import pytest
from starlette.applications import Starlette
from starlette.responses import JSONResponse, PlainTextResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from yunshu_gateway.middleware import request_logging as rl


def _make_app() -> Starlette:
    async def ok(request):
        return JSONResponse({"request_id": request.state.request_id})

    async def missing(request):
        return PlainTextResponse("nope", status_code=404)

    async def broken(request):
        return PlainTextResponse("bad", status_code=503)

    async def boom(request):
        raise RuntimeError("kaboom")

    async def health(request):
        return PlainTextResponse("ok")

    app = Starlette(
        routes=[
            Route("/ok", ok),
            Route("/missing", missing),
            Route("/broken", broken),
            Route("/boom", boom),
            Route("/health", health),
        ]
    )
    app.add_middleware(rl.RequestLoggingMiddleware)
    return app


@pytest.fixture
def client(monkeypatch):
    # The memory probe shells out to sysctl; it is not what these tests are about.
    monkeypatch.setattr(rl, "_check_memory_pressure", lambda: None)
    monkeypatch.setattr(rl.RequestLoggingMiddleware, "_last_mem_check", -1e9)
    with TestClient(_make_app(), raise_server_exceptions=False) as c:
        yield c


def _records(caplog):
    return [r for r in caplog.records if r.name == "yunshu.gateway"]


def test_request_id_is_generated_and_echoed_in_the_response(client):
    r = client.get("/ok")
    rid = r.headers["x-request-id"]
    assert rid.startswith("req_") and len(rid) == len("req_") + 24
    assert r.json()["request_id"] == rid


def test_caller_request_id_is_kept(client):
    r = client.get("/ok", headers={"X-Request-ID": "trace-123"})
    assert r.headers["x-request-id"] == "trace-123"
    assert r.json()["request_id"] == "trace-123"


@pytest.mark.parametrize(
    ("path", "level"),
    [("/ok", logging.DEBUG), ("/missing", logging.WARNING), ("/broken", logging.ERROR)],
)
def test_level_follows_the_status_class(client, caplog, path, level):
    with caplog.at_level(logging.DEBUG, logger="yunshu.gateway"):
        client.get(path, headers={"X-Request-ID": "rid-1"})
    (rec,) = [r for r in _records(caplog) if "rid-1" in r.getMessage()]
    assert rec.levelno == level
    assert f"GET {path}" in rec.getMessage()


def test_skip_paths_are_not_logged(client, caplog):
    with caplog.at_level(logging.DEBUG, logger="yunshu.gateway"):
        assert client.get("/health").status_code == 200
    assert not _records(caplog)


def test_slow_requests_warn_but_5xx_do_not_double_report(client, caplog, monkeypatch):
    monkeypatch.setattr(rl, "SLOW_REQUEST_THRESHOLD", 0.0)
    with caplog.at_level(logging.DEBUG, logger="yunshu.gateway"):
        client.get("/ok", headers={"X-Request-ID": "slow-1"})
        client.get("/broken", headers={"X-Request-ID": "slow-2"})
    slow = [
        r.getMessage() for r in _records(caplog) if "SLOW_REQUEST" in r.getMessage()
    ]
    assert len(slow) == 1 and "slow-1" in slow[0]


def test_an_exception_is_logged_as_an_error_and_still_propagates(client, caplog):
    with caplog.at_level(logging.DEBUG, logger="yunshu.gateway"):
        r = client.get("/boom", headers={"X-Request-ID": "err-1"})
    assert r.status_code == 500
    errors = [r for r in _records(caplog) if r.levelno == logging.ERROR]
    assert any("err-1" in e.getMessage() and "kaboom" in e.getMessage() for e in errors)


def test_memory_pressure_probe_runs_at_most_once_per_interval(monkeypatch):
    calls = []
    monkeypatch.setattr(rl, "_check_memory_pressure", lambda: calls.append(1))
    monkeypatch.setattr(rl.RequestLoggingMiddleware, "_last_mem_check", -1e9)
    with TestClient(_make_app()) as c:
        for _ in range(5):
            c.get("/ok")
    assert len(calls) == 1
