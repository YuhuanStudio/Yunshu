"""Realtime query model uses the same lazy loader as HTTP."""

from unittest.mock import AsyncMock

import pytest

from yunshu_gateway.routers import realtime


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/v1/realtime", "/realtime"])
async def test_query_loads_before_session(monkeypatch, path):
    monkeypatch.setattr("yunshu_gateway.engine.get_engine", lambda: None)
    engine = object()
    load = AsyncMock(return_value=engine)
    monkeypatch.setattr("yunshu_gateway.engine.get_engine_for_model", load)

    async def run(session):
        assert session.session.model == "alias"
        assert session._resolve_engine() is engine

    monkeypatch.setattr(realtime.RealtimeSession, "run", run)
    monkeypatch.setattr(realtime.settings, "get", lambda key: "")
    ws = AsyncMock()
    ws.headers = {}
    ws.query_params = {"model": "alias"}
    ws.scope = {"path": path, "subprotocols": []}
    await realtime.realtime_endpoint(ws)
    load.assert_awaited_once_with("alias")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error,code",
    [
        (KeyError("missing"), "model_not_found"),
        (RuntimeError("load"), "model_load_failed"),
    ],
)
async def test_query_failure_does_not_start_session(monkeypatch, error, code):
    monkeypatch.setattr("yunshu_gateway.engine.get_engine", lambda: None)
    monkeypatch.setattr(
        "yunshu_gateway.engine.get_engine_for_model", AsyncMock(side_effect=error)
    )
    run = AsyncMock()
    monkeypatch.setattr(realtime.RealtimeSession, "run", run)
    monkeypatch.setattr(realtime.settings, "get", lambda key: "")
    ws = AsyncMock()
    ws.headers = {}
    ws.query_params = {"model": "missing"}
    ws.scope = {"path": "/v1/realtime", "subprotocols": []}
    await realtime.realtime_endpoint(ws)
    assert ws.send_json.call_args.args[0]["error"]["code"] == code
    run.assert_not_awaited()
    ws.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_single_model_name_is_advisory(monkeypatch):
    from types import SimpleNamespace

    engine = SimpleNamespace(is_loaded=True)
    monkeypatch.setattr("yunshu_gateway.engine.get_engine", lambda: engine)
    load = AsyncMock(side_effect=KeyError("client placeholder"))
    monkeypatch.setattr("yunshu_gateway.engine.get_engine_for_model", load)

    async def run(session):
        assert session._resolve_engine() is engine

    monkeypatch.setattr(realtime.RealtimeSession, "run", run)
    monkeypatch.setattr(realtime.settings, "get", lambda key: "")
    ws = AsyncMock()
    ws.headers = {}
    ws.query_params = {"model": "gpt-placeholder"}
    ws.scope = {"path": "/v1/realtime", "subprotocols": []}
    await realtime.realtime_endpoint(ws)
    load.assert_not_awaited()
