"""Multi-model mode loads on demand: a server with registered models and nothing loaded yet is
ready (the real-server run of `serve --models-dir` stayed 503 on /health/ready until the first
request, so a service manager waiting for readiness never saw it up)."""

from __future__ import annotations

import types

import pytest
from fastapi.testclient import TestClient

from yunshu_gateway import main as gm


def _client(monkeypatch, entries):
    mgr = types.SimpleNamespace(list_entries=lambda: entries)
    monkeypatch.setattr(gm, "get_model_manager", lambda: mgr)
    monkeypatch.setattr(gm, "get_engine", lambda: None)
    app = gm.create_app()
    app.state.server_state = (
        gm.ServerState.RUNNING
    )  # other tests leave the module state draining
    return TestClient(app)


def entry(loaded):
    return types.SimpleNamespace(is_loaded=loaded)


@pytest.mark.parametrize("loaded", [False, True])
def test_registered_models_make_the_server_ready(monkeypatch, loaded):
    r = _client(monkeypatch, [entry(False), entry(loaded)]).get("/health/ready")
    assert r.status_code == 200
    body = r.json()
    assert body["ready"] is True
    assert body["checks"]["models_registered"] == 2
    assert body["checks"]["model_loaded"] is loaded


def test_no_registered_models_is_not_ready(monkeypatch):
    r = _client(monkeypatch, []).get("/health/ready")
    assert r.status_code == 503 and r.json()["checks"]["model_loaded"] is False
