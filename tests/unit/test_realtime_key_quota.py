"""Realtime counts each response to the stored key the session runs as (directly or
through a secret that key minted): quotas apply, the slot is released, tokens counted."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import yunshu_gateway.routers.realtime as rt
from yunshu_gateway import api_keys
from yunshu_gateway import realtime_secrets as rs


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.delenv("YUNSHU_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("YUNSHU_AUTH_DISABLED", raising=False)
    s = api_keys.configure(tmp_path / "keys.json")
    yield s
    api_keys._store = None


def _session(key_id):
    ws = MagicMock()
    ws.send_json = AsyncMock()
    s = rt.RealtimeSession(ws)
    s._key_id = key_id
    sent = []

    async def send_event(ev):
        sent.append(ev)

    s.send_event = send_event
    return s, sent


def test_response_is_refused_past_the_keys_daily_quota(store):
    rec, _ = store.create("rt", quotas={"requests_per_day": 1})
    s, sent = _session(rec.id)
    assert asyncio.run(s._admit_key())
    s._settle_key()
    assert not asyncio.run(s._admit_key())
    assert sent[-1]["error"]["code"] == "requests_per_day_exceeded"


def test_response_holds_a_slot_and_counts_its_tokens(store):
    rec, _ = store.create("rt", quotas={"max_concurrent": 1})
    s, sent = _session(rec.id)
    assert asyncio.run(s._admit_key())
    other, _ = _session(rec.id)
    assert not asyncio.run(other._admit_key())  # the key's one slot is taken
    s._last_usage = (30, 7)
    s._settle_key()
    st = store._state[rec.id]
    assert st.inflight == 0 and st.tok_sum == 37


def test_a_minted_secret_carries_the_minting_key(store):
    from yunshu_gateway.middleware.auth import AuthMiddleware
    from yunshu_gateway.routers import realtime_secrets

    rec, sec = store.create("minter")
    app = FastAPI()
    app.add_middleware(AuthMiddleware)
    app.include_router(realtime_secrets.router, prefix="/v1")
    r = TestClient(app).post(
        "/v1/realtime/client_secrets",
        json={"session": {"type": "realtime"}},
        headers={"Authorization": f"Bearer {sec}"},
    )
    assert r.status_code == 200, r.text
    assert rs.lookup(r.json()["value"]).key_id == rec.id


def test_response_create_past_the_quota_starts_no_generation(store):
    rec, _ = store.create("rt", quotas={"requests_per_day": 1})
    store.admit(store.principal_by_id(rec.id))  # the day's one request is spent
    s, sent = _session(rec.id)
    started = []
    s._generate_response = lambda *a, **k: started.append(a)
    asyncio.run(s._handle_response_create({"type": "response.create"}))
    assert not started and s._active_response is None
    assert sent and sent[-1]["error"]["code"] == "requests_per_day_exceeded"
