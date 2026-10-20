"""An in-process loopback call must not take a second concurrency slot of the caller's key."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from yunshu_gateway import api_keys
from yunshu_gateway.routers import ollama

from .test_api_keys import H, make_app


@pytest.fixture(name="store")
def _store(tmp_path, monkeypatch):
    monkeypatch.delenv("YUNSHU_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("YUNSHU_AUTH_DISABLED", raising=False)
    s = api_keys.configure(tmp_path / "keys.json")
    yield s
    api_keys._store = None


def _outer_holds_slot(store):
    rec, sec = store.create("c", quotas={"max_concurrent": 1})
    store.admit(api_keys.Principal("key", frozenset({"infer"}), rec.id))
    return sec


def test_loopback_header_bypasses_concurrency_slot(store):
    sec = _outer_holds_slot(store)
    c = TestClient(make_app())
    assert c.post("/v1/chat/completions", headers=H(sec)).status_code == 429
    h = {**H(sec), api_keys.INTERNAL_HEADER: api_keys.INTERNAL_SECRET}
    assert c.post("/v1/chat/completions", headers=h).status_code == 200


def test_wrong_internal_secret_does_not_bypass(store):
    sec = _outer_holds_slot(store)
    c = TestClient(make_app())
    h = {**H(sec), api_keys.INTERNAL_HEADER: "guess"}
    assert c.post("/v1/chat/completions", headers=h).status_code == 429


def test_internal_header_still_needs_a_valid_key(store):
    _outer_holds_slot(store)
    c = TestClient(make_app())
    h = {**H("sk-bogus"), api_keys.INTERNAL_HEADER: api_keys.INTERNAL_SECRET}
    assert c.post("/v1/chat/completions", headers=h).status_code == 401


def test_ollama_loopback_client_marks_itself_internal():
    req = SimpleNamespace(
        headers={"authorization": "Bearer k"},
        state=SimpleNamespace(request_id=None),
        url=SimpleNamespace(scheme="http", netloc="h"),
        scope={"server": ("127.0.0.1", 18990)},
    )
    c = ollama._client(req)
    assert c.headers[api_keys.INTERNAL_HEADER] == api_keys.INTERNAL_SECRET
