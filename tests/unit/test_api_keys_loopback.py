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


def _batch_app(monkeypatch, tmp_path):
    import asyncio

    from yunshu_gateway import files_store
    from yunshu_gateway.routers import batches, files

    monkeypatch.setenv("YUNSHU_FILES_DIR", str(tmp_path / "files"))
    files_store.reset_store()
    app = make_app()
    app.include_router(files.router, prefix="/v1")
    app.include_router(batches.router, prefix="/v1")
    runner = batches.runner
    monkeypatch.setattr(runner, "autostart", False)
    monkeypatch.setattr(runner, "yield_seconds", 0)
    monkeypatch.setattr(runner, "client_factory", batches._default_client)
    monkeypatch.setattr(runner, "app", runner.app)

    def drain():
        runner.app = app
        asyncio.run(runner.process_all())

    return TestClient(app), drain


def _submit(c, sec, n):
    import json

    line = {
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {"model": "m", "messages": [{"role": "user", "content": "hi"}]},
    }
    data = b"".join(
        json.dumps({**line, "custom_id": f"c{i}"}).encode() + b"\n" for i in range(n)
    )
    fid = c.post(
        "/v1/files",
        headers=H(sec),
        data={"purpose": "batch"},
        files={"file": ("in.jsonl", data, "application/x-jsonl")},
    ).json()["id"]
    r = c.post(
        "/v1/batches",
        headers=H(sec),
        json={
            "input_file_id": fid,
            "endpoint": "/v1/chat/completions",
            "completion_window": "24h",
        },
    )
    assert r.status_code == 200, r.text
    return r.json()["id"]


def test_batch_lines_run_as_and_are_counted_to_the_submitting_key(
    store, monkeypatch, tmp_path
):
    # Stored keys only: the worker used to send no credential, so every line got 401.
    from yunshu_gateway import files_store

    rec, sec = store.create("batcher")
    c, drain = _batch_app(monkeypatch, tmp_path)
    bid = _submit(c, sec, 3)
    drain()
    batch = files_store.get_store().load_batch(bid)
    assert batch["status"] == "completed" and batch["counts"]["succeeded"] == 3, batch
    # upload + create + three lines, all on this key
    assert store._state[rec.id].req_sum == 5
    assert store._state[rec.id].inflight == 0


def test_batch_lines_respect_the_submitting_keys_daily_quota(
    store, monkeypatch, tmp_path
):
    from yunshu_gateway import files_store

    _, sec = store.create("small", quotas={"requests_per_day": 3})
    c, drain = _batch_app(monkeypatch, tmp_path)
    bid = _submit(c, sec, 3)  # upload + create leave one request for the lines
    drain()
    counts = files_store.get_store().load_batch(bid)["counts"]
    assert counts["succeeded"] == 1 and counts["errored"] == 2, counts


def test_act_as_header_needs_the_internal_secret(store):
    rec, _ = store.create("victim")
    c = TestClient(make_app())
    h = {api_keys.INTERNAL_HEADER: "guess", api_keys.INTERNAL_KEY_HEADER: rec.id}
    assert c.post("/v1/chat/completions", headers=h).status_code == 401
