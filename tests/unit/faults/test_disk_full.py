"""A full disk under the stores (files, conversations) is a 507 the client can act on, never an
opaque 500, and leaves no temp files or half-written records. The APC / text SSD spill tiers
have their own ENOSPC tests in ``tests/unit/test_cache_disk_budget.py``."""

from __future__ import annotations

import errno
import os

import pytest
from fastapi.testclient import TestClient

from yunshu_gateway import conversations_store, files_store
from yunshu_gateway.main import create_app


def _enospc(*_a, **_k):
    raise OSError(errno.ENOSPC, "No space left on device")


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("YUNSHU_FILES_DIR", str(tmp_path / "files"))
    monkeypatch.setenv("YUNSHU_CONVERSATIONS_DIR", str(tmp_path / "conv"))
    files_store.reset_store()
    conversations_store.reset_store()
    with TestClient(create_app(), raise_server_exceptions=False) as c:
        yield c
    files_store.reset_store()
    conversations_store.reset_store()


def _leftovers(root):
    return sorted(p.name for p in root.rglob("*") if p.is_file())


def _upload(client, headers=None):
    return client.post(
        "/v1/files",
        files={"file": ("a.txt", b"hello", "text/plain")},
        data={"purpose": "user_data"},
        headers=headers or {},
    )


@pytest.mark.parametrize("fail_on_call", [1, 2], ids=["blob", "meta"])
def test_file_upload_on_a_full_disk_is_507_and_keeps_nothing(
    client, tmp_path, monkeypatch, fail_on_call
):
    calls = {"n": 0}
    real = os.fsync

    def fsync(fd):
        calls["n"] += 1
        if calls["n"] == fail_on_call:
            _enospc()
        return real(fd)

    monkeypatch.setattr(files_store.os, "fsync", fsync)
    r = _upload(client)
    assert r.status_code == 507
    err = r.json()["error"]
    assert err["code"] == "insufficient_storage" and err["type"] == "server_error"
    assert "disk is full" in err["message"]
    assert _leftovers(tmp_path / "files") == []  # no blob, no meta, no temp file
    assert client.get("/v1/files").json()["data"] == []
    # space is back: the same upload works
    monkeypatch.setattr(files_store.os, "fsync", real)
    assert _upload(client).status_code == 200


def test_file_upload_on_a_full_disk_speaks_anthropic_when_asked(client, monkeypatch):
    monkeypatch.setattr(files_store.os, "fsync", _enospc)
    r = _upload(client, {"anthropic-version": "2023-06-01"})
    assert r.status_code == 507
    assert r.json()["type"] == "error" and r.json()["error"]["type"] == "api_error"


def test_conversation_write_on_a_full_disk_is_507_and_keeps_nothing(
    client, tmp_path, monkeypatch
):
    monkeypatch.setattr(conversations_store.os, "fsync", _enospc)
    r = client.post("/v1/conversations", json={"metadata": {"a": "b"}})
    assert r.status_code == 507
    err = r.json()["error"]
    assert err["code"] == "insufficient_storage" and err["type"] == "server_error"
    assert _leftovers(tmp_path / "conv") == []


def test_conversation_survives_a_failed_item_append(client, tmp_path, monkeypatch):
    conv = client.post("/v1/conversations", json={}).json()
    item = {"type": "message", "role": "user", "content": "hi"}
    ok = client.post(f"/v1/conversations/{conv['id']}/items", json={"items": [item]})
    assert ok.status_code == 200
    real = os.fsync
    monkeypatch.setattr(conversations_store.os, "fsync", _enospc)
    r = client.post(f"/v1/conversations/{conv['id']}/items", json={"items": [item]})
    assert r.status_code == 507
    monkeypatch.setattr(conversations_store.os, "fsync", real)
    # the earlier item is intact and no temp file is left behind
    items = client.get(f"/v1/conversations/{conv['id']}/items").json()["data"]
    assert len(items) == 1
    assert [n for n in _leftovers(tmp_path / "conv") if n.endswith(".tmp")] == []


def test_any_other_route_that_hits_a_full_disk_gets_the_same_507(monkeypatch):
    app = create_app()

    @app.post("/v1/_probe_disk_full")
    async def probe():
        _enospc()

    with TestClient(app, raise_server_exceptions=False) as c:
        r = c.post("/v1/_probe_disk_full")
        assert r.status_code == 507
        assert r.json()["error"]["code"] == "insufficient_storage"
