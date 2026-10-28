"""Files API and conversations are per-key: an infer key sees only its own objects; the static
token (owner), auth-disabled mode and admin-scope keys see all. A foreign object reads as 404."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from yunshu_gateway import api_keys, files_store
from yunshu_gateway.middleware.auth import AuthMiddleware
from yunshu_gateway.routers import conversations as conv_router
from yunshu_gateway.routers import files as files_router


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.delenv("YUNSHU_AUTH_DISABLED", raising=False)
    monkeypatch.setenv("YUNSHU_AUTH_TOKEN", "tok")
    monkeypatch.setenv("YUNSHU_FILES_DIR", str(tmp_path / "files"))
    monkeypatch.setenv("YUNSHU_CONVERSATIONS_DIR", str(tmp_path / "conv"))
    files_store.reset_store()
    store = api_keys.configure(tmp_path / "keys.json")
    app = FastAPI()
    app.add_middleware(AuthMiddleware)
    app.include_router(files_router.router, prefix="/v1")
    app.include_router(conv_router.router, prefix="/v1")
    _, a = store.create("a", scopes=["infer"])
    _, b = store.create("b", scopes=["infer"])
    _, adm = store.create("adm", scopes=["infer", "admin"])
    yield TestClient(app), {"a": a, "b": b, "adm": adm, "tok": "tok"}
    api_keys._store = None
    files_store.reset_store()


def H(secret):
    return {"Authorization": f"Bearer {secret}"}


def _upload(c, secret, data=b"secret text"):
    r = c.post(
        "/v1/files",
        data={"purpose": "user_data"},
        files={"file": ("a.txt", data, "text/plain")},
        headers=H(secret),
    )
    assert r.status_code == 200, r.text
    return r.json()["id"]


def test_files_are_per_key(env):
    c, k = env
    fid = _upload(c, k["a"])
    # the owner, the static token and an admin key see it
    for who in ("a", "tok", "adm"):
        assert c.get(f"/v1/files/{fid}", headers=H(k[who])).status_code == 200
        assert c.get(f"/v1/files/{fid}/content", headers=H(k[who])).status_code == 200
        assert fid in [
            f["id"] for f in c.get("/v1/files", headers=H(k[who])).json()["data"]
        ]
    # another infer key gets 404 on every route and an empty list
    assert c.get(f"/v1/files/{fid}", headers=H(k["b"])).status_code == 404
    assert c.get(f"/v1/files/{fid}/content", headers=H(k["b"])).status_code == 404
    assert c.delete(f"/v1/files/{fid}", headers=H(k["b"])).status_code == 404
    assert c.get("/v1/files", headers=H(k["b"])).json()["data"] == []
    assert (
        c.get(f"/v1/files/{fid}", headers=H(k["a"])).status_code == 200
    )  # not deleted
    assert c.delete(f"/v1/files/{fid}", headers=H(k["a"])).status_code == 200


def test_legacy_unowned_files_stay_open(env):
    c, k = env
    meta = files_store.get_store().put(
        b"old", "old.txt"
    )  # stored before owners existed
    assert c.get(f"/v1/files/{meta['id']}", headers=H(k["b"])).status_code == 200


def test_file_references_in_a_request_respect_ownership(env):
    fid = _upload(env[0], env[1]["a"])
    block = {"type": "input_file", "file_id": fid}
    with files_store.file_access("key:other", False):
        with pytest.raises(files_store.FileRefError):
            files_store.resolve_file_refs([block])
    owner = files_store.get_store().get_meta(fid)["owner"]
    with files_store.file_access(owner, False):
        assert "file_data" in files_store.resolve_file_refs([block])[0]
    with files_store.file_access("key:other", True):  # admin
        assert "file_data" in files_store.resolve_file_refs([block])[0]
    assert (
        "file_data" in files_store.resolve_file_refs([block])[0]
    )  # no request context


def test_conversations_are_per_key(env):
    c, k = env
    r = c.post("/v1/conversations", json={"metadata": {"x": "1"}}, headers=H(k["a"]))
    assert r.status_code == 200, r.text
    cid = r.json()["id"]
    assert "owner" not in r.json()
    item = {"type": "message", "role": "user", "content": "hi"}
    add = c.post(
        f"/v1/conversations/{cid}/items", json={"items": [item]}, headers=H(k["a"])
    )
    iid = add.json()["data"][0]["id"]
    for who in ("a", "tok", "adm"):
        h = H(k[who])
        assert c.get(f"/v1/conversations/{cid}", headers=h).status_code == 200
        assert c.get(f"/v1/conversations/{cid}/items", headers=h).status_code == 200
        assert (
            c.get(f"/v1/conversations/{cid}/items/{iid}", headers=h).status_code == 200
        )
    h = H(k["b"])
    assert c.get(f"/v1/conversations/{cid}", headers=h).status_code == 404
    assert (
        c.post(f"/v1/conversations/{cid}", json={"metadata": {}}, headers=h).status_code
        == 404
    )
    assert (
        c.post(
            f"/v1/conversations/{cid}/items", json={"items": [item]}, headers=h
        ).status_code
        == 404
    )
    assert c.get(f"/v1/conversations/{cid}/items", headers=h).status_code == 404
    assert c.get(f"/v1/conversations/{cid}/items/{iid}", headers=h).status_code == 404
    assert (
        c.delete(f"/v1/conversations/{cid}/items/{iid}", headers=h).status_code == 404
    )
    assert c.delete(f"/v1/conversations/{cid}", headers=h).status_code == 404
    assert c.get(f"/v1/conversations/{cid}", headers=H(k["a"])).status_code == 200
    assert c.delete(f"/v1/conversations/{cid}", headers=H(k["a"])).status_code == 200


def test_legacy_unowned_conversation_stays_open(env):
    c, k = env
    from yunshu_gateway.conversations_store import get_store

    cid = get_store().create([], {})["id"]
    assert c.get(f"/v1/conversations/{cid}", headers=H(k["b"])).status_code == 200


def test_responses_conversation_reference_respects_ownership(env):
    from fastapi import Request

    from yunshu_gateway import responses_context
    from yunshu_gateway.conversations_store import ConversationNotFound

    c, k = env
    cid = c.post("/v1/conversations", json={}, headers=H(k["a"])).json()["id"]
    app = FastAPI()
    app.add_middleware(AuthMiddleware)

    @app.get("/gate")
    def gate(request: Request):
        try:
            responses_context._gate_conversation(request, cid)
            return {"ok": True}
        except ConversationNotFound:
            return {"ok": False}

    t = TestClient(app)
    assert t.get("/gate", headers=H(k["a"])).json() == {"ok": True}
    assert t.get("/gate", headers=H(k["b"])).json() == {"ok": False}
    assert t.get("/gate", headers=H(k["adm"])).json() == {"ok": True}


# --- batch outputs, voices, computer-use screenshots ---------------------------------------


@pytest.fixture
def fstore(tmp_path, monkeypatch):
    monkeypatch.setenv("YUNSHU_FILES_DIR", str(tmp_path / "files"))
    files_store.reset_store()
    yield files_store.get_store()
    files_store.reset_store()


def test_batch_output_files_belong_to_the_batch_creator(fstore):
    from yunshu_gateway.routers.batches import BatchRunner

    rec = {
        "id": "batch_" + "a" * 24,
        "api": "openai",
        "counts": {"succeeded": 1, "errored": 0, "canceled": 0, "expired": 0},
        "owner": "key:A",
    }
    fstore.save_batch(rec)
    fstore.append_line(rec["id"], "out", {"custom_id": "1"})
    BatchRunner._finalize(None, fstore, rec, [], "completed")
    fid = fstore.load_batch(rec["id"])["output_file_id"]
    meta = fstore.get_meta(fid)
    assert meta["owner"] == "key:A"
    assert files_store.visible_to(meta, "key:A", False)
    assert not files_store.visible_to(meta, "key:B", False)
    assert files_store.visible_to(meta, "key:B", True)


def test_new_batch_record_carries_the_creator(env):
    from fastapi import Request

    from yunshu_gateway.routers.batches import _new_record

    c, k = env
    seen = {}
    app = c.app

    @app.get("/who-batch")
    def who(request: Request):
        seen["owner"] = _new_record("openai", "batch", request, 1)["owner"]
        return {}

    c.get("/who-batch", headers=H(k["a"]))
    assert seen["owner"].startswith("key:")
    c.get("/who-batch", headers=H(k["tok"]))
    assert seen["owner"] == "owner"


def test_voices_are_per_key(fstore):
    from fastapi import HTTPException

    from yunshu_gateway.routers import voices

    cons = voices.put_record(
        b"RIFF",
        {"object": "audio.voice_consent", "name": "n", "language": "en"},
        "key:A",
    )
    voice = voices.put_record(
        b"RIFF",
        {"object": "audio.voice", "name": "v", "_consent": cons["id"]},
        "key:A",
    )
    with files_store.file_access("key:A", False):
        assert voices.get_record(voice["id"], "voice_")["name"] == "v"
        assert [v["id"] for v in voices.list_custom_voices()] == [voice["id"]]
    with files_store.file_access("key:B", False):
        with pytest.raises(HTTPException):
            voices.get_record(voice["id"], "voice_")
        assert voices.list_custom_voices() == []
    with files_store.file_access("key:B", True):
        assert voices.get_record(voice["id"], "voice_")
    legacy = voices.put_record(
        b"RIFF", {"object": "audio.voice_consent", "name": "n", "language": "en"}
    )
    with files_store.file_access("key:B", False):
        assert voices.get_record(legacy["id"], "cons_")


def test_computer_screenshot_file_id_respects_ownership(fstore):
    from fastapi import HTTPException

    from yunshu_gateway.server_tools.responses_loop import input_item_to_messages

    meta = fstore.put(b"\x89PNG....", "s.png", "vision", "image/png", owner="key:A")
    item = {
        "type": "computer_call_output",
        "call_id": "c1",
        "output": {"type": "computer_screenshot", "file_id": meta["id"]},
    }
    token = files_store._ACCESS.set(("key:B", False))
    try:
        with pytest.raises(HTTPException) as exc:
            input_item_to_messages(item)
        assert exc.value.status_code == 404
        files_store._ACCESS.set(("key:A", False))
        assert input_item_to_messages(item)[0]["role"] == "tool"
    finally:
        files_store._ACCESS.reset(token)


def test_responses_request_sets_the_file_access_of_its_caller(env, monkeypatch):
    """create_response limits every file_id it resolves (computer screenshots included) to the
    caller: the context var is set before anything reads a file."""
    from fastapi import HTTPException

    from yunshu_gateway.routers import responses as rr

    c, k = env
    seen = []

    def spy(req):
        seen.append(files_store._ACCESS.get())
        raise HTTPException(418, "stop")

    monkeypatch.setattr("yunshu_gateway.capability_gate.enforce_responses", spy)
    c.app.include_router(rr.router, prefix="/v1")
    for who in ("b", "tok"):
        r = c.post(
            "/v1/responses", json={"model": "m", "input": "hi"}, headers=H(k[who])
        )
        assert r.status_code == 418
    assert seen[0][1] is False and seen[0][0].startswith("key:")
    assert seen[1] == ("owner", True)
