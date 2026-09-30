"""Conversations API: CRUD, items, pagination, normalisation, persistence."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from yunshu_gateway import conversations_store as cs
from yunshu_gateway.routers import conversations as conv_router


@pytest.fixture
def store_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("YUNSHU_CONVERSATIONS_DIR", str(tmp_path / "convs"))
    monkeypatch.delenv("YUNSHU_AUTH_TOKEN", raising=False)
    cs.reset_store()
    yield tmp_path / "convs"
    cs.reset_store()


@pytest.fixture
def client(store_dir):
    app = FastAPI()
    app.include_router(conv_router.router, prefix="/v1")
    return TestClient(app)


def _msgs(n, prefix="m"):
    return [{"role": "user", "content": f"{prefix}{i}"} for i in range(n)]


def test_crud_roundtrip(client):
    r = client.post("/v1/conversations", json={"metadata": {"topic": "demo"}})
    assert r.status_code == 200
    c = r.json()
    assert c["id"].startswith("conv_") and len(c["id"]) == 5 + 24
    assert c["object"] == "conversation" and c["metadata"] == {"topic": "demo"}
    assert isinstance(c["created_at"], int)
    assert client.get(f"/v1/conversations/{c['id']}").json() == c
    u = client.post(f"/v1/conversations/{c['id']}", json={"metadata": {"a": "b"}})
    assert u.json()["metadata"] == {"a": "b"}
    d = client.delete(f"/v1/conversations/{c['id']}")
    assert d.json() == {
        "id": c["id"],
        "object": "conversation.deleted",
        "deleted": True,
    }
    r = client.get(f"/v1/conversations/{c['id']}")
    assert r.status_code == 404
    err = r.json()["error"]
    assert err["type"] == "invalid_request_error"
    assert err["code"] == "conversation_not_found"


def test_openai_sdk(client):
    openai = pytest.importorskip("openai")
    sdk = openai.OpenAI(
        base_url="http://testserver/v1", api_key="x", http_client=client
    )
    c = sdk.conversations.create(
        items=[{"type": "message", "role": "user", "content": "hi"}],
        metadata={"k": "v"},
    )
    assert c.id.startswith("conv_") and c.metadata == {"k": "v"}
    items = sdk.conversations.items.list(c.id, order="asc")
    assert [i.type for i in items] == ["message"]
    added = sdk.conversations.items.create(
        c.id, items=[{"type": "message", "role": "assistant", "content": "yo"}]
    )
    assert added.data[0].id.startswith("msg_")
    got = sdk.conversations.items.retrieve(added.data[0].id, conversation_id=c.id)
    assert got.role == "assistant"
    conv = sdk.conversations.items.delete(added.data[0].id, conversation_id=c.id)
    assert conv.id == c.id
    assert sdk.conversations.delete(c.id).deleted is True
    with pytest.raises(openai.NotFoundError):
        sdk.conversations.retrieve(c.id)


def test_item_normalisation(client):
    c = client.post(
        "/v1/conversations",
        json={
            "items": [
                {"role": "user", "content": "plain"},
                {"type": "message", "role": "assistant", "content": "answer"},
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "kept"}],
                },
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "f",
                    "arguments": "{}",
                },
                {"type": "function_call_output", "call_id": "call_1", "output": "ok"},
                {"type": "reasoning", "summary": []},
                {"type": "web_search_call", "status": "completed"},
                {"type": "mcp_call", "name": "n", "server_label": "s"},
                {"type": "message", "role": "user", "id": "msg_mine", "content": "x"},
            ]
        },
    ).json()
    data = client.get(
        f"/v1/conversations/{c['id']}/items", params={"order": "asc"}
    ).json()["data"]
    assert data[0]["content"] == [{"type": "input_text", "text": "plain"}]
    assert data[0]["id"].startswith("msg_") and data[0]["status"] == "completed"
    assert data[1]["content"] == [{"type": "output_text", "text": "answer"}]
    assert data[2]["content"] == [{"type": "input_text", "text": "kept"}]
    assert data[3]["id"].startswith("fc_") and data[3]["call_id"] == "call_1"
    assert data[4]["id"].startswith("fco_")
    assert data[5]["id"].startswith("rs_")
    assert data[6]["id"].startswith("ws_")
    assert data[7]["id"].startswith("mcp_") and data[7]["name"] == "n"
    assert data[8]["id"] == "msg_mine"


def test_pagination(client):
    c = client.post("/v1/conversations", json={"items": _msgs(12)}).json()
    base = f"/v1/conversations/{c['id']}/items"
    desc = client.get(base, params={"limit": 5}).json()
    assert desc["object"] == "list" and desc["has_more"] is True
    texts = [i["content"][0]["text"] for i in desc["data"]]
    assert texts == ["m11", "m10", "m9", "m8", "m7"]
    assert desc["first_id"] == desc["data"][0]["id"]
    assert desc["last_id"] == desc["data"][-1]["id"]
    nxt = client.get(base, params={"limit": 5, "after": desc["last_id"]}).json()
    assert [i["content"][0]["text"] for i in nxt["data"]] == [
        "m6",
        "m5",
        "m4",
        "m3",
        "m2",
    ]
    last = client.get(base, params={"limit": 5, "after": nxt["last_id"]}).json()
    assert [i["content"][0]["text"] for i in last["data"]] == ["m1", "m0"]
    assert last["has_more"] is False
    asc = client.get(base, params={"order": "asc", "limit": 3}).json()
    assert [i["content"][0]["text"] for i in asc["data"]] == ["m0", "m1", "m2"]
    assert len(client.get(base).json()["data"]) == 12  # default limit 20
    assert client.get(base, params={"limit": 0}).status_code == 400
    assert client.get(base, params={"limit": 101}).status_code == 400
    assert client.get(base, params={"order": "up"}).status_code == 400
    assert client.get(base, params={"after": "msg_nope"}).status_code == 404


def test_twenty_item_cap(client):
    r = client.post("/v1/conversations", json={"items": _msgs(21)})
    assert r.status_code == 400 and r.json()["error"]["code"] == "too_many_items"
    c = client.post("/v1/conversations", json={"items": _msgs(20)}).json()
    r = client.post(f"/v1/conversations/{c['id']}/items", json={"items": _msgs(21)})
    assert r.status_code == 400
    r = client.post(f"/v1/conversations/{c['id']}/items", json={"items": _msgs(20)})
    assert r.status_code == 200 and len(r.json()["data"]) == 20


def test_item_get_delete_and_errors(client):
    c = client.post("/v1/conversations", json={"items": _msgs(2)}).json()
    items = client.get(
        f"/v1/conversations/{c['id']}/items", params={"order": "asc"}
    ).json()["data"]
    one = client.get(f"/v1/conversations/{c['id']}/items/{items[0]['id']}")
    assert one.json() == items[0]
    d = client.delete(f"/v1/conversations/{c['id']}/items/{items[0]['id']}")
    assert d.json()["id"] == c["id"] and d.json()["object"] == "conversation"
    assert (
        client.get(f"/v1/conversations/{c['id']}/items/{items[0]['id']}").status_code
        == 404
    )
    assert (
        client.delete(f"/v1/conversations/{c['id']}/items/{items[0]['id']}").status_code
        == 404
    )
    assert (
        client.get("/v1/conversations/conv_" + "0" * 24 + "/items").status_code == 404
    )
    assert client.get("/v1/conversations/..%2Fetc").status_code == 404
    assert (
        client.post("/v1/conversations", json={"metadata": {"a": 1}}).status_code == 400
    )
    too_many = {f"k{i}": "v" for i in range(17)}
    assert (
        client.post("/v1/conversations", json={"metadata": too_many}).status_code == 400
    )
    assert (
        client.post("/v1/conversations", json={"items": [{"foo": 1}]}).status_code
        == 400
    )
    assert client.post(f"/v1/conversations/{c['id']}", json={}).status_code == 400


def test_persistence_across_reload(client, store_dir):
    c = client.post(
        "/v1/conversations", json={"items": _msgs(3), "metadata": {"a": "b"}}
    ).json()
    cs.reset_store()
    got = client.get(f"/v1/conversations/{c['id']}").json()
    assert got == c
    items = client.get(
        f"/v1/conversations/{c['id']}/items", params={"order": "asc"}
    ).json()["data"]
    assert [i["content"][0]["text"] for i in items] == ["m0", "m1", "m2"]
    assert (store_dir / f"{c['id']}.json").exists()
    assert not list(store_dir.glob(".*.tmp"))


def test_max_items_setting(client, monkeypatch):
    monkeypatch.setenv("YUNSHU_CONVERSATION_MAX_ITEMS", "3")
    c = client.post("/v1/conversations", json={"items": _msgs(2)}).json()
    r = client.post(f"/v1/conversations/{c['id']}/items", json={"items": _msgs(2)})
    assert r.status_code == 400 and r.json()["error"]["code"] == "too_many_items"
    assert (
        client.post(
            f"/v1/conversations/{c['id']}/items", json={"items": _msgs(1)}
        ).status_code
        == 200
    )
