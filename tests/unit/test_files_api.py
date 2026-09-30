"""Files API (OpenAI + Anthropic shapes) and the file-reference helpers."""

from __future__ import annotations

import base64
import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from yunshu_gateway import files_store
from yunshu_gateway.routers import files as files_router


@pytest.fixture
def store_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("YUNSHU_FILES_DIR", str(tmp_path / "files"))
    monkeypatch.delenv("YUNSHU_AUTH_TOKEN", raising=False)
    files_store.reset_store()
    yield tmp_path / "files"
    files_store.reset_store()


@pytest.fixture
def client(store_dir):
    app = FastAPI()
    app.include_router(files_router.router, prefix="/v1")
    return TestClient(app)


ANTH = {"anthropic-version": "2023-06-01", "anthropic-beta": "files-api-2025-04-14"}


def _upload(
    client,
    data=b"hello world",
    name="a.txt",
    purpose="user_data",
    headers=None,
    mime="text/plain",
):
    data_form = {} if headers else {"purpose": purpose}
    return client.post(
        "/v1/files",
        data=data_form,
        files={"file": (name, data, mime)},
        headers=headers or {},
    )


def test_openai_sdk_roundtrip(client):
    openai = pytest.importorskip("openai")
    sdk = openai.OpenAI(
        base_url="http://testserver/v1", api_key="x", http_client=client
    )
    f = sdk.files.create(file=("notes.txt", b"abc", "text/plain"), purpose="assistants")
    assert f.id.startswith("file_") and f.bytes == 3 and f.purpose == "assistants"
    assert f.status == "processed" and f.filename == "notes.txt"
    assert sdk.files.retrieve(f.id).id == f.id
    assert [x.id for x in sdk.files.list(purpose="assistants")] == [f.id]
    assert list(sdk.files.list(purpose="batch")) == []
    assert sdk.files.content(f.id).read() == b"abc"
    d = sdk.files.delete(f.id)
    assert d.deleted is True and d.id == f.id
    with pytest.raises(openai.NotFoundError):
        sdk.files.retrieve(f.id)


def test_openai_content_type_and_bytes(client):
    fid = _upload(
        client, b"\x89PNG\r\n\x1a\nxx", "p.bin", mime="application/octet-stream"
    ).json()["id"]
    r = client.get(f"/v1/files/{fid}/content")
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"
    assert r.content == b"\x89PNG\r\n\x1a\nxx"


def test_openai_validation(client):
    r = _upload(client, purpose="bogus")
    assert r.status_code == 400 and r.json()["error"]["param"] == "purpose"
    r = client.post("/v1/files", data={"purpose": "batch"})
    assert r.status_code == 400
    assert _upload(client, b"").status_code == 400
    r = client.get("/v1/files/not-an-id")
    assert r.status_code == 404 and r.json()["error"]["type"] == "not_found_error"
    r = client.get("/v1/files/..%2F..%2Fetc%2Fpasswd")
    assert r.status_code == 404
    assert client.get("/v1/files?order=sideways").status_code == 400
    assert client.get("/v1/files?limit=0").status_code == 400


def test_openai_expires_after(client):
    r = client.post(
        "/v1/files",
        data={
            "purpose": "batch",
            "expires_after[anchor]": "created_at",
            "expires_after[seconds]": "3600",
        },
        files={"file": ("x.jsonl", b"{}", "application/x-jsonl")},
    )
    body = r.json()
    assert body["expires_at"] == body["created_at"] + 3600
    bad = client.post(
        "/v1/files",
        data={
            "purpose": "batch",
            "expires_after[anchor]": "created_at",
            "expires_after[seconds]": "5",
        },
        files={"file": ("x", b"{}", "text/plain")},
    )
    assert bad.status_code == 400


def test_openai_pagination(client):
    ids = []
    for i in range(5):
        ids.append(_upload(client, b"x", f"{i}.txt", "user_data").json()["id"])
        # distinct created_at is not guaranteed within a second; order falls back to id
    all_ids = [f["id"] for f in client.get("/v1/files").json()["data"]]
    assert sorted(all_ids) == sorted(ids)
    p1 = client.get("/v1/files?limit=2").json()
    assert (
        p1["has_more"] is True and len(p1["data"]) == 2 and p1["first_id"] == all_ids[0]
    )
    p2 = client.get(f"/v1/files?limit=2&after={p1['last_id']}").json()
    assert [f["id"] for f in p2["data"]] == all_ids[2:4]
    p3 = client.get(f"/v1/files?limit=2&after={p2['last_id']}").json()
    assert len(p3["data"]) == 1 and p3["has_more"] is False
    asc = [f["id"] for f in client.get("/v1/files?order=asc").json()["data"]]
    assert asc == all_ids[::-1]


def test_anthropic_shapes(client):
    r = _upload(client, b"hello", "doc.txt", headers=ANTH)
    assert r.status_code == 200
    f = r.json()
    assert (
        f["type"] == "file" and f["size_bytes"] == 5 and f["mime_type"] == "text/plain"
    )
    assert f["downloadable"] is False and f["created_at"].endswith("Z")
    assert client.get(f"/v1/files/{f['id']}", headers=ANTH).json()["id"] == f["id"]
    lst = client.get("/v1/files", headers=ANTH).json()
    assert lst["has_more"] is False and lst["first_id"] == lst["last_id"] == f["id"]
    assert "object" not in lst
    # uploads cannot be downloaded through the Anthropic API
    r = client.get(f"/v1/files/{f['id']}/content", headers=ANTH)
    assert r.status_code == 403 and r.json()["type"] == "error"
    assert r.json()["error"]["type"] == "permission_error"
    d = client.delete(f"/v1/files/{f['id']}", headers=ANTH).json()
    assert d == {"id": f["id"], "type": "file_deleted"}
    r = client.get(f"/v1/files/{f['id']}", headers=ANTH)
    assert r.status_code == 404 and r.json()["error"]["type"] == "not_found_error"


def test_anthropic_downloadable_and_paging(client):
    store = files_store.get_store()
    meta = store.put(
        b"tool output", "out.txt", "user_data", "text/plain", downloadable=True
    )
    r = client.get(f"/v1/files/{meta['id']}/content", headers=ANTH)
    assert r.status_code == 200 and r.content == b"tool output"
    ids = [store.put(b"x", f"{i}", "user_data")["id"] for i in range(3)]
    page = client.get("/v1/files?limit=2", headers=ANTH).json()
    assert page["has_more"] is True and len(page["data"]) == 2
    nxt = client.get(
        f"/v1/files?limit=5&after_id={page['last_id']}", headers=ANTH
    ).json()
    assert len(nxt["data"]) == 2 and nxt["has_more"] is False
    back = client.get(
        f"/v1/files?before_id={nxt['data'][0]['id']}", headers=ANTH
    ).json()
    assert [x["id"] for x in back["data"]] == [x["id"] for x in page["data"]]
    assert client.get("/v1/files?limit=1001", headers=ANTH).status_code == 400
    assert len(ids) == 3


def test_size_limit(client, monkeypatch):
    monkeypatch.setenv("YUNSHU_FILES_MAX_BYTES", "10")
    r = _upload(client, b"x" * 11)
    assert r.status_code == 413
    assert _upload(client, b"x" * 10).status_code == 200
    r = _upload(client, b"x" * 11, headers=ANTH)
    assert r.status_code == 413 and r.json()["type"] == "error"


def test_auth_token_required(client, monkeypatch):
    monkeypatch.setenv("YUNSHU_AUTH_TOKEN", "sekret")
    monkeypatch.setenv("YUNSHU_AUTH_DISABLED", "false")
    assert client.get("/v1/files").status_code == 401
    r = client.get(
        "/v1/files", headers={"anthropic-version": "1", "x-api-key": "sekret"}
    )
    assert r.status_code == 200
    r = client.get("/v1/files", headers={"Authorization": "Bearer sekret"})
    assert r.status_code == 200
    r = client.get("/v1/files", headers=ANTH)
    assert r.status_code == 401 and r.json()["type"] == "error"


def test_persistence_across_reload(client, store_dir):
    fid = _upload(client, b"persist").json()["id"]
    files_store.reset_store()
    assert client.get(f"/v1/files/{fid}/content").content == b"persist"


def test_expiry_and_ttl(store_dir, monkeypatch):
    monkeypatch.setenv("YUNSHU_FILES_TTL_DAYS", "1")
    store = files_store.get_store()
    m = store.put(b"x", "a")
    assert m["expires_at"] - m["created_at"] == 86400
    meta_p = store.root / "meta" / f"{m['id']}.json"
    import json

    d = json.loads(meta_p.read_text())
    d["expires_at"] = int(time.time()) - 1
    meta_p.write_text(json.dumps(d))
    with pytest.raises(files_store.FileNotFound):
        store.get_meta(m["id"])
    assert not (store.root / "blobs" / m["id"]).exists()


def test_id_validation_blocks_traversal(store_dir):
    store = files_store.get_store()
    for bad in ("../x", "file_../../etc", "file_zz", "", "/etc/passwd"):
        with pytest.raises(files_store.FileNotFound):
            store.read(bad)


# -- helpers for the other routers -----------------------------------------


def test_helpers(store_dir):
    store = files_store.get_store()
    txt = store.put("héllo".encode(), "n.txt", "user_data", "text/plain")["id"]
    png = store.put(b"\x89PNG\r\n\x1a\n0000", "i.png", "vision")["id"]
    pdf = store.put(b"%PDF-1.4 x", "d.pdf", "user_data")["id"]
    assert files_store.read_file_bytes(txt) == "héllo".encode()
    assert files_store.text_of(txt) == "héllo"
    assert files_store.file_mime(pdf) == "application/pdf"
    assert files_store.guess_mime("x.md") == "text/markdown"
    assert files_store.guess_mime(None, b"GIF89a....") == "image/gif"
    assert (
        files_store.guess_mime("a.bin", None, "application/octet-stream")
        == "application/octet-stream"
    )
    with pytest.raises(files_store.FileRefError) as ei:
        files_store.read_file_bytes("file_" + "0" * 24)
    assert ei.value.status == 404

    doc = files_store.resolve_file_block(
        {"type": "document", "title": "T", "source": {"type": "file", "file_id": txt}}
    )
    assert doc["title"] == "T" and doc["source"] == {
        "type": "text",
        "media_type": "text/plain",
        "data": "héllo",
    }
    pdfb = files_store.resolve_file_block(
        {"type": "document", "source": {"type": "file", "file_id": pdf}}
    )
    assert (
        pdfb["source"]["type"] == "base64"
        and pdfb["source"]["media_type"] == "application/pdf"
    )
    assert base64.b64decode(pdfb["source"]["data"]) == b"%PDF-1.4 x"
    img = files_store.resolve_file_block(
        {"type": "image", "source": {"type": "file", "file_id": png}}
    )
    assert img["source"]["media_type"] == "image/png"

    inp = files_store.resolve_file_block({"type": "input_file", "file_id": txt})
    assert inp["filename"] == "n.txt" and inp["file_data"].startswith(
        "data:text/plain;base64,"
    )
    assert "file_id" not in inp
    ii = files_store.resolve_file_block({"type": "input_image", "file_id": png})
    assert ii["image_url"].startswith("data:image/png;base64,")
    cc = files_store.resolve_file_block({"type": "file", "file": {"file_id": pdf}})
    assert cc["file"]["file_data"].startswith("data:application/pdf;base64,")
    iu = files_store.resolve_file_block(
        {"type": "image_url", "image_url": {"url": png}}
    )
    assert iu["image_url"]["url"].startswith("data:image/png;base64,")
    # ordinary blocks and https urls pass through untouched
    plain = {"type": "image_url", "image_url": {"url": "https://x/y.png"}}
    assert files_store.resolve_file_block(plain) is plain
    # openai-style "file-" spelling of the id resolves too
    assert files_store.read_file_bytes("file-" + txt[5:]) == "héllo".encode()

    msgs = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "hi"},
                {"type": "document", "source": {"type": "file", "file_id": txt}},
            ],
        }
    ]
    assert files_store.has_file_refs(msgs)
    out = files_store.resolve_file_refs(msgs)
    assert out[0]["content"][1]["source"]["type"] == "text"
    assert msgs[0]["content"][1]["source"]["type"] == "file"  # input untouched
    assert not files_store.has_file_refs(out)
    with pytest.raises(files_store.FileRefError):
        files_store.resolve_file_block(
            {"type": "input_file", "file_id": "file_" + "1" * 24}
        )
