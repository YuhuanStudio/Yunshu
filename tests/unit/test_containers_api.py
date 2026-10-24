"""OpenAI /v1/containers and container files, driven through the official openai SDK (no srt needed:
the executor is a fake that only reports srt as present)."""

from __future__ import annotations

import os
from pathlib import Path

import openai
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from yunshu_gateway import files_store
from yunshu_gateway.routers import containers as containers_router
from yunshu_gateway.sandbox import containers as cont
from yunshu_gateway.sandbox import executor as exmod
from yunshu_gateway.sandbox.executor import CodeExecutor
from yunshu_gateway.sandbox.srt import SrtInfo


class FakeExec(CodeExecutor):
    def srt(self):
        return SrtInfo("/n", "/c")


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("YUNSHU_CODE_EXECUTION", "1")
    monkeypatch.setenv("YUNSHU_CODE_EXECUTION_ROOT", str(tmp_path / "sbx"))
    monkeypatch.setenv("YUNSHU_FILES_DIR", str(tmp_path / "files"))
    files_store.reset_store()
    cont.reset_manager()
    monkeypatch.setattr(exmod, "_executor", FakeExec(cont.get_manager()))
    app = FastAPI()
    app.include_router(containers_router.router, prefix="/v1")
    http = TestClient(app)
    client = openai.OpenAI(
        base_url="http://testserver/v1", api_key="k", http_client=http, max_retries=0
    )
    yield client, http, tmp_path
    cont.reset_manager()


def test_off_is_400_and_missing_srt_is_503(env, monkeypatch):
    client, http, _ = env
    monkeypatch.setenv("YUNSHU_CODE_EXECUTION", "0")
    with pytest.raises(openai.BadRequestError, match="YUNSHU_CODE_EXECUTION"):
        client.containers.create(name="x")
    monkeypatch.setenv("YUNSHU_CODE_EXECUTION", "1")

    class NoSrt(CodeExecutor):
        def srt(self):
            raise exmod.SandboxUnavailableError("srt missing; npm install")

    monkeypatch.setattr(exmod, "_executor", NoSrt(cont.get_manager()))
    with pytest.raises(openai.InternalServerError, match="srt missing"):
        client.containers.create(name="x")
    assert http.get("/v1/containers").status_code == 503


def test_create_retrieve_list_delete_lifecycle(env):
    client, _, _ = env
    a = client.containers.create(name="alpha")
    b = client.containers.create(name="beta")
    assert a.id.startswith("cntr_") and a.object == "container"
    assert a.name == "alpha" and a.status == "running"
    assert a.expires_after and a.expires_after.anchor == "last_active_at"
    assert a.expires_after.minutes == 20 and a.memory_limit == "1g"
    assert a.network_policy and a.network_policy.type == "disabled"
    got = client.containers.retrieve(a.id)
    assert got.id == a.id and got.created_at == a.created_at
    page = client.containers.list()
    assert [c.id for c in page.data] == [b.id, a.id]  # newest first
    asc = client.containers.list(order="asc", limit=1)
    assert [c.id for c in asc.data] == [a.id] and asc.has_more
    nxt = client.containers.list(order="asc", limit=1, after=a.id)
    assert [c.id for c in nxt.data] == [b.id] and not nxt.has_more
    assert [c.id for c in client.containers.list(name="beta").data] == [b.id]
    assert client.containers.delete(a.id) is None
    with pytest.raises(openai.NotFoundError):
        client.containers.retrieve(a.id)
    with pytest.raises(openai.NotFoundError):
        client.containers.delete(a.id)
    assert [c.id for c in client.containers.list().data] == [b.id]


def test_expiry_maps_onto_the_server_ttl(env, monkeypatch):
    client, _, _ = env
    monkeypatch.setenv("YUNSHU_CODE_EXECUTION_TTL_MIN", "10")
    long = client.containers.create(
        name="l", expires_after={"anchor": "last_active_at", "minutes": 60}
    )
    short = client.containers.create(
        name="s", expires_after={"anchor": "last_active_at", "minutes": 3}
    )
    assert long.expires_after.minutes == 10  # capped by YUNSHU_CODE_EXECUTION_TTL_MIN
    assert short.expires_after.minutes == 3
    for bad in (
        {"anchor": "created_at", "minutes": 5},
        {"anchor": "last_active_at", "minutes": 0},
        {"anchor": "last_active_at"},
    ):
        with pytest.raises(openai.BadRequestError):
            client.containers.create(name="b", expires_after=bad)


def test_idle_container_expires_and_is_gone(env):
    client, _, tmp = env
    c = client.containers.create(name="e")
    mgr = cont.get_manager()
    meta = mgr._meta(c.id)
    import json

    d = json.loads(meta.read_text())
    d["last_active"] -= 3600
    meta.write_text(json.dumps(d))
    with pytest.raises(openai.NotFoundError):
        client.containers.retrieve(c.id)
    assert not (tmp / "sbx" / c.id).exists()


def test_activity_refreshes_last_active_not_creation(env):
    client, _, _ = env
    c = client.containers.create(name="t")
    mgr = cont.get_manager()
    import json

    meta = mgr._meta(c.id)
    d = json.loads(meta.read_text())
    d["last_active"] -= 600
    meta.write_text(json.dumps(d))
    r = client.containers.retrieve(c.id)
    assert (
        r.last_active_at >= c.last_active_at + 599 - 5
        or r.last_active_at >= c.created_at
    )
    assert r.created_at == c.created_at


def test_memory_and_network_policy_only_narrow(env, monkeypatch):
    client, _, _ = env
    monkeypatch.setenv("YUNSHU_CODE_EXECUTION_MEMORY_MB", "6000")
    big = client.containers.create(name="m", memory_limit="16g")
    assert big.memory_limit == "4g"  # the server's setting is the ceiling
    with pytest.raises(openai.BadRequestError):
        client.containers.create(name="m", memory_limit="2g")  # type: ignore[arg-type]
    # no server allowlist: any allowlist request is refused, never silently widened
    with pytest.raises(
        openai.BadRequestError, match="YUNSHU_CODE_EXECUTION_NETWORK_ALLOW"
    ):
        client.containers.create(
            name="n",
            network_policy={"type": "allowlist", "allowed_domains": ["example.com"]},
        )
    monkeypatch.setenv("YUNSHU_CODE_EXECUTION_NETWORK_ALLOW", "example.com,*.pypi.org")
    ok = client.containers.create(
        name="n",
        network_policy={
            "type": "allowlist",
            "allowed_domains": ["example.com", "files.pypi.org"],
        },
    )
    assert ok.network_policy.type == "allowlist"
    assert ok.network_policy.allowed_domains == ["example.com", "files.pypi.org"]
    with pytest.raises(openai.BadRequestError):
        client.containers.create(
            name="n",
            network_policy={"type": "allowlist", "allowed_domains": ["evil.test"]},
        )
    inherit = client.containers.create(name="i")
    assert inherit.network_policy.allowed_domains == ["example.com", "*.pypi.org"]
    off = client.containers.create(name="o", network_policy={"type": "disabled"})
    assert off.network_policy.type == "disabled"
    # the operator removing a host later removes it from every container
    monkeypatch.setenv("YUNSHU_CODE_EXECUTION_NETWORK_ALLOW", "example.com")
    again = client.containers.retrieve(ok.id)
    assert again.network_policy.allowed_domains == ["example.com"]


def test_name_is_required_and_skills_refused(env):
    client, http, _ = env
    r = http.post("/v1/containers", json={})
    assert r.status_code == 400 and r.json()["error"]["param"] == "name"
    r = http.post(
        "/v1/containers", json={"name": "x", "skills": [{"type": "skill_reference"}]}
    )
    assert r.status_code == 400


def test_files_upload_list_retrieve_content_delete(env):
    client, _, tmp = env
    c = client.containers.create(name="f")
    f = client.containers.files.create(c.id, file=("data.csv", b"a,b\n1,2\n"))
    assert f.id.startswith("cfile_") and f.container_id == c.id
    assert f.path == "/mnt/data/data.csv" and f.source == "user" and f.bytes == 8
    assert client.containers.files.retrieve(f.id, container_id=c.id).path == f.path
    assert (
        client.containers.files.content.retrieve(f.id, container_id=c.id).read()
        == b"a,b\n1,2\n"
    )
    # a file the sandbox made is listed as the assistant's
    (Path(tmp / "sbx" / c.id / "work" / "out.txt")).write_text("made")
    rows = client.containers.files.list(c.id).data
    assert {r.path: r.source for r in rows} == {
        "/mnt/data/data.csv": "user",
        "/mnt/data/out.txt": "assistant",
    }
    assert client.containers.files.delete(f.id, container_id=c.id) is None
    with pytest.raises(openai.NotFoundError):
        client.containers.files.retrieve(f.id, container_id=c.id)
    assert not (tmp / "sbx" / c.id / "work" / "data.csv").exists()


def test_file_from_files_api_id_and_create_with_file_ids(env):
    client, _, _ = env
    up = client.files.create  # not mounted on this app: use the store directly
    del up
    meta = files_store.get_store().put(b"hello", "hello.txt", "user_data")
    c = client.containers.create(name="g", file_ids=[meta["id"]])
    rows = client.containers.files.list(c.id).data
    assert [r.path for r in rows] == ["/mnt/data/hello.txt"]
    again = client.containers.files.create(c.id, file_id=meta["id"])
    assert again.path == "/mnt/data/hello.txt"
    with pytest.raises(openai.NotFoundError):
        client.containers.create(name="g", file_ids=["file-nope"])


def test_file_ids_are_stable_and_per_container(env):
    client, _, _ = env
    a = client.containers.create(name="a")
    b = client.containers.create(name="b")
    fa = client.containers.files.create(a.id, file=("x.txt", b"1"))
    fb = client.containers.files.create(b.id, file=("x.txt", b"2"))
    assert fa.id != fb.id
    again = client.containers.files.create(a.id, file=("x.txt", b"11"))
    assert again.id == fa.id and again.bytes == 2
    with pytest.raises(openai.NotFoundError):  # another container's file id
        client.containers.files.retrieve(fb.id, container_id=a.id)


@pytest.mark.parametrize(
    "name",
    ["../escape.txt", "/etc/passwd", "a/../../b", "a\\b", "..", "a//b", "." * 1],
)
def test_hostile_file_names_never_leave_the_container(env, name):
    client, http, tmp = env
    c = client.containers.create(name="h")
    r = http.post(
        f"/v1/containers/{c.id}/files", files={"file": (name, b"x", "text/plain")}
    )
    if name == "/etc/passwd":
        # an absolute path is read as /mnt/data-relative, inside the container
        assert r.status_code == 200 and r.json()["path"] == "/mnt/data/etc/passwd"
    else:
        assert r.status_code == 400
    for p in (tmp / "sbx").rglob("*"):
        assert str(p).startswith(str(tmp / "sbx"))
    assert (
        not (tmp / "escape.txt").exists() and not (tmp / "sbx" / "escape.txt").exists()
    )


def test_symlink_planted_by_sandboxed_code_is_not_followed(env):
    client, http, tmp = env
    c = client.containers.create(name="s")
    work = tmp / "sbx" / c.id / "work"
    target = tmp / "host-secret"
    target.write_text("SECRET")
    os.symlink(target, work / "leak.txt")  # as code in the sandbox could
    os.symlink(tmp, work / "dir")
    names = [r.path for r in client.containers.files.list(c.id).data]
    assert names == []  # links are not files of the container
    r = http.post(
        f"/v1/containers/{c.id}/files", files={"file": ("leak.txt", b"X", "text/plain")}
    )
    assert r.status_code == 400 and target.read_text() == "SECRET"
    r = http.post(
        f"/v1/containers/{c.id}/files",
        files={"file": ("dir/new.txt", b"X", "text/plain")},
    )
    assert r.status_code == 400 and not (tmp / "new.txt").exists()
    # a content request for the id a link path would have must not read through it
    from yunshu_gateway.sandbox import files as cf

    fid = cf.file_id(c.id, "leak.txt")
    assert http.get(f"/v1/containers/{c.id}/files/{fid}/content").status_code == 404


def test_content_disposition_cannot_be_injected(env):
    client, http, _ = env
    c = client.containers.create(name="d")
    f = client.containers.files.create(c.id, file=('a"\r\nX-Evil: 1.txt', b"x"))
    r = http.get(f"/v1/containers/{c.id}/files/{f.id}/content")
    assert r.status_code == 200 and "x-evil" not in {k.lower() for k in r.headers}


def test_cross_owner_and_prefix_isolation(env, monkeypatch):
    client, http, _ = env
    mgr = cont.get_manager()
    anth = mgr.create()  # an Anthropic-made container is not visible as an OpenAI one
    assert http.get(f"/v1/containers/{anth.id}").status_code == 404
    assert [c.id for c in client.containers.list().data] == []
    monkeypatch.setattr(containers_router, "_actor", lambda r: ("alice", False))
    mine = client.containers.create(name="mine")
    monkeypatch.setattr(containers_router, "_actor", lambda r: ("bob", False))
    assert client.containers.list().data == []
    with pytest.raises(openai.NotFoundError):
        client.containers.retrieve(mine.id)
    monkeypatch.setattr(containers_router, "_actor", lambda r: ("bob", True))
    assert client.containers.retrieve(mine.id).id == mine.id


def test_unknown_and_malformed_ids_404(env):
    client, http, _ = env
    for cid in ("cntr_" + "0" * 24, "nope", "container_" + "0" * 24):
        assert http.get(f"/v1/containers/{cid}").status_code == 404
        assert http.get(f"/v1/containers/{cid}/files").status_code == 404
