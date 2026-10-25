"""Host-path safety of the code execution containers: names from the Files API and the editor never
reach outside the container, whatever links the sandboxed code planted (CPU only)."""

from __future__ import annotations

import asyncio
import os

import pytest

from yunshu_engine import settings
from yunshu_gateway import files_store
from yunshu_gateway.sandbox import containers as cont
from yunshu_gateway.sandbox import files as cf
from yunshu_gateway.sandbox.executor import CodeExecutor
from yunshu_gateway.sandbox.srt import host_allowed


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("YUNSHU_FILES_DIR", str(tmp_path / "files"))
    files_store.reset_store()
    m = cont.ContainerManager(tmp_path / "sbx", ttl_s=600)
    return CodeExecutor(m), m.create(), m, tmp_path


@pytest.mark.parametrize(
    "name,ok",
    [
        ("a.txt", "a.txt"),
        ("dir/a.txt", "dir/a.txt"),
        ("/mnt/data/a.txt", "a.txt"),
        ("/mnt/data/x/y.csv", "x/y.csv"),
        ("/etc/passwd", "etc/passwd"),
        ("..", None),
        ("../a", None),
        ("a/../../b", None),
        ("a//b", None),
        ("a\\b", None),
        ("a\0b", None),
        ("", None),
        (None, None),
        ("a/" * 9 + "b", None),
        ("x" * 300, None),
    ],
)
def test_clean_relpath(name, ok):
    assert cf.clean_relpath(name) == ok


def test_file_ops_refuse_links_anywhere_on_the_path(env):
    _, c, _, tmp = env
    outside = tmp / "outside"
    outside.mkdir()
    (outside / "secret").write_text("S")
    os.symlink(outside, c.work / "dir")
    os.symlink(outside / "secret", c.work / "leak")
    with pytest.raises(cf.FileOpError):
        cf.write_file(c, "dir/new", b"x")
    with pytest.raises(cf.FileOpError):
        cf.write_file(c, "leak", b"x")
    with pytest.raises(cf.FileOpError):
        cf.read_file(c, "dir/secret")
    with pytest.raises(cf.FileOpError):
        cf.read_file(c, "leak")
    assert (outside / "secret").read_text() == "S" and not (outside / "new").exists()
    # deleting a link removes the link only
    cf.delete_file(c, "leak")
    assert not (c.work / "leak").is_symlink() and (outside / "secret").exists()


def test_work_dir_replaced_by_a_link_is_refused(env):
    _, c, _, tmp = env
    outside = tmp / "elsewhere"
    outside.mkdir()
    (outside / "f").write_text("F")
    os.rename(c.work, c.path / "work.bak")
    os.symlink(outside, c.work)
    with pytest.raises(cf.FileOpError):
        cf.read_file(c, "f")
    assert cf.scan(c) == []


def test_files_api_name_with_shell_metacharacters_is_just_a_name(env):
    e, c, m, tmp = env
    store = files_store.get_store()
    for nm in (
        "a;touch pwned.txt",
        "$(touch pwned2)",
        "`id`.txt",
        "-rf",
        "a b'c\"d.txt",
    ):
        meta = store.put(b"x", nm, "user_data")
        rel = e.stage_file(c, meta["id"])
        assert (c.work / rel).read_bytes() == b"x"
    assert not list(tmp.rglob("pwned*"))
    assert all(p.parent == c.work for p in c.work.iterdir())


def test_files_api_name_with_path_parts_stays_inside(env):
    e, c, _, tmp = env
    meta = files_store.get_store().put(b"x", "../../escape.txt", "user_data")
    rel = e.stage_file(c, meta["id"])
    assert rel == "escape.txt" and not (tmp / "escape.txt").exists()


def test_staging_over_a_planted_link_does_not_write_through(env):
    e, c, _, tmp = env
    target = tmp / "victim.txt"
    target.write_text("V")
    os.symlink(target, c.work / "up.txt")
    meta = files_store.get_store().put(b"NEW", "up.txt", "user_data")
    from yunshu_gateway.files_store import FileStoreError

    with pytest.raises(FileStoreError):
        e.stage_file(c, meta["id"])
    assert target.read_text() == "V"


def test_editor_does_not_follow_planted_links(env):
    e, c, _, tmp = env
    target = tmp / "host-secret"
    target.write_text("SECRET")
    os.symlink(target, c.work / "l")
    r = asyncio.run(e.editor(c, {"command": "view", "path": "l"}))
    assert not r.ok and "SECRET" not in r.text
    r = asyncio.run(e.editor(c, {"command": "create", "path": "l", "file_text": "X"}))
    assert not r.ok and target.read_text() == "SECRET"
    r = asyncio.run(
        e.editor(c, {"command": "view", "path": str(target)})
    )  # an absolute host path
    assert not r.ok


def test_container_ids_cannot_be_path_traversal(env):
    _, _, m, tmp = env
    for bad in ("../x", "container_" + "../" * 3, "cntr_" + "0" * 23, "x/y"):
        assert not cont.valid_id(bad)
        m.delete(bad)  # a no-op, never a rmtree outside
    assert tmp.exists()


def test_container_network_only_narrows_the_server_list(env, monkeypatch):
    e, _, m, _ = env
    monkeypatch.setenv("YUNSHU_CODE_EXECUTION_NETWORK_ALLOW", "a.com,*.b.org")
    wide = m.create(
        network_allow=["a.com", "evil.com", "x.b.org", "*.b.org", "c.b.org:443"]
    )
    assert e.network_allow(wide) == ["a.com", "x.b.org", "*.b.org", "c.b.org:443"]
    none = m.create(network_allow=[])
    assert e.network_allow(none) == []
    inherit = m.create()
    assert e.network_allow(inherit) == ["a.com", "*.b.org"]
    monkeypatch.setenv("YUNSHU_CODE_EXECUTION_NETWORK_ALLOW", "")
    assert e.network_allow(wide) == [] and e.network_allow(inherit) == []


def test_host_allowed_matching():
    assert host_allowed("a.com", ["a.com"])
    assert host_allowed("x.a.com", ["*.a.com"])
    assert not host_allowed("a.com", ["*.a.com"])
    assert not host_allowed("evila.com", ["*.a.com"])
    assert not host_allowed("a.com.evil", ["a.com"])
    assert not host_allowed("", ["a.com"])
    assert host_allowed("a.com:443", ["a.com:443"])
    assert not host_allowed("a.com:80", ["a.com:443"])


def test_memory_limit_is_a_ceiling(env, monkeypatch):
    e, _, m, _ = env
    monkeypatch.setenv("YUNSHU_CODE_EXECUTION_MEMORY_MB", "3000")
    assert e.memory_bytes(m.create(memory_limit="1g")) == 1024 * 2**20
    assert e.memory_bytes(m.create(memory_limit="16g")) == 3000 * 2**20
    assert e.memory_bytes(m.create()) == 3000 * 2**20
    assert settings.get("YUNSHU_CODE_EXECUTION_MEMORY_MB") == 3000


def test_container_ttl_is_per_container_and_capped_by_the_setting(env, monkeypatch):
    _, _, m, _ = env
    a = m.create(ttl_s=60)
    b = m.create()
    assert a.ttl_s == 60 and b.ttl_s == 600
    m.ttl_s = 30  # the operator lowers YUNSHU_CODE_EXECUTION_TTL_MIN
    assert m.get(a.id, touch=False).ttl_s == 30 and m.get(b.id, touch=False).ttl_s == 30
