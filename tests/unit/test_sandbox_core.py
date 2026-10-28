"""Sandbox core without srt: settings builder, supervisor limits, container lifecycle, editor."""

import asyncio
import os
import sys
import time

import pytest

from yunshu_engine.paths import private_media_path, private_media_paths
from yunshu_gateway.sandbox import containers as cont
from yunshu_gateway.sandbox.executor import CodeExecutor, ExecResult, model_text
from yunshu_gateway.sandbox.srt import (
    DENY_READ_ROOTS,
    PRIVATE_RANGES,
    SrtInfo,
    build_settings,
    parse_allowlist,
)
from yunshu_gateway.sandbox.supervisor import run_supervised


def run(coro):
    return asyncio.run(coro)


ENV = {"PATH": "/usr/bin:/bin"}


def sup(argv, **kw):
    kw.setdefault("timeout", 10)
    kw.setdefault("mem_bytes", 1 << 30)
    return run(run_supervised(argv, env=ENV, cwd="/tmp", **kw))


# ── settings ────────────────────────────────────────────────────────────────
def test_settings_default_is_closed(tmp_path):
    s = build_settings(tmp_path / "c")
    assert s["network"]["allowedDomains"] == []
    assert s["network"]["deniedDomains"] == []
    assert s["network"]["allowLocalBinding"] is False
    assert s["network"]["deniedResolvedAddresses"] == PRIVATE_RANGES
    fs = s["filesystem"]
    assert fs["allowWrite"] == [str(tmp_path / "c")]
    assert (
        fs["denyRead"] == [*DENY_READ_ROOTS, str(tmp_path), *private_media_paths()]
        and "/Users" in fs["denyRead"]
    )
    assert str(tmp_path / "c") in fs["allowRead"]
    assert sys.base_prefix in fs["allowRead"]


def test_settings_allowlist(tmp_path):
    assert parse_allowlist(" a.com, *.b.org:443 ,,") == ["a.com", "*.b.org:443"]
    assert parse_allowlist(None) == []
    s = build_settings(tmp_path, network_allow=["a.com"])
    assert s["network"]["allowedDomains"] == ["a.com"]


def test_srt_argv_is_one_command_string():
    info = SrtInfo("/n/node", "/s/cli.js")
    a = info.argv(
        __import__("pathlib").Path("/x.json"), ["/bin/bash", "-c", "echo 'a b'"]
    )
    assert a[:5] == ["/n/node", "/s/cli.js", "--settings", "/x.json", "-c"]
    import shlex

    assert shlex.split(a[5]) == ["/bin/bash", "-c", "echo 'a b'"]


# ── supervisor ──────────────────────────────────────────────────────────────
def test_supervisor_ok_and_exit_code():
    r = sup(["/bin/sh", "-c", "echo out; echo err >&2; exit 3"])
    assert (r.stdout, r.stderr, r.return_code) == (b"out\n", b"err\n", 3)
    assert not (r.timed_out or r.oom or r.output_overflow)


def test_supervisor_wall_timeout_kills_tree():
    t0 = time.monotonic()
    r = sup(["/bin/sh", "-c", "sleep 30 & sleep 30"], timeout=1)
    assert r.timed_out and time.monotonic() - t0 < 6


def test_supervisor_output_truncated_and_overflow():
    r = sup(["/bin/sh", "-c", "yes x | head -c 200000"], per_stream=1000)
    assert len(r.stdout) == 1000 and r.stdout_truncated and not r.output_overflow
    r = sup(["/bin/sh", "-c", "yes x"], per_stream=1000, max_total=50_000)
    assert r.output_overflow and len(r.stdout) == 1000


def test_supervisor_memory_limit():
    code = "b = bytearray(400*1024*1024)\nimport time; time.sleep(20)"
    r = sup([sys.executable, "-c", code], mem_bytes=150 * 1024 * 1024, timeout=20)
    assert r.oom and not r.timed_out


def test_supervisor_leaves_no_survivors(tmp_path):
    marker = tmp_path / "alive"
    sup(["/bin/sh", "-c", f"(sleep 1; touch {marker}) & exit 0"])
    time.sleep(1.6)
    assert not marker.exists()


# ── containers ──────────────────────────────────────────────────────────────
@pytest.fixture
def mgr(tmp_path):
    now = [1000.0]
    m = cont.ContainerManager(tmp_path / "root", ttl_s=60, clock=lambda: now[0])
    m.now = now
    return m


def test_container_create_get_touch_expire(mgr):
    c = mgr.create()
    assert cont.valid_id(c.id) and c.work.is_dir() and c.out.is_dir()
    assert oct(c.path.stat().st_mode & 0o777) == "0o700"
    mgr.now[0] += 50
    c2 = mgr.get(c.id)  # touch resets the timer
    assert c2.expires_at == 1050 + 60
    mgr.now[0] += 59
    mgr.get(c.id)
    mgr.now[0] += 61
    with pytest.raises(cont.ContainerError) as e:
        mgr.get(c.id)
    assert e.value.code == "expired" and not c.path.exists()


def test_container_unknown_and_bad_ids(mgr):
    with pytest.raises(cont.ContainerError) as e:
        mgr.get("container_" + "0" * 24)
    assert e.value.code == "not_found"
    for bad in ("../x", "container_zz", "", "container_" + "0" * 23 + "/"):
        with pytest.raises(cont.ContainerError) as e:
            mgr.get(bad)
        assert e.value.code == "invalid"


def test_container_gc_orphans_and_capacity(mgr):
    a, b = mgr.create(), mgr.create()
    mgr.now[0] += 1
    (a.work / "big").write_bytes(b"x" * 5000)
    (b.work / "big").write_bytes(b"x" * 5000)
    mgr.get(b.id)  # b is now the most recently used
    mgr.max_total_bytes = 6000
    assert mgr.gc() == 1
    assert not a.path.exists() and b.path.exists()
    # a directory without bookkeeping (crash leftover) is removed
    orphan = mgr.root / ("container_" + "a" * 24)
    orphan.mkdir()
    assert mgr.gc() == 1 and not orphan.exists()


# ── text editor (in-process, confined) ──────────────────────────────────────
@pytest.fixture
def ex(mgr):
    return CodeExecutor(mgr), mgr.create()


def ed(ex, **a):
    e, c = ex
    return run(e.editor(c, a))


def test_editor_create_view_replace(ex):
    r = ed(ex, command="create", path="a/b.txt", file_text="one\ntwo\nthree\n")
    assert (
        r.ok
        and r.data["type"].endswith("create_result")
        and r.data["is_file_update"] is False
    )
    assert (
        ed(ex, command="create", path="a/b.txt", file_text="x\n").data["is_file_update"]
        is True
    )
    ed(ex, command="create", path="a/b.txt", file_text="one\ntwo\nthree\n")
    v = ed(ex, command="view", path="a/b.txt")
    assert v.data["content"] == "one\ntwo\nthree" and v.data["total_lines"] == 3
    v = ed(ex, command="view", path="a/b.txt", view_range=[2, 3])
    assert v.data["content"] == "two\nthree" and v.data["start_line"] == 2
    s = ed(ex, command="str_replace", path="a/b.txt", old_str="two", new_str="2")
    assert s.data["old_start"] == 2 and s.data["lines"] == ["-two", "+2"]
    assert ed(ex, command="view", path="a/b.txt").data["content"].splitlines()[1] == "2"
    assert (
        ed(
            ex, command="str_replace", path="a/b.txt", old_str="zzz", new_str=""
        ).error_code
        == "invalid_tool_input"
    )
    assert ed(ex, command="view", path="nope").error_code == "file_not_found"
    assert (
        ed(ex, command="insert", path="a", insert_line=1).error_code
        == "invalid_tool_input"
    )


def test_editor_cannot_escape(ex, tmp_path):
    e, c = ex
    secret = tmp_path / "secret.txt"
    secret.write_text("s")
    assert ed(ex, command="view", path="../../../../" + str(secret)).error_code in (
        "file_not_found",
        "invalid_tool_input",
    )
    # an absolute path outside the container is re-rooted into work/, never read from the host
    r = ed(ex, command="view", path=str(secret))
    assert not r.ok
    # symlink out of the container is refused
    os.symlink(secret, c.work / "link")
    assert ed(ex, command="view", path="link").error_code == "invalid_tool_input"
    assert (
        ed(ex, command="create", path="../../evil", file_text="x").error_code
        == "invalid_tool_input"
    )
    assert not (c.path.parent / "evil").exists()


def test_model_text_shapes():
    assert model_text(ExecResult(stdout="hi\n", return_code=0)).startswith(
        "exit code: 0\nstdout:\nhi"
    )
    assert "execution_time_exceeded" in model_text(
        ExecResult(False, "execution_time_exceeded")
    )


# ── doctor ──────────────────────────────────────────────────────────────────
def test_doctor_check_states(monkeypatch):
    import importlib

    import yunshu_gateway.sandbox.srt as srt_mod

    doctor = importlib.import_module("yunshu_cli.doctor")
    from yunshu_engine import settings as st

    monkeypatch.setattr(srt_mod, "find_srt", lambda: None)
    real = st.get
    monkeypatch.setattr(
        st,
        "get",
        lambda n, *a, **k: False if n == "YUNSHU_CODE_EXECUTION" else real(n, *a, **k),
    )
    c = doctor.check_code_execution()
    assert c.status == "ok" and "not installed" in c.detail
    monkeypatch.setattr(
        st,
        "get",
        lambda n, *a, **k: True if n == "YUNSHU_CODE_EXECUTION" else real(n, *a, **k),
    )
    c = doctor.check_code_execution()
    assert c.status == "fail" and "503" in c.detail and "sandbox-runtime" in c.fix
    monkeypatch.setattr(
        srt_mod, "find_srt", lambda: SrtInfo("/n", "/c", problem="Node 20.1 is too old")
    )
    assert "too old" in doctor.check_code_execution().detail


def test_decoded_request_media_sit_in_a_private_dir_the_sandbox_cannot_read(
    tmp_path, monkeypatch
):
    """Decoded base64 media used to land in the shared system temp dir, readable by sandboxed code."""
    import stat

    from yunshu_engine.vlm_engine import VLMEngine

    monkeypatch.setattr("tempfile.tempdir", str(tmp_path))
    eng = VLMEngine.__new__(VLMEngine)
    eng._temp_files = None
    import threading

    eng._temp_files_lock = threading.Lock()
    path = asyncio.run(eng._save_base64_file("aGVsbG8=", "png"))
    assert os.path.dirname(path) == private_media_path()
    assert stat.S_IMODE(os.stat(private_media_path()).st_mode) == 0o700
    assert (
        private_media_path() in build_settings(tmp_path / "c")["filesystem"]["denyRead"]
    )


def test_private_media_deny_names_the_symlink_and_its_real_path(tmp_path, monkeypatch):
    """macOS TMPDIR lives under /var/folders, a symlink to /private/var/folders."""
    import tempfile

    real_tmp = tmp_path / "private_var"
    real_tmp.mkdir()
    link = tmp_path / "var"
    link.symlink_to(real_tmp)
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(link))
    given = os.path.join(str(link), f"yunshu_decoded_{os.getuid()}")
    real = os.path.join(os.path.realpath(str(link)), f"yunshu_decoded_{os.getuid()}")
    assert given != real
    deny = build_settings(tmp_path / "c")["filesystem"]["denyRead"]
    assert given in deny and real in deny
