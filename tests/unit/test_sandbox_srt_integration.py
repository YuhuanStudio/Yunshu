"""Executor against the real srt (skipped when srt or Node >= 22.12 is absent)."""

import asyncio
import os
from pathlib import Path

import pytest

from yunshu_engine import settings
from yunshu_gateway import files_store
from yunshu_gateway.sandbox import containers as cont
from yunshu_gateway.sandbox.executor import CodeExecutor
from yunshu_gateway.sandbox.srt import find_srt

_DEV = Path(
    os.environ.get(
        "YUNSHU_TEST_SRT_BIN", "/Volumes/P5Plus/yunshu-build/srt/node_modules/.bin"
    )
)


@pytest.fixture(autouse=True)
def srt_on_path(monkeypatch):
    if (_DEV / "srt").exists():
        monkeypatch.setenv("PATH", f"{_DEV}:/opt/homebrew/bin:" + os.environ["PATH"])
    info = find_srt()
    if info is None or info.problem:
        pytest.skip("srt is not available")


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("YUNSHU_FILES_DIR", str(tmp_path / "files"))
    files_store.reset_store()
    m = cont.ContainerManager(tmp_path / "sbx", ttl_s=600)
    return CodeExecutor(m), m.create(), tmp_path


def bash(env, cmd, **kw):
    e, c, _ = env
    return asyncio.run(e.bash(c, cmd, **kw))


def test_write_read_python_and_container_reuse(env):
    r = bash(env, "echo hi > a.txt && cat a.txt && python3 -c 'print(6*7)'")
    assert r.ok and r.return_code == 0 and r.stdout.split() == ["hi", "42"]
    assert bash(env, "cat a.txt").stdout.strip() == "hi"  # same container keeps files


def test_nonzero_exit_is_a_normal_result(env):
    r = bash(env, "echo oops >&2; exit 4")
    assert r.ok and r.return_code == 4 and "oops" in r.stderr


def test_network_blocked(env):
    r = bash(
        env, "curl -s -m 8 -o /dev/null -w '%{http_code}' https://example.com; echo"
    )
    assert r.ok and r.stdout.strip().startswith("000")


def test_cannot_read_home_or_write_outside(env):
    home = Path.home()
    r = bash(env, f"ls {home} 2>&1; echo x > {home}/yunshu_probe 2>&1")
    assert "not permitted" in r.stdout.lower() or "not permitted" in r.stderr.lower()
    assert not (home / "yunshu_probe").exists()
    # another container is not readable either
    e, c, _ = env
    other = e.manager.create()
    (other.work / "secret").write_text("s")
    r = bash(env, f"cat {other.work}/secret")
    assert r.return_code != 0 and "s" not in r.stdout


def test_server_environment_not_inherited(env, monkeypatch):
    monkeypatch.setenv("YUNSHU_AUTH_TOKEN", "topsecret")
    r = bash(env, "env")
    assert "topsecret" not in r.stdout and "YUNSHU" not in r.stdout
    assert f"HOME={env[1].home}" in r.stdout


def test_output_dir_capture(env):
    r = bash(
        env,
        'echo data > "$OUTPUT_DIR/r.csv"; mkdir "$OUTPUT_DIR/sub"; echo x > "$OUTPUT_DIR/sub/n"',
    )
    assert [f["filename"] for f in r.files] == ["r.csv"]
    assert files_store.get_store().read(r.files[0]["file_id"]) == b"data\n"
    assert not list(env[1].out.iterdir())  # per-call dir is removed


def test_timeout_and_memory(env, monkeypatch):
    monkeypatch.setattr(
        settings, "get", _patched({"YUNSHU_CODE_EXECUTION_TIMEOUT": 2.0})
    )
    r = bash(env, "sleep 30")
    assert not r.ok and r.error_code == "execution_time_exceeded"
    monkeypatch.setattr(
        settings, "get", _patched({"YUNSHU_CODE_EXECUTION_MEMORY_MB": 100})
    )
    r = bash(
        env, "python3 -c 'b=bytearray(600*1024*1024); import time; time.sleep(20)'"
    )
    assert not r.ok and r.error_code == "unavailable"


def test_output_truncation(env):
    r = bash(env, "python3 -c 'print(\"x\"*300000)'")
    assert r.ok and len(r.stdout) < 70_000 and "[output truncated]" in r.stdout


def test_background_process_does_not_survive(env):
    bash(env, f"(sleep 1.5; touch {env[1].work}/alive) & echo started")
    import time

    time.sleep(2.5)
    assert not (env[1].work / "alive").exists()


_orig = settings.get


def _patched(over):
    return lambda name, *a, **k: over[name] if name in over else _orig(name, *a, **k)
