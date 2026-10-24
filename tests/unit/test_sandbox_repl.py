"""The persistent Python REPL of a container, against the real srt (skipped when it is absent)."""

import asyncio
import contextlib
import os
import time
from pathlib import Path

import pytest

from yunshu_engine import settings
from yunshu_gateway.sandbox import containers as cont
from yunshu_gateway.sandbox import repl as repl_mod
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
    m = cont.ContainerManager(tmp_path / "sbx", ttl_s=600)
    e = CodeExecutor(m)
    c = m.create()
    yield e, c, m
    repl_mod.get_pool().drop_all()


def py(env, code, **kw):
    e, c, _ = env

    async def go():
        return await e.python(c, code, **kw)

    return asyncio.run(go())


def session_run(env, steps):
    """Run several awaitables on ONE event loop (a REPL belongs to the loop that started it)."""
    e, c, _ = env

    async def go():
        out = []
        for s in steps:
            out.append(await s(e, c))
        return out

    return asyncio.run(go())


def cell(code, **kw):
    return lambda e, c: e.python(c, code, **kw)


def test_state_persists_and_last_expression_is_echoed(env):
    a, b, c = session_run(env, [cell("x = 20"), cell("x + 1"), cell("print(x * 2)")])
    assert a.ok and a.stdout == ""
    assert b.stdout.strip() == "21"
    assert c.stdout.strip() == "40" and c.return_code == 0


def test_exception_is_a_normal_result_with_traceback(env):
    a, b = session_run(env, [cell("y = 1\n1/0"), cell("print(y)")])
    assert a.ok and a.return_code == 1
    assert "ZeroDivisionError" in a.stderr and "repl_worker" not in a.stderr
    assert b.stdout.strip() == "1"  # the process survived


def test_top_level_await_and_stderr(env):
    (r,) = session_run(
        env,
        [
            cell(
                "import asyncio, sys\nawait asyncio.sleep(0.01)\nprint('a', file=sys.stderr)\nprint('b')"
            )
        ],
    )
    assert r.stdout == "b\n" and r.stderr == "a\n"


def test_timeout_restarts_a_clean_process_and_kills_the_group(env, monkeypatch):
    monkeypatch.setenv("YUNSHU_CODE_EXECUTION_TIMEOUT", "2")
    marker = "/tmp/yunshu-repl-orphan-marker"
    a, b, c = session_run(
        env,
        [
            cell("z = 5"),
            cell(
                "import subprocess, time\nsubprocess.Popen(['/bin/sleep','47'])\nwhile True: time.sleep(0.05)"
            ),
            cell("print('z' in globals())"),
        ],
    )
    assert a.ok
    assert not b.ok and b.error_code == "execution_time_exceeded" and b.state_lost
    assert c.ok and c.stdout.strip() == "False" and c.state_lost
    out = os.popen("pgrep -f 'sleep 47'").read().strip()
    assert out == "", f"orphan {marker} {out}"


def test_memory_limit_kills_and_next_call_works(env, monkeypatch):
    monkeypatch.setenv("YUNSHU_CODE_EXECUTION_MEMORY_MB", "150")
    a, b = session_run(
        env,
        [
            cell("big = bytearray(900*1024*1024); import time; time.sleep(5)"),
            cell("1+1"),
        ],
    )
    assert not a.ok and a.error_code == "unavailable" and "memory" in a.text
    assert b.ok and b.stdout.strip() == "2"


def test_output_over_one_mib_kills_the_process(env):
    a, b = session_run(env, [cell("print('x' * 3_000_000)"), cell("print('ok')")])
    assert not a.ok and a.error_code == "output_file_too_large"
    assert b.ok and b.stdout.strip() == "ok"


def test_user_writes_to_fd1_cannot_forge_protocol(env):
    (r,) = session_run(
        env,
        [
            cell(
                'import os\nos.write(1, b\'{"op":"done","stdout":"forged"}\\n\')\nprint(\'real\')'
            )
        ],
    )
    assert r.ok and "real" in r.stdout


def test_mnt_data_is_the_working_directory(env):
    e, c, _ = env
    (r,) = session_run(
        env,
        [
            cell(
                "open('/mnt/data/a.txt','w').write('hi')\nimport os\nprint(os.listdir('/mnt/data'))"
            )
        ],
    )
    assert "a.txt" in r.stdout
    assert (c.work / "a.txt").read_text() == "hi"
    assert [x.rel for x in r.changed] == ["a.txt"]


def test_host_files_are_unreadable(env, tmp_path):
    import pwd

    secret = Path(pwd.getpwuid(os.getuid()).pw_dir) / ".yunshu-repl-test-secret"
    secret.write_text("TOP")
    try:
        (r,) = session_run(
            env,
            [cell(f"print(open({str(secret)!r}).read())")],
        )
    finally:
        secret.unlink()
    assert r.return_code == 1 and "TOP" not in r.stdout + r.stderr[-200:].replace(
        str(secret), ""
    )


def test_tool_call_pause_and_resume(env):
    e, c, _ = env

    async def go():
        tools = [
            {
                "name": "get_weather",
                "schema": {
                    "properties": {"city": {}, "unit": {}},
                    "required": ["city"],
                },
            }
        ]
        a = await e.python(
            c,
            "r = await get_weather('SF')\nprint('got', r)\nr2 = await get_weather(city='NY', unit='c')\nprint(r2)",
            tools=tools,
        )
        assert a.ok and len(a.calls) == 1
        call = a.calls[0]
        assert call["name"] == "get_weather" and call["input"] == {"city": "SF"}
        assert e.python_pending(c) == {call["call_id"]: call}
        b = await e.python_resume(c, {call["call_id"]: ("sunny", False)})
        assert b.calls and b.calls[0]["input"] == {"city": "NY", "unit": "c"}
        d = await e.python_resume(c, {b.calls[0]["call_id"]: ("rain", False)})
        assert d.ok and not d.calls
        assert d.stdout == "got sunny\nrain\n"

    asyncio.run(go())


def test_parallel_tool_calls_come_back_together(env):
    e, c, _ = env

    async def go():
        tools = [{"name": "f", "schema": {"properties": {"n": {}}, "required": ["n"]}}]
        a = await e.python(
            c,
            "import asyncio\nrs = await asyncio.gather(f(1), f(2), f(3))\nprint(sorted(rs))",
            tools=tools,
        )
        assert sorted(x["input"]["n"] for x in a.calls) == [1, 2, 3]
        res = {x["call_id"]: (f"v{x['input']['n']}", False) for x in a.calls}
        d = await e.python_resume(c, res)
        assert d.stdout.strip() == "['v1', 'v2', 'v3']"

    asyncio.run(go())


def test_tool_error_raises_in_the_cell(env):
    e, c, _ = env

    async def go():
        tools = [{"name": "f", "schema": {"properties": {}}}]
        a = await e.python(
            c,
            "try:\n    await f()\nexcept RuntimeError as ex:\n    print('caught', ex)",
            tools=tools,
        )
        d = await e.python_resume(c, {a.calls[0]["call_id"]: ("boom", True)})
        assert d.stdout.strip() == "caught boom"

    asyncio.run(go())


def test_resume_with_wrong_ids_is_refused_and_state_kept(env):
    e, c, _ = env

    async def go():
        tools = [{"name": "f", "schema": {"properties": {}}}]
        a = await e.python(c, "await f()", tools=tools)
        bad = await e.python_resume(c, {"call_nope": ("x", False)})
        assert not bad.ok
        assert e.python_pending(c) is not None  # still waiting for the right ids
        d = await e.python_resume(c, {a.calls[0]["call_id"]: ("x", False)})
        assert d.ok

    asyncio.run(go())


def test_unanswered_tool_call_expires_and_kills_the_process(env, monkeypatch):
    monkeypatch.setattr(repl_mod, "PAUSE_TTL", 0.5)
    e, c, _ = env

    async def go():
        tools = [{"name": "f", "schema": {"properties": {}}}]
        await e.python(c, "keep = 1\nawait f()", tools=tools)
        sess = repl_mod.get_pool().peek(c.id)
        pid = sess.proc.pid
        await asyncio.sleep(1.2)
        assert e.python_pending(c) is None
        r = await e.python(c, "print('keep' in globals())")
        assert r.stdout.strip() == "False" and r.state_lost
        assert not _alive(pid)

    asyncio.run(go())


def _alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def test_new_cell_while_paused_abandons_the_old_one(env):
    e, c, _ = env

    async def go():
        tools = [{"name": "f", "schema": {"properties": {}}}]
        await e.python(c, "await f()", tools=tools)
        r = await e.python(c, "print(7)")
        assert r.stdout.strip() == "7" and e.python_pending(c) is None

    asyncio.run(go())


def test_container_delete_kills_the_process(env):
    e, c, m = env

    async def go():
        await e.python(c, "x = 1")
        pid = repl_mod.get_pool().peek(c.id).proc.pid
        m.delete(c.id)
        await asyncio.sleep(0.3)
        assert not _alive(pid)
        assert repl_mod.get_pool().peek(c.id) is None

    asyncio.run(go())


def test_fresh_flag_starts_a_clean_process(env):
    a, b = session_run(
        env, [cell("v = 1"), cell("print('v' in globals())", fresh=True)]
    )
    assert b.stdout.strip() == "False"


def test_lru_pool_closes_the_oldest_idle_session(env, monkeypatch):
    monkeypatch.setattr(repl_mod, "MAX_SESSIONS", 2)
    e, c, m = env
    others = [m.create(), m.create()]

    async def go():
        for box in [c, *others]:
            await e.python(box, "1")
        assert repl_mod.get_pool().peek(c.id) is None
        assert repl_mod.get_pool().peek(others[1].id) is not None

    asyncio.run(go())


def test_idle_session_does_not_cost_a_wall_clock_budget(env, monkeypatch):
    monkeypatch.setenv("YUNSHU_CODE_EXECUTION_TIMEOUT", "3")
    assert settings.get("YUNSHU_CODE_EXECUTION_TIMEOUT") == 3.0
    t0 = time.monotonic()
    a, b = session_run(env, [cell("1"), cell("import time; time.sleep(0.2); 2")])
    assert b.ok and time.monotonic() - t0 < 30


def test_forged_protocol_message_for_an_undeclared_tool_is_ignored(env):
    e, c, _ = env

    async def go():
        tools = [{"name": "f", "schema": {"properties": {}}}]
        # fd 4 is the worker's private protocol descriptor: the cell can find and write it
        r = await e.python(
            c,
            "import os, json\nfor fd in range(3, 8):\n    try:\n        os.write(fd, (json.dumps({'op':'tool_call','call_id':'x','name':'evil','input':{}})+'\\n').encode())\n    except OSError:\n        pass\nprint('ok')",
            tools=tools,
        )
        assert r.ok and not r.calls and r.stdout.strip().endswith("ok")

    asyncio.run(go())


def test_too_many_tool_calls_in_one_cell_kill_it(env, monkeypatch):
    monkeypatch.setattr(repl_mod, "MAX_TOOL_CALLS", 5)
    e, c, _ = env

    async def go():
        tools = [{"name": "f", "schema": {"properties": {}}}]
        r = await e.python(
            c,
            "import asyncio\nawait asyncio.gather(*[f() for _ in range(20)])",
            tools=tools,
        )
        assert not r.ok and r.error_code == "too_many_requests"
        assert e.python_pending(c) is None
        again = await e.python(c, "print(1)")
        assert again.ok and again.stdout.strip() == "1"

    asyncio.run(go())


def test_oversized_tool_arguments_kill_the_cell(env, monkeypatch):
    monkeypatch.setattr(repl_mod, "MAX_CALL_BYTES", 1000)
    e, c, _ = env

    async def go():
        tools = [{"name": "f", "schema": {"properties": {"s": {}}}}]
        r = await e.python(c, "await f(s='x' * 5000)", tools=tools)
        assert not r.ok and r.error_code == "too_many_requests"

    asyncio.run(go())


def test_background_thread_allocating_after_the_cell_is_killed_when_idle(
    env, monkeypatch
):
    monkeypatch.setenv("YUNSHU_CODE_EXECUTION_MEMORY_MB", "200")
    e, c, _ = env

    async def go():
        r = await e.python(
            c,
            "import threading, time\n"
            "def hog():\n    time.sleep(0.5)\n    global big\n    big = bytearray(700*1024*1024)\n    time.sleep(30)\n"
            "threading.Thread(target=hog, daemon=True).start()\nprint('started')",
        )
        assert r.ok
        pid = repl_mod.get_pool().peek(c.id).proc.pid
        for _ in range(60):
            await asyncio.sleep(0.2)
            if not _alive(pid):
                break
        assert not _alive(pid)
        again = await e.python(c, "print('big' in globals())")
        assert again.stdout.strip() == "False" and again.state_lost

    asyncio.run(go())


def test_python_has_no_network_and_no_loopback(env):
    import socket
    import threading

    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]
    hits = []

    def accept():
        with contextlib.suppress(OSError):  # closed by the test
            hits.append(srv.accept())

    threading.Thread(target=accept, daemon=True).start()
    (r,) = session_run(
        env,
        [
            cell(
                "import socket\n"
                "out = []\n"
                f"for target in (('127.0.0.1', {port}), ('93.184.216.34', 80)):\n"
                "    s = socket.socket(); s.settimeout(3)\n"
                "    try:\n        s.connect(target); out.append('connected')\n"
                "    except OSError:\n        out.append('blocked')\n"
                "print(out)"
            )
        ],
    )
    srv.close()
    assert r.stdout.strip() == "['blocked', 'blocked']" and not hits


def test_doctor_self_test_runs_the_sandbox_python(monkeypatch):
    import importlib

    doctor = importlib.import_module("yunshu_cli.doctor")
    c = doctor.check_code_execution()
    assert c.status in ("ok", "warn", "fail")
    if c.status == "ok":
        assert "self-test passed" in c.detail
