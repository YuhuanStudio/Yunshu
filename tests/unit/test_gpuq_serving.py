"""gpuq serving awareness: idle-gated start, SIGSTOP preemption, pause records, memory admission."""

import contextlib
import http.server
import importlib.util
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


class FakeProd:
    """A tiny stand-in for /v1/yunshu/status whose load a test toggles."""

    def __init__(self):
        self.active = 0
        self.queued = 0
        outer = self

        class H(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                body = json.dumps(
                    {"requests": {"active": outer.active, "queued": outer.queued}}
                ).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *a):
                pass

        self.srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.srv.server_address[1]}"
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def close(self):
        self.srv.shutdown()


@pytest.fixture
def prod():
    p = FakeProd()
    yield p
    p.close()


@pytest.fixture
def q(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "gpuq_serving_test", REPO / "scripts/dev/gpuq.py"
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    monkeypatch.setattr(m, "ROOT", tmp_path)
    monkeypatch.setattr(m, "JOBS", tmp_path / "jobs")
    monkeypatch.setattr(m, "LOGS", tmp_path / "logs")
    monkeypatch.setattr(m, "POLL_S", 0.05)
    (tmp_path / "jobs").mkdir()
    (tmp_path / "logs").mkdir()
    for k in list(os.environ):
        if k.startswith("GPUQ_") and k != "GPUQ_DIR":
            monkeypatch.delenv(k)
    yield m
    # A failed assertion can leave a job stopped (SIGSTOP) with no runner left to
    # resume or cancel it; reap every job process group the test started.
    for f in (tmp_path / "jobs").glob("*.json"):
        pid = m._read(f).get("pid")
        if pid:
            for sig in (signal.SIGCONT, signal.SIGKILL):
                with contextlib.suppress(OSError):
                    os.killpg(pid, sig)


@pytest.fixture
def gp():
    sys.path.insert(0, str(REPO / "scripts/dev"))
    import gpuq_pause

    return gpuq_pause


def _gate(q, prod, **kw):
    g = q.ServingGate()
    cfg = {
        "urls": [prod.url],
        "poll_s": 0.05,
        "idle_start_s": 0.3,
        "idle_resume_s": 0.3,
    }
    g.configure({**cfg, **kw})
    return g


def _job(q, cmd, **kw):
    jid = f"j{len(list(q.JOBS.glob('*.json')))}"
    job = {
        "id": jid,
        "label": "t",
        "cmd": cmd,
        "cwd": os.getcwd(),
        "env": dict(os.environ),
        "timeout_s": 60,
        "stall_s": 60,
        "priority": 0,
        "submitted": time.time(),
        "state": "pending",
        "mem_gb": 0,
        **kw,
    }
    path = q.JOBS / f"{jid}.json"
    q._write(path, job)
    return job, path


def _state(pid):
    return subprocess.run(
        ["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True
    ).stdout.strip()


def _run_bg(q, job, path, gate):
    t = threading.Thread(target=q._run_one, args=(job, path, gate), daemon=True)
    t.start()
    return t


def _wait_for(cond, timeout=10.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.02)
    return False


def test_default_without_servers_is_unchanged(q):
    g = q.ServingGate()
    g.configure(q.load_serving_config())
    assert not g.enabled and g.may_start() and not g.busy
    job, path = _job(
        q, ["true"], mem_gb=10_000
    )  # absurd need: ignored without a serving config
    assert q._blocker(job, g) is None
    q._run_one(job, path, g)
    done = q._read(path)
    assert done["state"] == "done" and done["pauses"] == []


def test_config_file_and_env(q, monkeypatch):
    (q.ROOT / "serving.json").write_text(
        json.dumps({"urls": ["http://a"], "idle_start_s": 5})
    )
    assert q.load_serving_config()["urls"] == ["http://a"]
    monkeypatch.setenv("GPUQ_SERVING_URLS", "http://b, http://c")
    monkeypatch.setenv("GPUQ_IDLE_RESUME_S", "7")
    cfg = q.load_serving_config()
    assert cfg["urls"] == ["http://b", "http://c"]
    assert cfg["idle_resume_s"] == 7 and cfg["idle_start_s"] == 5


def test_idle_gated_start(q, prod):
    g = _gate(q, prod)
    job, _ = _job(q, ["true"])
    prod.active = 1
    g.poll()
    assert q._blocker(job, g) == "idle"
    prod.active = 0
    g.poll()
    assert q._blocker(job, g) == "idle"  # idle, but not for idle_start_s yet
    time.sleep(0.35)
    g.poll()
    assert q._blocker(job, g) is None
    prod.queued = 1  # queued alone also counts as busy
    g.poll()
    assert g.busy and q._blocker(job, g) == "idle"


def test_every_server_must_be_idle(q, prod):
    other = FakeProd()
    try:
        g = _gate(q, prod, urls=[prod.url, other.url], idle_start_s=0)
        other.active = 1
        g.poll()
        assert g.busy
        other.active = 0
        g.poll()
        assert not g.busy and g.may_start()
    finally:
        other.close()


def test_serving_ok_bypasses_gate(q, prod):
    g = _gate(q, prod)
    prod.active = 3
    g.poll()
    job, _ = _job(q, ["true"], serving_ok=True)
    assert q._blocker(job, g) is None
    other, _ = _job(q, ["true"])
    picked = q._pick([job, other], lambda j: q._blocker(j, g) is None)
    assert picked["id"] == job["id"]


def test_unreachable_is_idle_but_errors_are_busy(q):
    assert q._fetch_busy("http://127.0.0.1:9", None, timeout=1) is False  # refused

    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(500)
            self.end_headers()

        def log_message(self, *a):
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        assert q._fetch_busy(f"http://127.0.0.1:{srv.server_address[1]}", None) is True
    finally:
        srv.shutdown()


def test_preempt_pause_resume_and_intervals(q, prod):
    g = _gate(q, prod)
    job, path = _job(q, ["sleep", "30"], timeout_s=30)
    g.poll()
    t = _run_bg(q, job, path, g)
    assert _wait_for(lambda: job.get("pid"))
    pid = job["pid"]
    assert _wait_for(lambda: _state(pid)[:1] in "SR")
    t_busy = time.time()
    prod.active = 1
    assert _wait_for(lambda: _state(pid).startswith("T"), 2.0)
    assert time.time() - t_busy < 1.0  # within about one poll interval
    assert q._read(path)["paused"] is True
    pf = json.loads((q.LOGS / f"{job['id']}.pauses.json").read_text())
    assert len(pf["pauses"]) == 1 and pf["pauses"][0][1] is None
    prod.active = 0
    time.sleep(0.1)
    assert _state(pid).startswith("T")  # still inside the idle-resume window
    assert _wait_for(lambda: _state(pid)[:1] in "SR", 3.0)
    d = q._read(path)
    assert d["paused"] is False and len(d["pauses"]) == 1
    t0, t1 = d["pauses"][0]
    assert t0 >= t_busy - 0.1 and t1 - t0 >= 0.25
    q._patch_job(path, cancel=True)
    t.join(10)
    assert q._read(path)["state"] == "cancelled"


def test_paused_time_excluded_from_timeout_and_stall(q, prod):
    g = _gate(q, prod, idle_resume_s=0.1)
    # 1.2 s of work, 1.5 s timeout and stall limits, stopped for well over 1.5 s in the middle
    job, path = _job(
        q,
        ["sh", "-c", "echo a; sleep 0.6; echo b; sleep 0.6; echo c"],
        timeout_s=1.5,
        stall_s=1.5,
    )
    g.poll()
    t = _run_bg(q, job, path, g)
    assert _wait_for(lambda: job.get("pid"))
    time.sleep(0.2)
    prod.active = 1
    assert _wait_for(lambda: q._read(path).get("paused"), 2)
    time.sleep(1.8)
    prod.active = 0
    t.join(15)
    d = q._read(path)
    assert d["state"] == "done", d
    assert sum(b - a for a, b in d["pauses"]) > 1.5


def test_timeout_still_fires_on_active_time(q, prod):
    g = _gate(q, prod)
    job, path = _job(q, ["sleep", "30"], timeout_s=0.5, stall_s=60)
    g.poll()
    t = _run_bg(q, job, path, g)
    t.join(15)
    assert q._read(path)["state"] == "timeout"


def test_serving_ok_job_is_never_paused(q, prod):
    g = _gate(q, prod)
    prod.active = 1
    g.poll()
    job, path = _job(q, ["sh", "-c", "sleep 0.5"], serving_ok=True)
    q._run_one(job, path, g)
    d = q._read(path)
    assert d["state"] == "done" and d["pauses"] == []


def test_pause_file_env_and_helper(q, prod, gp, monkeypatch):
    g = _gate(q, prod)
    out = q.ROOT / "o.txt"
    job, path = _job(
        q,
        ["sh", "-c", 'echo "$GPUQ_PAUSE_FILE" > "$OUTF"'],
        env={**os.environ, "OUTF": str(out)},
    )
    q._run_one(job, path, g)
    assert out.read_text().strip() == str(q.LOGS / f"{job['id']}.pauses.json")

    f = q.ROOT / "p.json"
    f.write_text(json.dumps({"pauses": [[100.0, 110.0], [200.0, None]]}))
    monkeypatch.setenv("GPUQ_PAUSE_FILE", str(f))
    assert gp.was_paused(105, 106) and gp.was_paused(95, 101)
    assert not gp.was_paused(111, 150)
    assert gp.was_paused(300, 301)  # an open interval runs to now
    f.write_text("garbage")
    assert gp.was_paused(0, 1)  # unreadable file fails closed
    monkeypatch.delenv("GPUQ_PAUSE_FILE")
    assert not gp.was_paused(0, 1e12)  # not under gpuq


def test_timed_retries_then_fails_closed(q, gp, monkeypatch):
    monkeypatch.setattr(gp, "_RETRY_WAIT_S", 0)
    f = q.ROOT / "p.json"
    monkeypatch.setenv("GPUQ_PAUSE_FILE", str(f))
    calls = []

    def sample():
        calls.append(1)
        now = time.time()
        pauses = [[now - 1, now + 1]] if len(calls) == 1 else []
        f.write_text(json.dumps({"pauses": pauses}))
        return "ok"

    f.write_text(json.dumps({"pauses": []}))
    assert gp.timed(sample) == "ok" and len(calls) == 2
    f.write_text(json.dumps({"pauses": [[0, 1e12]]}))
    with pytest.raises(gp.PausedSampleError):
        gp.timed(lambda: 1, retries=2)


def test_memory_admission(q, prod):
    g = _gate(q, prod, reserve_gb=16)
    job, _ = _job(q, ["true"], mem_gb=24)
    assert q.mem_blocked(job, g, free=lambda: 50.0) is False  # 50 - 24 >= 16
    assert q.mem_blocked(job, g, free=lambda: 39.0) is True  # 39 - 24 < 16
    assert q.mem_blocked(job, g, free=lambda: None) is False  # unmeasurable: no wedge
    small, _ = _job(q, ["true"], mem_gb=0)
    assert q.mem_blocked(small, g, free=lambda: 1.0) is False
    g.poll()
    time.sleep(0.35)
    g.poll()
    assert q._blocker(job, g, free=lambda: 39.0) == "mem"
    assert q._blocker(job, g, free=lambda: 60.0) is None
    off = q.ServingGate()
    off.configure({})
    assert q.mem_blocked(job, off, free=lambda: 1.0) is False  # no config: unchanged
    explicit = q.ServingGate()
    explicit.configure({"reserve_gb": 8})
    assert q.mem_blocked(job, explicit, free=lambda: 20.0) is True


def test_free_memory_reads_something(q):
    v = q.free_memory_gb()
    assert v is None or v > 0


def test_status_labels(q):
    assert q.display_state({"state": "running", "paused": True}) == "paused"
    assert q.display_state({"state": "pending", "waiting": "idle"}) == "wait-idle"
    assert q.display_state({"state": "pending", "waiting": "mem"}) == "wait-mem"
    assert q.display_state({"state": "running"}) == "running"


def test_pauser_only_signals_job_group(q, monkeypatch):
    sent = []
    monkeypatch.setattr(q.os, "killpg", lambda pid, sig: sent.append((pid, sig)))
    job, path = _job(q, ["true"], pid=4242)
    p = q.Pauser(job, path)
    p.pause(1.0)
    p.resume(2.0)
    assert sent == [(4242, q.signal.SIGSTOP), (4242, q.signal.SIGCONT)]
    job["pid"] = 1  # never pid 0/1 (would be everything / launchd)
    sent.clear()
    p.pause(3.0)
    assert sent == []


def test_priority_preemption_runs_high_while_low_stopped(q, monkeypatch):
    # CPU stand-ins only: low work has tight active timeout/stall limits.
    monkeypatch.setattr(q, "free_memory_gb", lambda: 100.0)
    low, path = _job(
        q,
        [
            sys.executable,
            "-c",
            "import time; print('low', flush=True); time.sleep(1.0)",
        ],
        priority=-1,
        timeout_s=1.4,
        stall_s=1.4,
        mem_gb=24,
    )
    t = _run_bg(q, low, path, q.ServingGate())
    assert _wait_for(lambda: low.get("pid"))
    high, hp = _job(
        q, [sys.executable, "-c", "import time; time.sleep(1.6)"], priority=0, mem_gb=24
    )
    try:
        assert _wait_for(lambda: q._read(hp).get("state") == "running", 3)
        assert _state(low["pid"]).startswith("T")
        assert q._read(path)["paused"] is True
        assert (
            json.loads((q.LOGS / (low["id"] + ".pauses.json")).read_text())["pauses"][
                -1
            ][1]
            is None
        )
        t.join(8)
        assert not t.is_alive()
        done = q._read(path)
        hd = q._read(hp)
        assert done["state"] == hd["state"] == "done"
        assert done["pauses"][0][1] - done["pauses"][0][0] >= 1.5
        assert done["paused"] is False
        assert hd["ended"] <= done["pauses"][0][1]
    finally:
        q._patch_job(path, cancel=True)
        q._patch_job(hp, cancel=True)
        t.join(8)


@pytest.mark.parametrize("available", [20.0, None])
def test_priority_preemption_never_stops_when_memory_not_admitted(
    q, monkeypatch, available
):
    monkeypatch.setattr(q, "free_memory_gb", lambda: available)
    low, path = _job(q, ["sleep", "0.5"], priority=-2, mem_gb=24)
    high, hp = _job(q, ["true"], priority=0, mem_gb=24)
    q._run_one(low, path)
    assert q._read(path)["state"] == "done"
    assert q._read(path)["pauses"] == []
    assert q._read(hp)["state"] == "pending"


def test_priority_pause_drains_all_high_jobs_before_resuming(q, monkeypatch):
    monkeypatch.setattr(q, "free_memory_gb", lambda: 100.0)
    low, path = _job(q, ["sleep", "0.6"], priority=-1, mem_gb=24)
    _, first = _job(q, ["sleep", "0.1"], priority=0, mem_gb=24)
    _, second = _job(q, ["sleep", "0.1"], priority=1, mem_gb=24)
    q._run_one(low, path)
    done = q._read(path)
    assert done["state"] == "done" and len(done["pauses"]) == 1
    assert q._read(second)["ended"] <= q._read(first)["started"]
    assert done["pauses"][0][1] >= q._read(first)["ended"]


def test_priority_pause_waits_for_pending_then_resumes_on_cancel(q, monkeypatch):
    # Requeues exhausted: a stopped backlog job keeps its pause while p0 waits.
    low, path = _job(
        q, ["true"], priority=-1, pid=4242, state="running", requeues=q.MAX_REQUEUES
    )
    _, hp = _job(q, ["true"], priority=0, mem_gb=24)
    monkeypatch.setattr(q.os, "killpg", lambda *_: None)
    monkeypatch.setattr(q, "free_memory_gb", lambda: 1.0)
    p = q.Pauser(low, path)
    p.pause(time.time(), reason="priority")
    g = q.ServingGate()
    assert q._priority_step(p, g, time.time()) is True
    assert q._read(hp)["waiting"] == "mem"
    q._patch_job(hp, cancel=True)
    assert q._priority_step(p, g, time.time()) is False
    p.step(g, False, time.time())
    assert not p.paused


def test_preempting_runner_error_does_not_abandon_low_job(q, monkeypatch):
    low, path = _job(q, ["sleep", "0.3"], priority=-1, mem_gb=24)
    _, hp = _job(q, ["true"], priority=0, mem_gb=24)
    run = q._run_one

    def fail_high(job, path, gate=None):
        if job["priority"] >= 0:
            raise RuntimeError("injected runner error")
        return run(job, path, gate)

    monkeypatch.setattr(q, "_run_one", fail_high)
    monkeypatch.setattr(q, "free_memory_gb", lambda: 100.0)
    run(low, path)
    assert q._read(path)["state"] == "done"
    assert q._read(path)["paused"] is False
    assert q._read(hp)["state"] == "failed"


def test_adoption_closes_open_pause_file(q, monkeypatch):
    job, path = _job(
        q,
        ["true"],
        state="running",
        priority=-1,
        pid=123,
        pauses=[[time.time() - 3, None]],
        paused=True,
        pause_reason="priority",
    )
    monkeypatch.setattr(q, "_alive", lambda _: False)
    q._adopt(job, path)
    saved = json.loads((q.LOGS / (job["id"] + ".pauses.json")).read_text())
    assert saved["pauses"][0][1] is not None
    assert q._read(path)["state"] == "lost"


@pytest.mark.parametrize("already_paused", [False, True])
def test_priority_holds_pause_during_serving_start_window(
    q, monkeypatch, already_paused
):
    monkeypatch.setattr(q, "free_memory_gb", lambda: 100.0)
    signals = []
    monkeypatch.setattr(q.os, "killpg", lambda pid, sig: signals.append((pid, sig)))
    low, path = _job(q, ["true"], state="running", pid=4242, priority=-1, mem_gb=24)
    _, hp = _job(q, ["true"], priority=0, mem_gb=24)
    p = q.Pauser(low, path)
    if already_paused:
        p.pause(time.time() - 2)
    g = q.ServingGate(fetch=lambda *_: False)
    g.configure(
        {"urls": ["http://production"], "idle_start_s": 120, "idle_resume_s": 0}
    )
    g.poll()
    assert g.may_resume() and not g.may_start()
    assert q._priority_step(p, g, time.time()) is True
    assert p.paused and low["pause_reason"] == "priority"
    assert q._read(hp)["state"] == "pending"
    assert q._read(hp)["waiting"] == "idle"
    assert signals == [(4242, q.signal.SIGSTOP)]
