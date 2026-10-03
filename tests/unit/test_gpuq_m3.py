"""Second-lane transport contracts without contacting the user's laptop."""

import importlib
import sys
import threading
import time
from pathlib import Path

import pytest

DEV = Path(__file__).resolve().parents[2] / "scripts/dev"
sys.path.insert(0, str(DEV))
sys.path.insert(0, str(DEV.parent / "research"))


@pytest.fixture
def queue(tmp_path, monkeypatch):
    import gpuq

    monkeypatch.setenv("GPUQ_DIR", str(tmp_path))
    q = importlib.reload(gpuq)
    q.JOBS.mkdir()
    q.LOGS.mkdir()
    monkeypatch.setattr(q, "_ensure_daemon", lambda: None)
    return q


def test_device_admission(queue):
    with pytest.raises(ValueError, match="quiet"):
        queue.submit(["true"], "quiet", 1, 0, quiet=True, device="any")
    with pytest.raises(ValueError, match="28 GB"):
        queue.submit(["true"], "big", 1, 0, mem_gb=29, device="m3")
    for device in ("m3", "any", "m5"):
        jid = queue.submit(["true"], device, 1, 0, mem_gb=16, device=device)
        row = queue._read(queue.JOBS / (jid + ".json"))
        assert row["device"] == device
        assert row["pid"] is None and row["rc"] is None
    jid = queue.submit(["true"], "default", 1, 0)
    assert queue._read(queue.JOBS / (jid + ".json"))["device"] == "m5"


def test_mapping(monkeypatch):
    from gpuq_remote import mapped, paths

    monkeypatch.setattr("subprocess.check_output", lambda *a, **k: "/src/wt\n")
    job = dict(
        id="smoke",
        cwd="/src/wt/sub",
        cmd=[
            "/src/.venv/bin/python",
            "--model=/Volumes/Micron/models/Q27/a",
            "/Volumes/P5Plus/yunshu-build/one.json",
        ],
        env={"MODEL": "/Volumes/P5Plus/models/Q08", "SRC": "/src/wt"},
    )
    top, wt, pairs, models = paths(job, "/remote")
    assert top == "/src/wt"
    assert mapped(job["cwd"], pairs) == wt + "/sub"
    assert mapped(job["cmd"][0], pairs) == "/remote/.venv/bin/python"
    assert mapped(job["cmd"][1], pairs) == "--model=/remote/.m3-home/models/Q27/a"
    assert mapped(job["cmd"][2], pairs) == "/remote/.m3-home/out/one.json"
    assert len(models) == 2


def test_lanes_concurrent_any_once(queue, monkeypatch):
    events = []
    barrier = threading.Barrier(2)

    def execute(job, path, gate):
        events.append((job["id"], job["device"]))
        barrier.wait(timeout=5)
        job.update(state="done", rc=0, ended=time.time())
        queue._write(path, job)

    monkeypatch.setattr(queue, "_execute", execute)
    monkeypatch.setattr(queue, "POLL_S", 0.01)
    monkeypatch.setattr(queue, "IDLE_EXIT_S", 0.03)
    queue.submit(["true"], "m5", 1, 0)
    queue.submit(["true"], "any", 1, 0, device="any")
    queue.daemon()
    assert len(events) == 2
    assert {device for _, device in events} == {"m3", "m5"}
    assert len({jid for jid, _ in events}) == 2


@pytest.mark.parametrize("reason", ["cancelled", "timeout", "stalled"])
def test_transport_interrupt_kills_owned_process(queue, monkeypatch, reason):
    from gpuq_remote import InterruptedError, Remote

    job = dict(
        id="interrupt",
        env={},
        timeout_s=0.02 if reason == "timeout" else 60,
        stall_s=0.02 if reason == "stalled" else 60,
        state="running",
    )
    path = queue.JOBS / "interrupt.json"
    queue._write(path, {**job, "cancel": reason == "cancelled"})
    # Persisted cancel is merged in production by the CLI after launch.
    original = queue._write

    def write(p, data):
        original(p, {**data, "cancel": reason == "cancelled"})

    monkeypatch.setattr(queue, "_write", write)
    with (queue.LOGS / "interrupt.log").open("w") as log:
        remote = Remote(job, path, queue, log)
        with pytest.raises(InterruptedError, match=reason):
            remote.call([sys.executable, "-c", "import time; time.sleep(30)"])
    assert not queue._alive(job["pid"])


@pytest.mark.parametrize("rc,broken", [(0, False), (7, False), (None, True)])
def test_remote_receipt_and_rsync_on_failure(queue, monkeypatch, tmp_path, rc, broken):
    from gpuq_remote import Remote

    calls, copies = [], []
    source = str(tmp_path / "result.json")
    job = dict(
        id="fake",
        cwd=str(tmp_path),
        env={},
        cmd=["python", "--out", source],
        timeout_s=60,
        stall_s=60,
        state="running",
        rc=None,
    )

    def check(cmd, **kwargs):
        if "rev-parse" in cmd:
            return str(tmp_path) + "\n"
        if "write-tree" in cmd or "commit-tree" in cmd:
            return "abc\n"
        return str(rc)

    monkeypatch.setattr("subprocess.check_output", check)

    def call(self, cmd, env=None):
        calls.append(cmd)
        if broken and "remote lock timeout" in " ".join(cmd):
            raise RuntimeError("ssh failed")

    def sync(self, src, dest, back=False):
        copies.append((src, dest, back))
        if back:
            Path(dest).write_text('{"complete": true}\n')

    monkeypatch.setattr(Remote, "call", call)
    monkeypatch.setattr(Remote, "sync", sync)
    monkeypatch.setattr(Remote, "stop", lambda self: calls.append("stop"))
    path = queue.JOBS / "fake.json"
    queue._write(path, job)
    with (queue.LOGS / "fake.log").open("w") as log:
        remote = Remote(job, path, queue, log)
        if broken:
            with pytest.raises(RuntimeError, match="ssh failed"):
                remote.run()
        else:
            remote.run()
            assert job["rc"] == rc
    assert calls[-1] == "stop"
    assert copies[-1][2] is True
    import json

    assert json.loads(Path(source).read_text())["device"] == "m3"


def test_analysis_guardrails():
    from device_evidence import require_same_device

    with pytest.raises(ValueError, match="mixed-device"):
        require_same_device([{"device": "m3"}], [{"device": "m5"}])
    with pytest.raises(ValueError, match="performance verdict refused"):
        require_same_device([{"arms": [{"device": "m3"}]}], performance=True)
    assert require_same_device([{"device": "m3"}], [{"device": "m3"}]) == "m3"
    from decision_table import verdict

    assert "performance verdict refused" in verdict(
        [{"device": "m3", "complete": True}], None
    )
    from analyze_dflash_shape import summarize

    with pytest.raises(ValueError, match="performance verdict refused"):
        summarize([], {"device": "m3"})


def test_claim_any_is_exclusive(queue):
    jid = queue.submit(["true"], "any", 1, 0, device="any")
    path = queue.JOBS / (jid + ".json")
    first, second = queue._read(path), queue._read(path)
    assert queue._claim(first, "m3")
    assert not queue._claim(second, "m5")
    assert queue._read(path)["device"] == "m3"


def test_m3_smoke_cannot_approve_m5_receipt():
    from audit_flag_matrix import validate_smoke
    from audit_legacy_replay import eligible_smoke

    with pytest.raises(ValueError, match="performance verdict refused"):
        validate_smoke({"device": "m3"}, "sha", [])
    with pytest.raises(ValueError, match="mixed-device"):
        eligible_smoke({"device": "m5"}, {"device": "m3"}, "sha")


def test_mapping_respects_path_boundary():
    from gpuq_remote import mapped

    assert (
        mapped("/src/wt-other/file", [("/src/wt", "/remote/wt")])
        == "/src/wt-other/file"
    )
    assert (
        mapped("--out=/src/wt/file", [("/src/wt", "/remote/wt")])
        == "--out=/remote/wt/file"
    )


def test_remote_stop_only_signals_job_group(queue, monkeypatch):
    from gpuq_remote import Remote

    calls = []

    def run(cmd, **kw):
        calls.append(cmd)
        return type("Result", (), {"returncode": 0})()

    monkeypatch.setattr("subprocess.run", run)
    job = dict(id="owned", env={}, timeout_s=30)
    with (queue.LOGS / "owned.log").open("w") as log:
        Remote(job, queue.JOBS / "owned.json", queue, log).stop()
    script = calls[0][-1]
    assert "owned.pid" in script
    assert "kill -TERM --" in script and "kill -KILL --" in script
    assert "daemon" not in script


def test_quarantine_blocks_only_m3_lane(queue, monkeypatch):
    queue._write(queue.ROOT / "m3-quarantine.json", {"error": "ssh down"})
    remote = queue.submit(["true"], "remote", 1, 0, device="m3")
    local = queue.submit(["true"], "local", 1, 0)

    class StopError(Exception):
        pass

    def execute(job, path, gate):
        assert job["id"] == local
        raise StopError

    monkeypatch.setattr(queue, "_execute", execute)
    with pytest.raises(StopError):
        queue.daemon()
    assert queue._read(queue.JOBS / (remote + ".json"))["state"] == "pending"


def test_collection_never_promotes_m3_source_evidence(tmp_path):
    import json

    from device_evidence import require_same_device, stamp_output

    path = tmp_path / "analysis.json"
    path.write_text('{"arms": [{"device": "m3", "tps": 1}]}')
    stamp_output(path, "m5")
    rows = json.loads(path.read_text())
    assert rows["execution_device"] == "m5"
    assert rows["arms"][0]["device"] == "m3"
    with pytest.raises(ValueError, match="mixed-device"):
        require_same_device([rows], performance=True)


@pytest.mark.parametrize("cancel_before_start", [False, True])
def test_remote_worker_cancelled_while_lock_waiting(tmp_path, cancel_before_start):
    import fcntl
    import json
    import os
    import subprocess

    lock_path = tmp_path / "m3run.lock"
    pid = tmp_path / "worker.pid"
    rc = tmp_path / "worker.rc"
    unexpected = tmp_path / "must-not-run"
    payload = dict(
        env={**os.environ, "TMPDIR": str(tmp_path)},
        timeout=4,
        cwd=str(tmp_path),
        pid=str(pid),
        rc=str(rc),
        lock=str(lock_path),
        cmd=[
            sys.executable,
            "-c",
            f"from pathlib import Path; Path({str(unexpected)!r}).touch()",
        ],
    )
    with lock_path.open("a") as lock:
        fcntl.lockf(lock, fcntl.LOCK_EX)
        if cancel_before_start:
            Path(str(pid) + ".cancel").touch()
        proc = subprocess.Popen(
            [sys.executable, str(DEV / "gpuq_remote_worker.py"), json.dumps(payload)]
        )
        try:
            deadline = time.monotonic() + 2
            while (
                not pid.exists() and proc.poll() is None and time.monotonic() < deadline
            ):
                time.sleep(0.01)
            Path(str(pid) + ".cancel").touch()
            fcntl.lockf(lock, fcntl.LOCK_UN)
            assert proc.wait(timeout=3) == 0
            assert rc.read_text() == "130"
            assert not unexpected.exists()
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()


def test_remote_worker_timeout_includes_lock_wait(tmp_path):
    import fcntl
    import json
    import os
    import subprocess

    lock_path = tmp_path / "lock"
    rc = tmp_path / "rc"
    payload = dict(
        env={**os.environ, "TMPDIR": str(tmp_path)},
        timeout=0.1,
        cwd=str(tmp_path),
        pid=str(tmp_path / "pid"),
        rc=str(rc),
        lock=str(lock_path),
        cmd=[sys.executable, "-c", "raise SystemExit(99)"],
    )
    with lock_path.open("a") as lock:
        fcntl.lockf(lock, fcntl.LOCK_EX)
        subprocess.run(
            [sys.executable, str(DEV / "gpuq_remote_worker.py"), json.dumps(payload)],
            check=True,
            timeout=3,
        )
    assert rc.read_text() == "124"


def test_remote_worker_cancel_kills_descendants(tmp_path):
    import json
    import os
    import subprocess

    beat = tmp_path / "heartbeat"
    ready = tmp_path / "ready"
    pid = tmp_path / "pid"
    rc = tmp_path / "rc"
    child = f"import time\nf=open({str(beat)!r}, 'a')\nwhile True:\n f.write('x'); f.flush(); time.sleep(.02)\n"
    command = f'import subprocess,time; from pathlib import Path; subprocess.Popen([{sys.executable!r},"-c",{child!r}]); Path({str(ready)!r}).touch(); time.sleep(30)'
    payload = dict(
        env={**os.environ, "TMPDIR": str(tmp_path)},
        timeout=5,
        cwd=str(tmp_path),
        pid=str(pid),
        rc=str(rc),
        lock=str(tmp_path / "lock"),
        cmd=[sys.executable, "-c", command],
    )
    proc = subprocess.Popen(
        [sys.executable, str(DEV / "gpuq_remote_worker.py"), json.dumps(payload)]
    )
    try:
        deadline = time.monotonic() + 3
        while not beat.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ready.exists() and beat.exists()
        proc.terminate()
        assert proc.wait(timeout=3) == 0
        assert rc.read_text() == "130"
        size = beat.stat().st_size
        time.sleep(0.1)
        assert beat.stat().st_size == size
        assert not pid.exists()
    finally:
        if proc.poll() is None:
            proc.terminate()
            proc.wait(timeout=3)


def test_paired_accuracy_report_refuses_mixed_devices(monkeypatch):
    from types import SimpleNamespace

    sys.path.insert(0, str(DEV.parent / "research" / "accuracy"))
    import paired_eval

    monkeypatch.setattr(
        paired_eval,
        "arm_attempt_counts",
        lambda *a: dict(
            attempts=1, error_attempts=0, attempted_items=1, unresolved_items=0
        ),
    )
    monkeypatch.setattr(
        paired_eval,
        "load_arm",
        lambda bench, arm: {
            "q": dict(correct=True, device="m3" if arm == "ref" else "m5")
        },
    )
    args = SimpleNamespace(bench="gsm8k", ref="ref", cand="cand", field="correct")
    with pytest.raises(ValueError, match="mixed-device"):
        paired_eval.cmd_report(args)


@pytest.mark.parametrize("device", ["m3", "any"])
def test_legacy_daemon_cannot_misroute_remote_jobs(queue, monkeypatch, device):
    monkeypatch.setattr(queue, "_daemon_running", lambda: True)
    (queue.ROOT / "daemon.lock").write_text("123")
    # A stale receipt from a different daemon must not authorize the lane either.
    queue._write(
        queue.ROOT / "daemon-capabilities.json", {"pid": 456, "devices": ["m5", "m3"]}
    )
    with pytest.raises(ValueError, match="running daemon does not support M3"):
        queue.submit(["true"], "remote", 1, 0, device=device)
    assert not list(queue.JOBS.glob("*.json"))
    queue._write(
        queue.ROOT / "daemon-capabilities.json", {"pid": 123, "devices": ["m5", "m3"]}
    )
    assert queue.submit(["true"], "remote", 1, 0, device=device)


def test_default_m5_still_submits_to_legacy_daemon(queue, monkeypatch):
    monkeypatch.setattr(queue, "_daemon_running", lambda: True)
    assert queue.submit(["true"], "local", 1, 0)
