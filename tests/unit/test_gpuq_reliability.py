"""Queue CLI contracts use an isolated queue; never touch the live daemon."""

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

DEV = Path(__file__).resolve().parents[2] / "scripts/dev"


@pytest.fixture
def q(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "gpuq_reliability_test", DEV / "gpuq.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name, path in (
        ("ROOT", tmp_path),
        ("JOBS", tmp_path / "jobs"),
        ("LOGS", tmp_path / "logs"),
    ):
        monkeypatch.setattr(module, name, path)
        path.mkdir(exist_ok=True)
    monkeypatch.setattr(module, "_ensure_daemon", lambda: None)
    return module


def job(q, jid, **kw):
    data = {
        "id": jid,
        "label": jid,
        "state": "done",
        "rc": 0,
        "submitted": 1,
        "cwd": str(q.ROOT),
        "cmd": ["true"],
        **kw,
    }
    q._write(q.JOBS / (jid + ".json"), data)
    return data


def test_duplicate_active_label_refused_and_prior_outputs_preserved(q):
    output = q.ROOT / "old.jsonl"
    output.write_text("original complete\n")
    first = q.submit(["true"], "reuse", 1, 0)
    before = {p: p.read_bytes() for p in q.ROOT.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match="active label"):
        q.submit(["true"], "reuse", 1, 0)
    assert all(p.read_bytes() == content for p, content in before.items())
    q._patch_job(q.JOBS / (first + ".json"), state="done", rc=0)
    second = q.submit(["true"], "reuse", 1, 0)
    assert first != second and output.read_text() == "original complete\n"
    assert (q.JOBS / (first + ".json")).exists()


def test_submit_declares_outputs_and_complete(q, monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gpuq",
            "submit",
            "--label",
            "declared",
            "--out",
            "a",
            "--out",
            "b",
            "--expect-complete",
            "--",
            "true",
        ],
    )
    assert q.main() == 0
    jid = capsys.readouterr().out.strip()
    data = q._read(q.JOBS / (jid + ".json"))
    assert data["outputs"] == [str(Path.cwd() / "a"), str(Path.cwd() / "b")]
    assert data["expect_complete"] is True


def test_bounded_wait_checks_all_ids_and_prints_summary(q, capsys):
    job(q, "pending", state="pending", rc=None)
    job(q, "ok")
    assert q.wait(["pending", "ok"], max_seconds=0) == 2
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 2
    assert "pending: pending rc=None" in lines[0] and "missing_outputs=" in lines[0]
    assert "ok: done rc=0" in lines[1]
    job(q, "pending")
    assert q.wait(["pending", "ok"], max_seconds=0) == 0
    job(q, "ok", state="failed", rc=7)
    assert q.wait(["pending", "ok"], max_seconds=0) == 1
    assert q.wait(["unknown"], max_seconds=0) == 1


@pytest.mark.parametrize(
    "content,expect_complete,expected",
    [
        (None, False, 1),
        ("", False, 1),
        ("partial\n", True, 1),
        ("partial\ncomplete\n", True, 0),
        ("partial\n", False, 0),
    ],
)
def test_wait_verifies_output_contract(q, content, expect_complete, expected, capsys):
    path = q.ROOT / "result"
    if content is not None:
        path.write_text(content)
    job(q, "result", outputs=[str(path)], expect_complete=expect_complete)
    assert q.wait(["result"], max_seconds=0) == expected
    text = capsys.readouterr().out
    assert "missing_outputs=" in text
    if expected:
        assert str(path) in text


def test_wait_cli_max_seconds(q, monkeypatch, capsys):
    job(q, "pending", state="pending")
    monkeypatch.setattr(sys, "argv", ["gpuq", "wait", "--max-seconds", "0", "pending"])
    assert q.main() == 2
    assert "pending: pending" in capsys.readouterr().out


@pytest.mark.parametrize(
    "waiting,reason",
    [("mem", "memory admission"), ("idle", "serving"), ("pause", "pause")],
)
def test_idle_status_explains_pending_blocker(q, waiting, reason, monkeypatch, capsys):
    job(q, "pending", state="pending", waiting=waiting)
    monkeypatch.setattr(q, "_daemon_running", lambda: True)
    q.status()
    assert "idle:" in capsys.readouterr().out
    q.status()
    assert reason in capsys.readouterr().out


def test_gpuq_scripts_byte_compile_with_python39(tmp_path):
    candidates = [shutil.which("python3.9"), "/usr/bin/python3"]
    interpreter = next(
        (
            p
            for p in candidates
            if p
            and subprocess.run(
                [p, "-c", "import sys; sys.exit(sys.version_info[:2] != (3, 9))"],
                capture_output=True,
            ).returncode
            == 0
        ),
        None,
    )
    if interpreter is None:
        pytest.skip("Python 3.9 is unavailable")
    script = "import py_compile, sys; [py_compile.compile(p, doraise=True) for p in sys.argv[1:]]"
    subprocess.run(
        [interpreter, "-c", script, *map(str, DEV.glob("gpuq*.py"))],
        env={**os.environ, "PYTHONPYCACHEPREFIX": str(tmp_path)},
        check=True,
    )


def test_concurrent_submit_serializes_label_admission(q):
    from concurrent.futures import ThreadPoolExecutor

    def submit():
        try:
            return q.submit(["true"], "shared", 1, 0)
        except ValueError:
            return None

    with ThreadPoolExecutor(max_workers=4) as pool:
        ids = list(pool.map(lambda _: submit(), range(4)))
    assert sum(jid is not None for jid in ids) == 1
    assert len(q._jobs()) == 1


def test_wait_deadline_is_global_and_sleep_is_bounded(q, monkeypatch, capsys):
    job(q, "first", state="pending")
    job(q, "second", state="running")
    elapsed = [0.0]
    sleeps = []
    monkeypatch.setattr(q.time, "monotonic", lambda: elapsed[0])

    def sleep(seconds):
        sleeps.append(seconds)
        elapsed[0] += seconds

    monkeypatch.setattr(q.time, "sleep", sleep)
    assert q.wait(["first", "second"], max_seconds=0.1) == 2
    assert sleeps == [0.1]
    assert len(capsys.readouterr().out.splitlines()) == 2


def test_label_reuse_preserves_all_previous_job_artifacts(q):
    first = q.submit(["true"], "reuse-files", 1, 0)
    for suffix in ("log", "rc", "pauses.json"):
        (q.LOGS / (first + "." + suffix)).write_text("original " + suffix)
    q._patch_job(q.JOBS / (first + ".json"), state="done", rc=0)
    snapshot = {
        p: p.read_bytes() for p in [q.JOBS / (first + ".json"), *q.LOGS.iterdir()]
    }
    q.submit(["true"], "reuse-files", 1, 0)
    assert all(p.read_bytes() == content for p, content in snapshot.items())


def test_starved_backlog_job_ages_one_step(q, monkeypatch):
    now = 100_000.0
    monkeypatch.setattr(q.time, "time", lambda: now)
    old = dict(
        id="old", state="pending", priority=-1, submitted=now - q.AGE_S - 1, env={}
    )
    deep = dict(
        id="deep", state="pending", priority=-3, submitted=now - 10 * q.AGE_S, env={}
    )
    fresh = dict(id="fresh", state="pending", priority=0, submitted=now - 5, env={})
    # p-1 that waited AGE_S competes with p0 (older first); deep backlog only rises one step.
    assert q._eff_priority(old, now) == 0
    assert q._eff_priority(deep, now) == -2
    assert q._pick([old, deep, fresh])["id"] == "old"
    young = dict(old, id="young", submitted=now - 60)
    assert q._pick([young, fresh])["id"] == "fresh"


def test_aged_backlog_job_does_not_preempt_running_backlog(q):
    now = 100_000.0
    aged = dict(
        id="aged", state="pending", priority=-1, submitted=now - q.AGE_S - 1, env={}
    )
    interactive = dict(id="p0", state="pending", priority=0, submitted=now, env={})
    assert q._eff_priority(aged, now) == 0
    assert [j["id"] for j in q._preempting([aged, interactive])] == ["p0"]


def test_aged_running_backlog_job_is_not_preempted(q, monkeypatch):
    now = 100_000.0
    monkeypatch.setattr(q.time, "time", lambda: now)
    aged = dict(
        id="aged",
        state="running",
        priority=-1,
        submitted=now - q.AGE_S - 1,
        started=now - 60,
        pid=1,
        env={},
    )
    fresh = dict(
        id="fresh", state="running", priority=-1, submitted=now - 60, pid=2, env={}
    )
    assert q._eff_priority(aged, now) == 0
    assert q._eff_priority(fresh, now) == -1
    q._write(q.JOBS / "aged.json", aged)
    q.submit(["true"], "interactive", 1, 0)
    pauser = q.Pauser(aged, q.JOBS / "aged.json")
    monkeypatch.setattr(pauser, "_signal", lambda sig: None)
    assert q._priority_step(pauser, q.ServingGate(), now) is False
    assert not pauser.paused


def test_priority_caps_demote_secondary_labels(q):
    (q.ROOT / "priority_caps.json").write_text('{"bigmoe-": -2, "mm-": -9}')
    job(q, "a", label="bigmoe-quality", state="pending", priority=0, submitted=1)
    job(q, "b", label="wide4-timing", state="pending", priority=0, submitted=2)
    job(q, "c", label="mm-smoke", state="pending", priority=-9, submitted=3)
    jobs = {j["id"]: j for j in q._jobs()}
    assert jobs["a"]["priority"] == -2 and jobs["a"]["priority_requested"] == 0
    assert jobs["b"]["priority"] == 0 and "priority_requested" not in jobs["b"]
    assert jobs["c"]["priority"] == -9 and "priority_requested" not in jobs["c"]
    assert q._pick(q._jobs())["id"] == "b"


def test_missing_priority_caps_change_nothing(q):
    job(q, "a", label="bigmoe-quality", state="pending", priority=0)
    assert q._jobs()[0]["priority"] == 0


def test_aged_running_backlog_job_yields_after_its_slice(q, monkeypatch):
    monkeypatch.setattr(q, "free_memory_gb", lambda: 100.0)
    now = 100_000.0
    monkeypatch.setattr(q.time, "time", lambda: now)
    aged = dict(
        id="aged",
        state="running",
        priority=-1,
        submitted=now - q.AGE_S - 1,
        started=now - q.AGED_SLICE_S - 1,
        pid=1,
        env={},
    )
    q._write(q.JOBS / "aged.json", aged)
    q.submit(["true"], "interactive", 1, 0)
    pauser = q.Pauser(aged, q.JOBS / "aged.json")
    monkeypatch.setattr(pauser, "_signal", lambda sig: None)
    monkeypatch.setattr(q, "_execute", lambda *a, **k: None)
    q._priority_step(pauser, q.ServingGate(), now)
    assert pauser.paused
    # A resume restarts the slice.
    aged2 = dict(aged, id="aged2", pauses=[[now - 600, now - 60]])
    q._write(q.JOBS / "aged2.json", aged2)
    pauser2 = q.Pauser(aged2, q.JOBS / "aged2.json")
    monkeypatch.setattr(pauser2, "_signal", lambda sig: None)
    assert q._priority_step(pauser2, q.ServingGate(), now) is False
    assert not pauser2.paused


def test_short_checks_run_first_within_a_priority(q):
    import time

    now = time.time()
    big = dict(
        id="big",
        label="wide5-matrix-r3",
        state="pending",
        priority=0,
        submitted=now - 9,
        env={},
    )
    tiny = dict(
        id="tiny",
        label="prefill5-identity-tiny-1933",
        state="pending",
        priority=0,
        submitted=now,
        env={},
    )
    low = dict(
        id="low",
        label="audit-smoke-x",
        state="pending",
        priority=-1,
        submitted=now - 10,
        env={},
    )
    assert q._pick([big, tiny, low])["id"] == "tiny"
    assert q._pick([big, low])["id"] == "big"
    assert not q._is_short(dict(label="tinyllama-bench")) and q._is_short(
        dict(label="x-smoke")
    )


def test_running_p0_job_is_never_paused_for_priority(q, monkeypatch):
    now = 100_000.0
    monkeypatch.setattr(q.time, "time", lambda: now)
    job = dict(
        id="p0",
        state="running",
        priority=0,
        submitted=now - 3 * q.AGE_S,
        started=now - 3 * q.AGED_SLICE_S,
        pid=1,
        env={},
    )
    q._write(q.JOBS / "p0.json", job)
    q.submit(["true"], "x-smoke", 1, 0)
    pauser = q.Pauser(job, q.JOBS / "p0.json")
    monkeypatch.setattr(pauser, "_signal", lambda sig: None)
    monkeypatch.setattr(q, "_execute", lambda *a, **k: None)
    assert q._priority_step(pauser, q.ServingGate(), now) is False
    assert not pauser.paused


def test_exited_jobs_are_adopted_before_live_ones(q, monkeypatch):
    monkeypatch.setattr(q, "_alive", lambda pid: pid == 1)
    jobs = [
        dict(id="live", state="running", pid=1),
        dict(id="gone", state="running", pid=2),
        dict(id="queued", state="pending"),
    ]
    assert [j["id"] for j in q._adoption_order(jobs)] == ["gone", "live"]


def test_aging_never_lifts_a_capped_job_over_its_cap(q, monkeypatch):
    now = 100_000.0
    monkeypatch.setattr(q.time, "time", lambda: now)
    (q.ROOT / "priority_caps.json").write_text('{"research-": -1}')
    job(
        q,
        "r",
        label="research-timing",
        state="pending",
        priority=0,
        submitted=now - 3 * q.AGE_S,
    )
    job(q, "core", label="core-timing", state="pending", priority=0, submitted=now - 5)
    jobs = {j["id"]: j for j in q._jobs()}
    assert q._eff_priority(jobs["r"], now) == -1
    assert q._pick(q._jobs())["id"] == "core"


def test_memory_starved_p0_job_stops_preemption(q, monkeypatch):
    # A paused backlog job stays resident; a large p0 job cannot fit beside it.
    # After MEM_RESERVE_S the backlog job resumes instead of yielding to smaller
    # p0 jobs, so it finishes and the large job can run.
    monkeypatch.setattr(q, "free_memory_gb", lambda: 50.0)
    now = 100_000.0
    monkeypatch.setattr(q.time, "time", lambda: now)
    low = dict(
        id="low",
        state="running",
        priority=-1,
        submitted=now - 60,
        started=now - 60,
        pid=1,
        env={},
    )
    q._write(q.JOBS / "low.json", low)
    q.submit(["true"], "big-128k", 1, 0, mem_gb=80.0)
    q.submit(["true"], "small", 1, 0, mem_gb=10.0)
    ran = []
    monkeypatch.setattr(q, "_execute", lambda job, *a, **k: ran.append(job["label"]))
    pauser = q.Pauser(low, q.JOBS / "low.json")
    monkeypatch.setattr(pauser, "_signal", lambda sig: None)
    assert q._priority_step(pauser, q.ServingGate(), now) is True
    assert ran == ["small"]
    # The small job finished; another arrives after the big one starved long enough.
    later = now + q.MEM_RESERVE_S + 1
    monkeypatch.setattr(q.time, "time", lambda: later)
    for j in q._jobs():
        if j.get("label") == "small":
            q._patch_job(q.JOBS / f"{j['id']}.json", state="done", ended=later)
    q.submit(["true"], "small2", 1, 0, mem_gb=10.0)
    assert q._priority_step(pauser, q.ServingGate(), later) is False
    assert ran == ["small"]
