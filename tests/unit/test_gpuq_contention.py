"""Q02: isolated CPU contention samples, admission and benchmark contracts."""

import importlib.util
import io
import json
import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
DEV = REPO / "scripts/dev"
sys.path.insert(0, str(DEV))


@pytest.fixture(autouse=True)
def _fixed_threshold(monkeypatch):
    # The default scales with the machine's core count; these tests pin 150%.
    monkeypatch.setenv("GPUQ_CPU_THRESHOLD", "150")
    monkeypatch.setitem(helper().DEFAULTS, "threshold_pct", 150.0)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def q(tmp_path, monkeypatch):
    m = load("gpuq_cpu_test", DEV / "gpuq.py")
    for name in ("ROOT", "JOBS", "LOGS"):
        path = tmp_path if name == "ROOT" else tmp_path / name.lower()
        path.mkdir(exist_ok=True)
        monkeypatch.setattr(m, name, path)
    monkeypatch.setattr(m, "_ensure_daemon", lambda: None)
    return m


def helper():
    import gpuq_contention

    return gpuq_contention


def sample(cpu, t=0):
    return dict(
        time=t, since=t - 2, foreign_cpu_pct=cpu, load_1m=3, load_5m=2, top_cpu=[]
    )


def test_cpu_sampler_excludes_descendants_and_orphaned_group_members(monkeypatch):
    h = helper()
    outputs = iter(
        [
            "10 1 10 0 90 0:01.00 /bin/job\n11 10 10 0 80 0:02.00 /bin/child\n"
            "12 1 10 0 70 0:03.00 /bin/orphan\n20 1 20 0 25 0:01.00 /app/next dev\n",
            "10 1 10 0 90 0:02.00 /bin/job\n11 10 10 0 80 0:03.00 /bin/child\n"
            "12 1 10 0 70 0:04.00 /bin/orphan\n20 1 20 0 25 0:05.00 /app/next dev\n",
        ]
    )
    monkeypatch.setattr(h, "process_snapshot", lambda: next(outputs))
    monkeypatch.setattr(h.os, "getloadavg", lambda: (4, 3, 2))
    s = h.CpuSampler()
    assert s.sample(10, now=100)["foreign_cpu_pct"] == 25
    row = s.sample(10, now=102)
    assert row["foreign_cpu_pct"] == 200  # interval CPU, not lifetime ps %CPU
    assert row["top_cpu"] == [dict(pid=20, name="next dev", cpu_pct=200)]
    assert (row["load_1m"], row["load_5m"]) == (4, 3)


def test_quiet_gate_requires_continuous_window_and_has_bounded_wait(q):
    h = helper()
    gate = h.QuietGate()
    job = dict(
        priority=0,
        quiet=True,
        env={},
        contention_config=dict(threshold_pct=150, window_s=20, max_wait_s=60),
    )
    assert gate.blocked(job, sample(50, 100), 100)
    assert gate.blocked(job, sample(200, 119), 119)
    assert gate.blocked(job, sample(50, 120), 120)
    for now in range(122, 140, 2):
        assert gate.blocked(job, sample(50, now), now)
    assert not gate.blocked(job, sample(50, 140), 140)
    other = dict(
        priority=0, quiet=True, env={}, contention_config=job["contention_config"]
    )
    assert gate.blocked(other, sample(500, 100), 100)
    for now in range(102, 160, 2):
        assert gate.blocked(other, sample(500, now), now)
    assert not gate.blocked(other, sample(500, 160), 160)
    assert other["quiet_timeout"] is True
    assert not gate.blocked(dict(priority=-1), sample(500, 100), 100)
    assert gate.blocked(dict(priority=-1, quiet=True), sample(500, 100), 100)


def test_submit_quiet_cli_and_configuration(q, monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gpuq",
            "submit",
            "--quiet",
            "--priority",
            "-1",
            "--cpu-threshold",
            "80",
            "--quiet-window",
            "4",
            "--quiet-max-wait",
            "9",
            "--",
            "true",
        ],
    )
    assert q.main() == 0
    data = q._read(q.JOBS / (capsys.readouterr().out.strip() + ".json"))
    assert data["quiet"]
    assert data["contention_config"] == dict(
        threshold_pct=80,
        window_s=4,
        max_wait_s=9,
        sample_s=30,
        foreign_model_cpu_pct=15,
    )


def test_cpu_gate_integrates_serial_and_priority_admission(q, monkeypatch):
    h = helper()
    gate = q.ServingGate()
    gate.cpu = h.QuietGate()
    monkeypatch.setattr(gate.cpu, "sample", lambda now: sample(500, now))
    monkeypatch.setattr(q, "free_memory_gb", lambda: 100)
    job = dict(id="perf", priority=0, quiet=True, mem_gb=0, env={})
    assert q._blocker(job, gate) == "cpu"
    assert q._preempt_blocker(job, gate) == "cpu"


def test_monitor_marks_midrun_contamination_and_keeps_stats(q, monkeypatch):
    h = helper()
    job = dict(id="monitor", state="running", priority=0, quiet=True, env={}, pid=123)
    path = q.JOBS / "monitor.json"
    q._write(path, {**job, "cancel": True})
    monitor = h.ContentionMonitor(job, path, q.LOGS, q._write, q._read)
    rows = iter([sample(50, 100), sample(250, 130), sample(20, 160)])
    monkeypatch.setattr(monitor.sampler, "sample", lambda *a, **kw: next(rows))
    monitor.record("start", 100)
    monitor.record("periodic", 130)
    monitor.record("end", 160)
    data = q._read(path)
    assert data["cancel"]
    assert job["contended"]
    assert job["foreign_cpu_max_pct"] == 250
    assert job["foreign_cpu_mean_pct"] == pytest.approx(320 / 3)
    assert [r["phase"] for r in data["cpu_samples"]] == ["start", "periodic", "end"]
    flag = json.loads((q.LOGS / "monitor.contention.json").read_text())
    assert flag["contended"] and flag["events"] == [[128, 130]]
    assert "foreign CPU" in monitor.summary()


def test_runner_exports_flag_and_logs_start_end_without_touching_daemon(q, monkeypatch):
    h = helper()
    monkeypatch.setattr(h.CpuSampler, "sample", lambda *a, **kw: sample(200))
    jid = q.submit(
        [
            sys.executable,
            "-c",
            "import os,json; print(json.load(open(os.environ['GPUQ_CONTENTION_FILE']))['contended'])",
        ],
        "cpu",
        1,
        0,
    )
    path = q.JOBS / (jid + ".json")
    job = q._read(path)
    q._run_one(job, path)
    assert job["contended"] and job["state"] == "done" and job["rc"] == 0
    assert [r["phase"] for r in job["cpu_samples"]] == ["start", "end"]
    log = (q.LOGS / (jid + ".log")).read_text()
    assert "True" in log and "foreign CPU" in log


def test_wait_contended_perf_is_separate_from_failure(q, capsys):
    for jid, kw in [
        ("perf", dict(priority=0, quiet=True)),
        ("audit", dict(priority=-1)),
        ("quiet", dict(priority=-1, quiet=True)),
    ]:
        q._write(
            q.JOBS / (jid + ".json"),
            dict(id=jid, state="done", rc=0, contended=True, **kw),
        )
    assert q.wait(["perf"], 0) == 3
    assert "perf: contended rc=0" in capsys.readouterr().out
    assert q.wait(["audit"], 0) == 0
    assert q.wait(["quiet"], 0) == 3
    q._patch_job(q.JOBS / "perf.json", rc=1, state="failed")
    assert q.wait(["perf"], 0) == 1


def test_digest_contended_is_distinct_and_flagged_for_perf(q):
    gd = load("digest_cpu_test", DEV / "gpuq_digest.py")
    for jid, priority in [("perf", 0), ("audit", -1)]:
        q._write(
            q.JOBS / (jid + ".json"),
            dict(
                id=jid,
                label=jid,
                state="done",
                rc=0,
                ended=10,
                started=1,
                contended=True,
                priority=priority,
            ),
        )
    res = gd.collect(q.ROOT, 0, 20)
    assert res["families"]["perf"][0]["state"] == "contended"
    assert res["families"]["audit"][0]["state"] == "contended"
    assert [e["id"] for e in res["problems"]] == ["perf"]


def test_contention_reader_uses_attempt_window_and_fails_closed(tmp_path, monkeypatch):
    h = helper()
    flag = tmp_path / "cpu.json"
    monkeypatch.setenv("GPUQ_CONTENTION_FILE", str(flag))
    flag.write_text(json.dumps(dict(contended=True, events=[[100, 120]])))
    assert h.was_contended(110, 130)
    assert not h.was_contended(121, 130)
    assert h.was_contended()
    flag.write_text("invalid")
    assert h.was_contended(121, 130)
    monkeypatch.delenv("GPUQ_CONTENTION_FILE")
    assert not h.was_contended()


def test_tfbench_and_gate_rows_include_contention(tmp_path, monkeypatch):
    flag = tmp_path / "cpu.json"
    flag.write_text(json.dumps(dict(contended=True, events=[[0, 1e12]])))
    monkeypatch.setenv("GPUQ_CONTENTION_FILE", str(flag))
    tf = load("tf_cpu_test", REPO / "scripts/research/tfbench.py")
    out = io.StringIO()
    tf.emit(out, kind="decode", tps=90)
    assert json.loads(out.getvalue())["contended"] is True
    gc = load("gate_cpu_test", REPO / "scripts/release/gate_checks.py")
    results = tmp_path / "gate.jsonl"
    gc.Results(results).add("check", "PASS")
    assert json.loads(results.read_text())["contended"] is True


@pytest.mark.parametrize(
    "rows,expected",
    [
        (
            [
                dict(status="FAIL", same_text=True, contended=True),
                dict(status="PASS", same_text=True, contended=False),
            ],
            "PASS",
        ),
        (
            [
                dict(status="FAIL", same_text=True, contended=True),
                dict(status="FAIL", same_text=True, contended=False),
            ],
            "FAIL",
        ),
        ([dict(status="FAIL", same_text=True, contended=True)] * 2, "CONTENDED"),
        ([dict(status="FAIL", same_text=False, contended=True)], "FAIL"),
        ([dict(status="ERROR", contended=True)], "FAIL"),
    ],
)
def test_gate_retries_contended_ratio_once(tmp_path, rows, expected):
    h = helper()
    calls = []

    def run():
        calls.append(1)
        return rows[len(calls) - 1]

    result = h.server_path_attempts(run)
    assert result["status"] == expected
    assert len(calls) == len(rows)
    assert len(result["attempts"]) == len(rows)


def test_max_wait_is_visible_in_record_even_if_instantaneous_cpu_drops(q, monkeypatch):
    h = helper()
    jid = q.submit(["true"], "maxwait", 1, 0, quiet=True)
    path = q.JOBS / (jid + ".json")
    job = q._read(path)
    job["quiet_timeout"] = True
    monkeypatch.setattr(h.CpuSampler, "sample", lambda *a, **kw: sample(20, 100))
    q._run_one(job, path)
    assert job["contended"]
    assert q.wait([jid], 0) == 3


def test_daemon_persists_quiet_window_across_job_reload(q, monkeypatch):
    h = helper()
    jid = q.submit(["true"], "admission", 1, 0, quiet=True, cpu_config=dict(window_s=4))
    path = q.JOBS / (jid + ".json")
    ticks = [100.0]
    gate = q.ServingGate()
    gate.cpu = h.QuietGate()
    monkeypatch.setattr(q, "_now", lambda: ticks[0])
    monkeypatch.setattr(gate.cpu, "sample", lambda now: sample(20, now))
    assert q._blocker(q._read(path), gate) == "cpu"
    ticks[0] = 103
    assert q._blocker(q._read(path), gate) == "cpu"
    ticks[0] = 104
    job = q._read(path)
    assert q._blocker(job, gate) is None
    assert job["quiet_since"] == 100


def test_cpu_sampling_failure_is_contended_and_does_not_wedge_queue(q, monkeypatch):
    h = helper()

    def fail():
        raise OSError("ps unavailable")

    monkeypatch.setattr(h, "process_snapshot", fail)
    jid = q.submit(["true"], "unknowncpu", 1, 0)
    job = q._read(q.JOBS / (jid + ".json"))
    q._run_one(job, q.JOBS / (jid + ".json"))
    assert job["state"] == "done" and job["contended"]
    assert job["foreign_cpu_mean_pct"] is None
    assert "ps unavailable" in job["cpu_samples"][0]["error"]


def test_gate_summary_distinguishes_contended_from_failure(tmp_path, capsys):
    from types import SimpleNamespace

    gc = load("gate_summary_cpu_test", REPO / "scripts/release/gate_checks.py")
    path = tmp_path / "rows.jsonl"
    rows = gc.Results(path)
    rows.add("ratio", "CONTENDED", contended=True)
    assert gc.summary(SimpleNamespace(results=path)) == 3
    assert "GATE: CONTENDED" in capsys.readouterr().out
    rows.add("text", "FAIL", contended=True)
    assert gc.summary(SimpleNamespace(results=path)) == 1


def test_checker_cli_runs_retry_and_preserves_both_attempts(
    tmp_path, monkeypatch, capsys
):
    checker = load(
        "server_path_cpu_test", REPO / "scripts/release/check_server_path.py"
    )
    rows = iter(
        [
            dict(status="FAIL", same_text=True, contended=True),
            dict(status="PASS", same_text=True, contended=False),
        ]
    )
    calls = []
    monkeypatch.setattr(checker, "measure", lambda _: next(rows))
    monkeypatch.setattr(checker, "wait_for_quiet", lambda pid: calls.append(pid))
    path = tmp_path / "result.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "check",
            "--url",
            "http://localhost:18990",
            "--model",
            "fake",
            "--retry-contended",
            "--output",
            str(path),
        ],
    )
    assert checker.main() == 0
    summary = json.loads(path.read_text())
    assert summary["status"] == "PASS" and len(summary["attempts"]) == 2
    assert calls == [os.getpgrp()]
    assert json.loads(capsys.readouterr().out)["contended"] is False


def test_admission_sample_does_not_include_prestart_contention(q, monkeypatch):
    h = helper()
    jid = q.submit(["true"], "quiet-start", 1, 0)
    path = q.JOBS / (jid + ".json")
    job = q._read(path)
    # Admission's interval was quiet after earlier CPU saturation. ps's first
    # estimate alone still includes that old load; don't contaminate the start.
    job["cpu_admission_sample"] = sample(20, 100)
    rows = iter([sample(250, 100), sample(20, 101)])
    monkeypatch.setattr(h.CpuSampler, "sample", lambda *a, **kw: next(rows))
    q._run_one(job, path)
    assert not job["contended"]
    assert job["cpu_samples"][0]["foreign_cpu_pct"] == 20


def test_sampler_does_not_hide_foreign_reuse_of_an_exited_child_pid(monkeypatch):
    h = helper()
    outputs = iter(
        [
            "10 1 10 0 0 0:00.00 /bin/job\n11 10 10 0 80 0:01.00 /bin/child\n",
            "10 1 10 0 0 0:00.00 /bin/job\n",
            "10 1 10 0 0 0:00.00 /bin/job\n11 1 11 0 80 0:01.00 /bin/next\n",
        ]
    )
    monkeypatch.setattr(h, "process_snapshot", lambda: next(outputs))
    sampler = h.CpuSampler()
    assert sampler.sample(10, 100)["foreign_cpu_pct"] == 0
    assert sampler.sample(10, 102)["foreign_cpu_pct"] == 0
    assert sampler.sample(10, 104)["foreign_cpu_pct"] == 50


def test_runner_exports_saved_cpu_threshold_to_harness(q, monkeypatch):
    h = helper()
    monkeypatch.setattr(h.CpuSampler, "sample", lambda *a, **kw: sample(20, 100))
    jid = q.submit(
        [
            sys.executable,
            "-c",
            "import os; print(os.environ['GPUQ_CPU_THRESHOLD'], os.environ['GPUQ_QUIET_WINDOW_S'])",
        ],
        "cfg-env",
        1,
        0,
        cpu_config=dict(threshold_pct=55, window_s=8),
    )
    path = q.JOBS / (jid + ".json")
    q._run_one(q._read(path), path)
    assert "55 8" in (q.LOGS / (jid + ".log")).read_text()


def test_adopted_pre_q02_job_is_untrustworthy_without_restarting_live_daemon(
    q, monkeypatch
):
    h = helper()
    monkeypatch.setattr(h.CpuSampler, "sample", lambda *a, **kw: sample(20, 100))
    monkeypatch.setattr(q, "_alive", lambda _: False)
    jid = "pre-q02"
    path = q.JOBS / (jid + ".json")
    job = dict(id=jid, pid=123, state="running", started=90, priority=0, quiet=True)
    q._write(path, job)
    (q.LOGS / (jid + ".rc")).write_text("0")
    q._adopt(job, path)
    assert job["adopted"] and job["contended"] and job["state"] == "done"
    assert "unmonitored_before_adoption" in job["contention_reasons"]
    assert q.wait([jid], 0) == 3


def test_top_cpu_truncation_does_not_truncate_foreign_cpu_sum(monkeypatch):
    h = helper()
    monkeypatch.setattr(
        h,
        "process_snapshot",
        lambda: "\n".join(
            f"{p} 1 {p} 0 20 0:01.00 /bin/process" for p in range(10, 22)
        ),
    )
    row = h.CpuSampler().sample(now=100)
    assert row["foreign_cpu_pct"] == 240
    assert len(row["top_cpu"]) == 8


def test_quiet_window_restarts_after_unobserved_scheduling_gap():
    h = helper()
    gate = h.QuietGate()
    job = dict(priority=0, quiet=True, env={})
    assert gate.blocked(job, sample(20, 100), 100)
    assert gate.blocked(job, sample(20, 102), 102)
    # A long higher-priority job ran. Two quiet endpoints don't prove that the
    # machine stayed quiet in between; collect a new complete quiet window.
    assert gate.blocked(job, sample(20, 130), 130)
    for now in range(132, 150, 2):
        assert gate.blocked(job, sample(20, now), now)
    assert not gate.blocked(job, sample(20, 150), 150)


def test_stale_contention_file_fails_closed_if_daemon_stops(tmp_path, monkeypatch):
    h = helper()
    path = tmp_path / "flag.json"
    monkeypatch.setenv("GPUQ_CONTENTION_FILE", str(path))
    monkeypatch.setattr(h.time, "time", lambda: 130)
    path.write_text(
        json.dumps(dict(contended=False, events=[], last_sample=dict(time=100)))
    )
    assert h.was_contended(120, 130)
    path.write_text(
        json.dumps(dict(contended=False, events=[], last_sample=dict(time=128)))
    )
    assert not h.was_contended(120, 130)


def test_gate_shell_uses_final_attempt_status_not_pass_in_earlier_attempt(tmp_path):
    import subprocess

    source = (REPO / "scripts/release/gate.sh").read_text()
    start = source.index("      sps=$(")
    end = source.index("      esac", start) + len("      esac")
    body = source[start:end]
    assert "--retry-contended" in source
    for status in ("PASS", "FAIL", "CONTENDED"):
        summary = dict(
            status=status,
            contended=status == "CONTENDED",
            attempts=[dict(status="PASS", contended=True)],
        )
        proc = subprocess.run(
            ["zsh", "-c", 'rec(){ print -r -- "$*"; };\n' + body],
            env={
                **os.environ,
                "PY": sys.executable,
                "sp": json.dumps(summary),
                "spd": "ratio",
            },
            capture_output=True,
            text=True,
            check=True,
        )
        assert proc.stdout.strip().startswith("serve-27b.server_path " + status + " ")


def test_adoption_marks_a_gap_even_when_old_cpu_samples_exist(q, monkeypatch):
    h = helper()
    jid = "old-monitor"
    path = q.JOBS / (jid + ".json")
    job = dict(
        id=jid,
        pid=123,
        state="running",
        priority=0,
        quiet=True,
        started=90,
        cpu_samples=[sample(20, 100)],
    )
    q._write(path, job)
    (q.LOGS / (jid + ".rc")).write_text("0")
    (q.LOGS / (jid + ".contention.json")).write_text(
        json.dumps(dict(contended=False, events=[], last_sample=sample(20, 100)))
    )
    monkeypatch.setattr(q, "_alive", lambda _: False)
    monkeypatch.setattr(q, "_now", lambda: 130)
    monkeypatch.setattr(h.CpuSampler, "sample", lambda *a, **kw: sample(20, 130))
    q._adopt(job, path)
    assert job["contended"]
    assert "unmonitored_before_adoption" in job["contention_reasons"]


@pytest.mark.parametrize("payload", ["null", "[]", '{"last_sample": null}'])
def test_structurally_invalid_flag_fails_closed(tmp_path, payload):
    h = helper()
    path = tmp_path / "invalid.json"
    path.write_text(payload)
    assert h.was_contended(0, 1, path=str(path))


@pytest.mark.parametrize("payload", ["null", "[]", '{"last_sample": null}'])
def test_adoption_survives_structurally_invalid_flag(q, monkeypatch, payload):
    h = helper()
    jid = "invalid-flag"
    path = q.JOBS / (jid + ".json")
    job = dict(
        id=jid,
        state="running",
        pid=123,
        priority=0,
        quiet=True,
        started=90,
        cpu_samples=[sample(20, 100)],
    )
    q._write(path, job)
    (q.LOGS / (jid + ".rc")).write_text("0")
    (q.LOGS / (jid + ".contention.json")).write_text(payload)
    monkeypatch.setattr(q, "_alive", lambda _: False)
    monkeypatch.setattr(h.CpuSampler, "sample", lambda *a, **kw: sample(20, 130))
    q._adopt(job, path)
    assert job["contended"] and job["state"] == "done"


def test_only_timing_jobs_wait_for_a_quiet_cpu():
    from gpuq_contention import requires_quiet

    # A correctness / smoke job at any priority starts at once; contention is
    # still recorded for it, it just doesn't block admission.
    assert not requires_quiet(dict(priority=1))
    assert not requires_quiet(dict(priority=0))
    assert requires_quiet(dict(priority=0, quiet=True))
    assert requires_quiet(dict(priority=-1, quiet=True))


@pytest.mark.parametrize(
    "cpu,window,max_wait,cap,expected,timeout",
    [
        (20, 4, 60, 300, 4, False),
        (500, 20, 6, 300, 6, True),
        (500, 20, 1800, 8, 8, True),
        (None, 20, 1800, 8, 8, True),
        (500, 20, 1800, 300, 300, True),
    ],
)
def test_daemon_reserves_quiet_winner_instead_of_starting_nonquiet_work(
    q, monkeypatch, cpu, window, max_wait, cap, expected, timeout
):
    ticks = [100.0]
    monkeypatch.setattr(q, "_now", lambda: ticks[0])
    monkeypatch.setattr(q.time, "time", lambda: ticks[0])
    monkeypatch.setenv("GPUQ_QUIET_HOLD_MAX_S", str(cap))
    quiet = q.submit(
        ["true"],
        "quiet-winner",
        1,
        0,
        quiet=True,
        cpu_config=dict(window_s=window, max_wait_s=max_wait),
    )
    ticks[0] += 1
    regular = q.submit(["true"], "regular", 1, 0)
    ticks[0] = 100
    monkeypatch.setattr(
        helper().QuietGate, "sample", lambda self, now: sample(cpu, now)
    )
    monkeypatch.setattr(
        q.time, "sleep", lambda seconds: ticks.__setitem__(0, ticks[0] + seconds)
    )

    class Finished(BaseException):
        pass

    def execute(job, path, gate):
        assert job["id"] == quiet
        assert ticks[0] == 100 + expected
        assert job.get("quiet_timeout", False) is timeout
        assert q._read(q.JOBS / (regular + ".json"))["state"] == "pending"
        # Even a timeout at an instantaneously quiet start must remain contended.
        monkeypatch.setattr(
            helper().CpuSampler, "sample", lambda *a, **kw: sample(20, ticks[0])
        )
        q._run_one(job, path, gate)
        assert job["contended"] is timeout
        raise Finished

    monkeypatch.setattr(q, "_execute", execute)
    with pytest.raises(Finished):
        q.daemon()


def test_reserved_quiet_deadline_survives_poll_gap():
    gate = helper().QuietGate()
    job = dict(quiet=True, env={}, quiet_hold_started=100, quiet_hold_limit_s=8)
    assert gate.blocked(job, sample(500, 100), 100)
    assert not gate.blocked(job, sample(500, 110), 110)
    assert job["quiet_timeout"] is True


def test_cpu_sampler_excludes_nice_ten_and_background_builds(monkeypatch):
    outputs = iter(
        [
            "20 1 20 9 25 0:01.00 /bin/foreground\n"
            "21 1 21 10 300 0:01.00 /bin/build\n"
            "22 1 22 15 400 0:01.00 /bin/pytest\n",
            "20 1 20 9 25 0:03.00 /bin/foreground\n"
            "21 1 21 10 300 0:07.00 /bin/build\n"
            "22 1 22 15 400 0:09.00 /bin/pytest\n",
        ]
    )
    monkeypatch.setattr(helper(), "process_snapshot", lambda: next(outputs))
    sampler = helper().CpuSampler()
    assert sampler.sample(now=100)["foreign_cpu_pct"] == 25
    row = sampler.sample(now=102)
    assert row["foreign_cpu_pct"] == 100
    assert [p["pid"] for p in row["top_cpu"]] == [20]


def test_status_reports_quiet_hold_duration_limit_and_reason(q, monkeypatch, capsys):
    jid = q.submit(["true"], "held", 1, 0, quiet=True)
    q._patch_job(
        q.JOBS / (jid + ".json"),
        waiting="cpu",
        quiet_hold_started=100,
        quiet_hold_limit_s=300,
        quiet_hold_reason="foreign CPU 500.0% >= 150.0%",
    )
    monkeypatch.setattr(q, "_now", lambda: 142)
    monkeypatch.setattr(q, "_daemon_running", lambda: True)
    q.status()
    text = capsys.readouterr().out
    assert "held=42/300s" in text
    assert "foreign CPU 500.0% >= 150.0%" in text


def test_priority_pause_reserves_quiet_winner(q, monkeypatch):
    ticks = [100.0]
    monkeypatch.setattr(q, "_now", lambda: ticks[0])
    monkeypatch.setattr(q.time, "time", lambda: ticks[0])
    monkeypatch.setattr(q, "free_memory_gb", lambda: 100)
    monkeypatch.setattr(q.os, "killpg", lambda *a: None)
    low = q.submit(["true"], "low", 1, -1)
    lp = q.JOBS / (low + ".json")
    q._patch_job(lp, state="running", pid=4242, started=90)
    quiet = q.submit(["true"], "quiet", 1, 0, quiet=True, cpu_config=dict(max_wait_s=6))
    q.submit(["true"], "regular", 1, 0)
    gate = q.ServingGate()
    gate.cpu = helper().QuietGate()
    monkeypatch.setattr(gate.cpu, "sample", lambda now: sample(500, now))
    executed = []
    monkeypatch.setattr(q, "_execute", lambda job, *a: executed.append(job))
    pauser = q.Pauser(q._read(lp), lp)
    for now in (100, 102, 104):
        ticks[0] = now
        assert q._priority_step(pauser, gate, now)
        assert pauser.paused and not executed
    ticks[0] = 106
    assert q._priority_step(pauser, gate, 106)
    assert [j["id"] for j in executed] == [quiet]
    assert executed[0]["quiet_timeout"]


def test_quiet_reservation_follows_owner_fairness_and_releases_for_higher_priority(
    q, monkeypatch
):
    ticks = [100.0]
    monkeypatch.setattr(q, "_now", lambda: ticks[0])
    monkeypatch.setattr(q.time, "time", lambda: ticks[0])
    quiet = q.submit(["true"], "quiet", 1, 0, quiet=True)
    regular = q.submit(["true"], "regular", 1, 0)
    q._patch_job(
        q.JOBS / (quiet + ".json"), cwd="/tmp", env=dict(GPUQ_OWNER="quiet-owner")
    )
    q._patch_job(
        q.JOBS / (regular + ".json"),
        cwd="/tmp",
        env=dict(GPUQ_OWNER="recent-owner"),
        submitted=99,
    )
    q._write(
        q.JOBS / "history.json",
        dict(
            id="history",
            state="done",
            submitted=1,
            started=98,
            cwd="/tmp",
            env=dict(GPUQ_OWNER="recent-owner"),
        ),
    )
    gate = q.ServingGate()
    gate.cpu = helper().QuietGate()
    monkeypatch.setattr(gate.cpu, "sample", lambda now: sample(500, now))
    assert q._admit(q._jobs(), gate) is None
    path = q.JOBS / (quiet + ".json")
    assert q._read(path)["quiet_hold_started"] == 100
    ticks[0] = 102
    urgent = q.submit(["true"], "urgent", 1, 1)
    assert q._admit(q._jobs(), gate)["id"] == urgent
    held = q._read(path)
    assert held["quiet_hold_started"] is None
    assert held["quiet_hold_elapsed_s"] == 2
    assert held["waiting"] == "slot"


def test_default_threshold_scales_with_cores(monkeypatch):
    import importlib

    h = helper()
    monkeypatch.setattr(h.os, "cpu_count", lambda: 18)
    monkeypatch.delenv("GPUQ_CPU_THRESHOLD", raising=False)
    fresh = importlib.reload(h)
    try:
        assert fresh.DEFAULTS["threshold_pct"] == 1620.0
    finally:
        monkeypatch.undo()
        importlib.reload(fresh)


# ---- foreign model processes (servers started outside gpuq) ----


def _fake_ps(monkeypatch, h, snaps, args):
    it = iter(snaps)
    monkeypatch.setattr(h, "process_snapshot", lambda: next(it))
    monkeypatch.setattr(h, "process_args", lambda: args)
    monkeypatch.setattr(h.os, "getloadavg", lambda: (1, 1, 1))


SERVE = "python -m yunshu_cli serve --port 8000 --models-dir /tmp/m"
ARGS = {
    10: "python /app/bench.py",
    30: SERVE,
    31: "uvicorn yunshu_gateway.app:app",
    32: "/usr/bin/vim notes.txt",
    33: "python -m mlx_lm.server --model x",
    34: "python -m yunshu_cli serve --port 1",  # nice 10: excluded
    35: "python gpuq.py run -- python -m yunshu_cli serve",  # gpuq client
}


def _rows(t30, t31=0, t33=0):
    return (
        "10 1 10 0 5 0:01.00 python\n"
        f"30 1 30 0 0 0:{t30:05.2f} python\n31 1 31 0 0 0:{t31:05.2f} python\n"
        "32 1 32 0 0 0:00.00 vim\n"
        f"33 1 33 0 0 0:{t33:05.2f} python\n34 1 34 10 0 0:00.00 python\n"
        "35 1 35 0 0 0:00.00 python\n"
    )


def test_foreign_model_listed_but_idle_does_not_latch(monkeypatch, q):
    h = helper()
    _fake_ps(monkeypatch, h, [_rows(0)] * 3, ARGS)
    job = dict(id="j", quiet=True, env={})
    path = q.JOBS / "j.json"
    q._write(path, job)
    m = h.ContentionMonitor(job, path, q.LOGS, q._write, q._read)
    m.record("start", now=100)
    m.record("poll", now=102)
    assert not job.get("contended")
    pids = {x["pid"] for x in job["foreign_model_processes"]}
    assert pids == {30, 31, 33}  # not vim, nice>=10, gpuq client, or the job
    assert all(x["max_cpu_pct"] < 15 for x in job["foreign_model_processes"])


def test_busy_foreign_model_latches_with_reason(monkeypatch, q):
    h = helper()
    _fake_ps(monkeypatch, h, [_rows(0), _rows(1.0)], ARGS)
    job = dict(id="j", quiet=True, env={})
    path = q.JOBS / "j.json"
    q._write(path, job)
    m = h.ContentionMonitor(job, path, q.LOGS, q._write, q._read)
    m.record("start", now=100)
    assert not job.get("contended")
    m.record("poll", now=102)  # 1s cpu in 2s = 50%
    assert job["contended"]
    reason = [r for r in job["contention_reasons"] if r.startswith("foreign model")]
    assert reason == [f"foreign model process 30 busy 50.0% ({SERVE})"]
    top = max(job["foreign_model_processes"], key=lambda x: x["max_cpu_pct"])
    assert (top["pid"], top["max_cpu_pct"]) == (30, 50.0)


def test_foreign_model_threshold_env_and_extra_pattern(monkeypatch, q):
    h = helper()
    args = {10: "x", 30: "/opt/mymodel-runner --serve"}
    monkeypatch.setenv("GPUQ_FOREIGN_MODEL_PATTERNS", "mymodel-runner")
    _fake_ps(monkeypatch, h, [_rows(0), _rows(1.0)], args)
    job = dict(id="j", quiet=True, env=dict(os.environ, GPUQ_FOREIGN_MODEL_CPU="80"))
    path = q.JOBS / "j.json"
    q._write(path, job)
    m = h.ContentionMonitor(job, path, q.LOGS, q._write, q._read)
    m.record("start", now=100)
    m.record("poll", now=102)  # 50% < 80%
    assert not job.get("contended")
    assert [x["pid"] for x in job["foreign_model_processes"]] == [30]


def test_busy_foreign_model_blocks_quiet_admission(monkeypatch):
    h = helper()
    _fake_ps(monkeypatch, h, [_rows(0), _rows(1.0), _rows(2.0)], ARGS)
    gate = h.QuietGate()
    job = dict(priority=0, quiet=True, env={})
    gate.blocked(job, gate.sample(100), 100)
    for t in (102, 104):
        assert gate.blocked(job, gate.sample(t), t)  # busy server: never quiet
    assert job["quiet_since"] is None


def test_cli_and_ci_sandbox_commands_are_not_foreign_models():
    """2026-10-09: a 52-min snapshot cell was voided because ci-local's unit tests ran
    `python -m yunshu_cli --json model info` (CPU only) and every yunshu_cli invocation
    matched as a foreign model runtime. Only commands that load a model count."""
    h = helper()
    pats = h.foreign_model_patterns({})

    def hit(cmd):
        return h.is_foreign_model_cmd(cmd, pats)

    assert not hit(
        "/tmp/ycl-2b62cc7e/.venv/bin/python -m yunshu_cli --json model info x"
    )
    assert not hit("/Users/u/Yunshu/.venv/bin/python -m yunshu_cli config")
    assert not hit(
        "/tmp/ycl-abc/.venv/bin/python -m yunshu_cli serve -m tiny"
    )  # CI sandbox
    assert hit("/Users/u/Yunshu/.venv/bin/python -m yunshu_cli serve -m /models/27b")
    assert hit("/opt/homebrew/bin/yunshu serve -m qwen")
    assert hit("python -m mlx_lm.server --model x")
