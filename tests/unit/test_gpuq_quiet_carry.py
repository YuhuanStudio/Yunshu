"""Quiet-window carry-over, scheduler wake-up, hold gap and daemon.log timestamps (gpuq idle gaps)."""

import importlib
import importlib.util
import sys
import types
from pathlib import Path

import pytest

DEV = Path(__file__).resolve().parents[2] / "scripts" / "dev"
sys.path.insert(0, str(DEV))


@pytest.fixture(autouse=True)
def _threshold(monkeypatch):
    import gpuq_contention as h

    monkeypatch.setitem(h.DEFAULTS, "threshold_pct", 150.0)


def contention():
    import gpuq_contention

    return gpuq_contention


class FakeSampler:
    """CpuSampler stand-in driven by a script: cpu(now) -> foreign cpu %, records the pid it was given."""

    def __init__(self, cpu):
        self.cpu = cpu
        self.owned = {99}
        self.pids = []

    def sample(self, pid=None, now=None):
        self.pids.append(pid)
        return dict(
            time=now,
            since=now - 1,
            foreign_cpu_pct=self.cpu(now),
            load_1m=1,
            load_5m=1,
            top_cpu=[],
            foreign_models=[],
        )


def gate_with(cpu):
    gate = contention().QuietGate()
    gate.sampler = FakeSampler(cpu)
    return gate


def quiet_job(window=20, max_wait=300):
    return dict(
        priority=0,
        quiet=True,
        env={},
        contention_config=dict(threshold_pct=150, window_s=window, max_wait_s=max_wait),
    )


def admit_at(gate, job, now):
    return gate.blocked(job, gate.sample(now), now)


def first_pass(gate, job, start, stop=None):
    """Poll every second like the daemon; the first second the job is admitted (None: never)."""
    for t in range(int(start), int(start) + 200 if stop is None else stop):
        if not admit_at(gate, job, float(t)):
            return t
    return None


def test_quiet_started_during_previous_job_counts_at_admission():
    gate = gate_with(lambda now: 10.0)
    for t in range(1000, 1015):  # the lane job runs; observed quiet from t=1000
        gate.observe(float(t), running_pid=4242)
    job = quiet_job()
    # Admission at 1015: 15 s already quiet, so the hold is 5 s, not a fresh 20 s.
    assert first_pass(gate, job, 1015) == 1020


def test_carry_is_zero_hold_when_window_already_full():
    gate = gate_with(lambda now: 0.0)
    for t in range(1000, 1030):
        gate.observe(float(t), running_pid=7)
    assert first_pass(gate, quiet_job(), 1030) == 1030


def test_busy_sample_restarts_the_carried_window():
    gate = gate_with(lambda now: 400.0 if 1010 <= now < 1011 else 5.0)
    for t in range(1000, 1015):
        gate.observe(float(t), running_pid=7)
    job = quiet_job()
    assert first_pass(gate, job, 1015) == 1031  # quiet only since 1011


def test_sampling_gap_breaks_the_carry():
    gate = gate_with(lambda now: 5.0)
    for t in (1000, 1001, 1002):
        gate.observe(float(t), running_pid=7)
    for t in range(1010, 1015):  # 8 s without samples > MAX_POLL_GAP_S
        gate.observe(float(t), running_pid=7)
    job = quiet_job()
    assert first_pass(gate, job, 1015) == 1030  # quiet only since 1010


def test_busy_foreign_model_breaks_the_carry():
    gate = gate_with(lambda now: 5.0)
    orig = gate.sampler.sample

    def with_model(pid=None, now=None):
        row = orig(pid=pid, now=now)
        if now == 1010.0:
            row["foreign_models"] = [dict(pid=1, cmd="mlx_lm.server", cpu_pct=80.0)]
        return row

    gate.sampler.sample = with_model
    for t in range(1000, 1015):
        gate.observe(float(t), running_pid=7)
    assert (
        first_pass(gate, quiet_job(), 1015) == 1031
    )  # busy at 1010, quiet again from 1011


def test_same_thresholds_apply_to_carried_samples():
    gate = gate_with(
        lambda now: 100.0
    )  # quiet for threshold 150, busy for threshold 50
    for t in range(1000, 1030):
        gate.observe(float(t), running_pid=7)
    strict = quiet_job()
    strict["contention_config"]["threshold_pct"] = 50
    assert admit_at(
        gate, strict, 1030.0
    )  # nothing carried under the stricter threshold
    assert not admit_at(gate, quiet_job(), 1030.0)


def test_running_job_tree_is_excluded_and_leftovers_are_not():
    gate = gate_with(lambda now: 1.0)
    gate.observe(1000.0, running_pid=4242)
    assert gate.sampler.pids[-1] == 4242
    assert gate.sampler.owned == {99}  # owned kept while a lane job runs
    gate.observe(1002.0, running_pid=None)
    assert gate.sampler.pids[-1] is None
    assert (
        gate.sampler.owned == set()
    )  # the finished job's leftovers count as foreign again


def test_unobserved_gate_behaves_as_before():
    gate = gate_with(lambda now: 5.0)
    job = quiet_job()
    assert first_pass(gate, job, 2000) == 2020


# ---- scheduler wake-up, hold gap, daemon.log ------------------------------------------------------


@pytest.fixture
def q(tmp_path, monkeypatch):
    monkeypatch.setenv("GPUQ_DIR", str(tmp_path))
    spec = importlib.util.spec_from_file_location("gpuq_carry_test", DEV / "gpuq.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    m.time = types.SimpleNamespace(**vars(m.time))
    for name in ("ROOT", "JOBS", "LOGS"):
        path = tmp_path if name == "ROOT" else tmp_path / name.lower()
        path.mkdir(exist_ok=True)
        monkeypatch.setattr(m, name, path)
    monkeypatch.setattr(m, "_ensure_daemon", lambda: None)
    m.WAKE.clear()
    return m


def test_wake_event_ends_the_sleep_at_once(q):
    slept = []
    q.time.sleep = lambda s: slept.append(s)
    q.WAKE.set()
    q._wake_wait(2.0, q._wake_mtime())
    assert slept == [] and not q.WAKE.is_set()


def test_submit_pokes_the_wake_file_and_ends_the_sleep(q):
    seen = q._wake_mtime()
    q.submit(["true"], "x", 1, 0)
    assert q._wake_mtime() != seen
    slept = []
    q.time.sleep = lambda s: slept.append(s)
    q._wake_wait(2.0, seen)
    assert slept == []


def test_poll_is_the_fallback_when_nothing_wakes_the_daemon(q):
    slept = []
    q.time.sleep = lambda s: slept.append(s)
    q._wake_wait(q.POLL_S, q._wake_mtime())
    assert sum(slept) == q.POLL_S and len(slept) > 1


def test_hold_gap_is_30_s_and_configurable(tmp_path, monkeypatch):
    monkeypatch.setenv("GPUQ_DIR", str(tmp_path))
    monkeypatch.delenv("GPUQ_HOLD_GAP_S", raising=False)
    import gpuq

    assert importlib.reload(gpuq).HOLD_GAP_S == 30
    monkeypatch.setenv("GPUQ_HOLD_GAP_S", "7")
    assert importlib.reload(gpuq).HOLD_GAP_S == 7
    monkeypatch.delenv("GPUQ_HOLD_GAP_S")
    importlib.reload(gpuq)


def mk(i, prio=0, quiet=False, device="m5", state="pending", **kw):
    return {
        "id": i,
        "label": i,
        "state": state,
        "submitted": 1000.0,
        "priority": prio,
        "timeout_s": 3600,
        "cwd": "/x/a",
        "device": device,
        "quiet": quiet,
        "env": {},
        "contention_config": dict(threshold_pct=150, window_s=20, max_wait_s=300),
        **kw,
    }


@pytest.mark.parametrize("prio", [-1, 0, 1])
def test_admit_carry_across_priorities_and_logs(q, monkeypatch, capsys, prio):
    now = [1015.0]
    monkeypatch.setattr(q, "_now", lambda: now[0])
    monkeypatch.setattr(q, "_patch_job", lambda *a, **k: None)
    gate = q.ServingGate()
    gate.cpu = q.QuietGate()
    gate.cpu.sampler = FakeSampler(lambda t: 3.0)
    for t in range(1000, 1015):
        gate.cpu.observe(float(t), running_pid=5)
    quiet = mk("q", prio=prio, quiet=True)
    plain = mk("p", prio=prio + 1)  # higher priority, no quiet requirement
    # The non-quiet higher-priority job wins and starts at once; quiet bookkeeping stays untouched.
    assert q._admit([quiet, plain], gate)["id"] == "p"
    assert quiet.get("quiet_hold_started") is None
    # Only the quiet job: it holds 5 s, then runs with a 5 s hold, not 20.
    quiet = mk("q", prio=prio, quiet=True)
    assert q._admit([quiet], gate) is None
    now[0] = 1020.0
    assert q._admit([quiet], gate)["id"] == "q"
    out = capsys.readouterr().out
    assert "quiet hold start job=q" in out and "quiet hold end job=q hold_s=5.0" in out


def test_admit_paused_running_job_and_m3_lane_do_not_change_the_carry(q, monkeypatch):
    now = [1015.0]
    monkeypatch.setattr(q, "_now", lambda: now[0])
    monkeypatch.setattr(q, "_patch_job", lambda *a, **k: None)
    gate = q.ServingGate()
    gate.cpu = q.QuietGate()
    gate.cpu.sampler = FakeSampler(lambda t: 3.0)
    for t in range(1000, 1015):
        gate.cpu.observe(float(t), running_pid=5)
    paused = mk("r", state="running", paused=True, pid=5)
    quiet = mk(
        "q", quiet=True
    )  # the M3 lane's jobs never reach the M5 admission (daemon filters by device)
    assert q._admit([paused, quiet], gate) is None  # holds only the remaining 5 s
    now[0] = 1020.0
    assert q._admit([paused, quiet], gate)["id"] == "q"


def test_short_lane_job_is_unaffected_by_carry(q, monkeypatch):
    monkeypatch.setattr(q, "_now", lambda: 1015.0)
    monkeypatch.setattr(q, "_patch_job", lambda *a, **k: None)
    gate = q.ServingGate()
    gate.cpu = q.QuietGate()
    gate.cpu.sampler = FakeSampler(lambda t: 3.0)
    short = mk("s", timeout_s=300)  # not quiet: no CPU admission at all
    assert q._admit([short], gate)["id"] == "s"


def test_dlog_lines_carry_epoch_timestamps(q, capsys):
    q._dlog("lane released", job="j1", release_lag_s="0.012")
    line = capsys.readouterr().out.strip()
    assert " t=" in line and "lane released job=j1 release_lag_s=0.012" in line
