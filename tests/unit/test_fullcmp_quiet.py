"""Unit tests for scripts/research/fullcmp_quiet.py (quiet-machine gate).

Fakes only: ps, sysctl, clock and sleep are injected; nothing is killed and no process is read.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location(
    "fullcmp_quiet", ROOT / "scripts" / "research" / "fullcmp_quiet.py"
)
q = importlib.util.module_from_spec(_SPEC)
sys.modules["fullcmp_quiet"] = q
_SPEC.loader.exec_module(q)

ME = 5000


class FakeHost:
    """Scripted ps snapshots. Each entry: {pid: (ppid, cputime_seconds, command)}; the clock and
    sleep advance together, so the window length is exact."""

    def __init__(self, frames):
        self.frames = frames
        self.i = 0
        self.now = 0.0

    def ps(self):
        frame = self.frames[min(self.i, len(self.frames) - 1)]
        self.i += 1
        return {
            pid: {
                "pid": pid,
                "ppid": ppid,
                "pcpu": 0.0,
                "etime": "01:00",
                "cpu_s": cpu,
                "command": cmd,
            }
            for pid, (ppid, cpu, cmd) in frame.items()
        }

    def clock(self):
        return self.now

    def sleep(self, s):
        self.now += s

    def load(self):
        return 2.5


def steady(procs_cpu_per_s, samples=11):
    """One frame per ps call; a check uses 11 of them (10 s window, 1 s interval)."""
    """Frames where each (pid -> (ppid, rate_pct, cmd)) burns rate_pct of a core per second."""
    frames = []
    for k in range(samples):
        frames.append(
            {
                pid: (ppid, rate / 100.0 * k, cmd)
                for pid, (ppid, rate, cmd) in procs_cpu_per_s.items()
            }
        )
    return frames


def run_check(
    frames, single=20.0, total=150.0, window=10.0, interval=1.0, self_pid=ME, monkeypatch=None
):
    host = FakeHost(frames)
    if monkeypatch is not None:
        monkeypatch.setattr(q.os, "getpid", lambda: self_pid)
    return q.check(
        single,
        total,
        window,
        interval,
        run_ps_fn=host.ps,
        loadavg_fn=host.load,
        clock=host.clock,
        sleep=host.sleep,
    )


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------


def test_parse_cputime_formats():
    assert q.parse_cputime("0:01.50") == pytest.approx(1.5)
    assert q.parse_cputime("1:02:03") == pytest.approx(3723.0)
    assert q.parse_cputime("2-00:00:10") == pytest.approx(2 * 86400 + 10)


def test_parse_ps_reads_columns_and_keeps_command_spaces():
    text = "  123   1  25.0  01:00  0:15.00 /usr/bin/some tool --flag x\n"
    procs = q.parse_ps(text)
    assert procs[123]["ppid"] == 1
    assert procs[123]["pcpu"] == 25.0
    assert procs[123]["cpu_s"] == pytest.approx(15.0)
    assert procs[123]["command"] == "/usr/bin/some tool --flag x"


def test_parse_ps_fails_closed_on_a_malformed_line():
    with pytest.raises(q.QuietError):
        q.parse_ps("12 1 3.0\n")


def test_parse_loadavg():
    assert q.parse_loadavg("{ 1.23 1.10 0.98 }\n") == pytest.approx(1.23)
    with pytest.raises(q.QuietError):
        q.parse_loadavg("nothing")


# ---------------------------------------------------------------------------
# exclusions: self, ancestors, descendants, harness markers
# ---------------------------------------------------------------------------


def test_harness_pids_excludes_self_ancestors_descendants_and_markers():
    procs = {
        1: {"pid": 1, "ppid": 0, "command": "launchd", "cpu_s": 0},
        300: {"pid": 300, "ppid": 1, "command": "gpuq daemon", "cpu_s": 0},
        400: {"pid": 400, "ppid": 300, "command": "/bin/sh -c job", "cpu_s": 0},
        ME: {
            "pid": ME,
            "ppid": 400,
            "command": "python fullcmp_quiet.py gate",
            "cpu_s": 0,
        },
        ME + 1: {"pid": ME + 1, "ppid": ME, "command": "footprint -p 77", "cpu_s": 0},
        600: {"pid": 600, "ppid": 1, "command": "mds_stores", "cpu_s": 0},
        601: {
            "pid": 601,
            "ppid": 300,
            "command": "python tfbench.py --part decode",
            "cpu_s": 0,
        },
        602: {"pid": 602, "ppid": ME, "command": "/usr/bin/yunshu_helper", "cpu_s": 0},
        603: {"pid": 603, "ppid": 400, "command": "other job child", "cpu_s": 0},
    }
    keep = q.harness_pids(procs, me=ME)
    assert {ME, ME + 1, 400, 300, 601, 602} <= keep
    assert 1 not in keep  # pid 1 is never an ancestor-exclusion (it is host load)
    assert 600 not in keep  # a foreign daemon stays visible
    assert 603 not in keep  # sibling of this job's shell is not this process's tree


def test_check_excludes_harness_and_counts_only_foreign(monkeypatch):
    frames = steady(
        {
            ME: (1, 90.0, "python fullcmp_quiet.py gate"),  # self: never counted
            ME + 1: (ME, 60.0, "footprint -p 77"),  # descendant of self: never counted
            600: (1, 3.0, "mds_stores"),
        }
    )
    rec = run_check(frames, monkeypatch=monkeypatch)
    assert rec["total_foreign_pct"] == pytest.approx(3.0, abs=0.05)
    assert rec["ok"] is True
    assert rec["violators"] == []
    assert [h["name"] for h in rec["holders"]] == ["mds_stores"]


# ---------------------------------------------------------------------------
# measurement, limit and holders
# ---------------------------------------------------------------------------


def test_slow_run_stray_process_fails_the_default_limit(monkeypatch):
    # The A/A slow run: a stray process at 25% CPU appeared during the server start.
    frames = steady({600: (1, 25.0, "/usr/local/bin/stray --loop")})
    rec = run_check(frames, monkeypatch=monkeypatch)
    assert rec["total_foreign_pct"] == pytest.approx(25.0, abs=0.05)
    assert rec["ok"] is False  # one holder >= 20% of a core
    assert rec["violators"][0]["name"] == "stray"
    assert rec["holders"][0]["pid"] == 600


def test_single_holder_at_the_limit_fails(monkeypatch):
    frames = steady({600: (1, 20.0, "burner")})
    rec = run_check(frames, monkeypatch=monkeypatch)
    assert rec["ok"] is False
    assert rec["violators"][0]["pid"] == 600


def test_single_holder_just_under_the_limit_passes(monkeypatch):
    frames = steady({600: (1, 19.0, "burner")})
    rec = run_check(frames, monkeypatch=monkeypatch)
    assert rec["ok"] is True
    assert rec["violators"] == []


def test_system_noise_is_exempt_from_single_rule_but_reported(monkeypatch):
    # WindowServer / kernel_task / mds_stores are macOS noise: listed and counted, never a violator.
    frames = steady(
        {
            700: (1, 43.0, "/System/Library/mds/mds_stores"),
            701: (1, 30.0, "/System/Library/Frameworks/WindowServer"),
        }
    )
    rec = run_check(frames, monkeypatch=monkeypatch)
    assert rec["violators"] == []
    assert rec["ok"] is True  # 73% total < 150%
    assert [h["name"] for h in rec["holders"]] == ["mds_stores", "WindowServer"]


def test_many_small_holders_fail_the_total_rule(monkeypatch):
    # Eight 19% holders: no single violator, but 152% total >= 150%.
    frames = steady({800 + i: (1, 19.0, f"small{i}") for i in range(8)})
    rec = run_check(frames, monkeypatch=monkeypatch)
    assert rec["violators"] == []
    assert rec["total_foreign_pct"] == pytest.approx(152.0, abs=0.1)
    assert rec["ok"] is False
    assert len(rec["holders"]) == 8


def test_small_background_load_passes_and_is_not_listed_as_holder(monkeypatch):
    frames = steady({800: (1, 2.0, "cfprefsd"), 801: (1, 1.5, "WindowServer")})
    rec = run_check(frames, monkeypatch=monkeypatch)
    assert rec["ok"] is True
    assert rec["violators"] == []
    assert [t["name"] for t in rec["top"]] == ["cfprefsd", "WindowServer"]


def test_process_that_exits_keeps_its_last_seen_cpu(monkeypatch):
    # 30% of a core for 5 s, then gone: 1.5 CPU-s over the 10 s window = 15%.
    frames = steady({900: (1, 30.0, "burner")}, samples=6) + [{}] * 6
    rec = run_check(frames, monkeypatch=monkeypatch)
    assert rec["holders"] == [] or rec["holders"][0]["pid"] == 900
    assert rec["total_foreign_pct"] == pytest.approx(15.0, abs=0.05)


def test_check_records_limits_load_window_and_top_five(monkeypatch):
    frames = steady(
        {pid: (1, float(pid - 1000), f"proc{pid}") for pid in range(1001, 1008)}
    )
    rec = run_check(frames, single=40.0, total=200.0, monkeypatch=monkeypatch)
    assert rec["max_single_pct"] == 40.0
    assert rec["max_total_pct"] == 200.0
    assert rec["window_s"] == pytest.approx(10.0)
    assert rec["samples"] == 11
    assert rec["load_avg"] == {"start": 2.5, "end": 2.5, "mean": 2.5}
    assert len(rec["top"]) == 5
    assert rec["top"][0]["pid"] == 1007  # highest CPU first
    json.dumps(rec)  # plain JSON


def test_check_rejects_nonpositive_window():
    with pytest.raises(ValueError):
        q.check(20.0, 150.0, 0.0)


def test_check_raises_when_ps_cannot_run():
    def broken():
        raise q.QuietError("ps failed (rc 1)")

    with pytest.raises(q.QuietError):
        q.check(
            20.0, 150.0, 10.0, run_ps_fn=broken, loadavg_fn=lambda: 1.0, sleep=lambda s: None
        )


def test_check_never_signals_a_process(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("kill/signal must never be called")

    monkeypatch.setattr(q.os, "kill", boom)
    frames = steady({600: (1, 60.0, "hog")})
    rec = run_check(frames, monkeypatch=monkeypatch)
    assert rec["ok"] is False


# ---------------------------------------------------------------------------
# gate: bounded retries, waits, log, busy message
# ---------------------------------------------------------------------------


def test_gate_passes_first_try_without_waiting(monkeypatch):
    frames = steady({800: (1, 1.0, "cfprefsd")})
    host = FakeHost(frames)
    waits = []
    rec = q.gate(
        20.0,
        150.0,
        10.0,
        retries=3,
        wait_s=30.0,
        run_ps_fn=host.ps,
        loadavg_fn=host.load,
        clock=host.clock,
        sleep=lambda s: (waits.append(s), host.sleep(s)),
    )
    assert rec["passed"] is True
    assert rec["attempts"] == [pytest.approx(1.0, abs=0.05)]
    assert 30.0 not in waits  # only sampling-interval sleeps


def test_gate_retries_three_times_waiting_30s_then_busy(monkeypatch, tmp_path):
    # Busy on every attempt: 1 try + 3 retries, 30 s between tries (3 waits).
    frames = steady({600: (1, 60.0, "hog")}, samples=44)  # 4 attempts x 11 samples
    host = FakeHost(frames)
    waits = []
    log = tmp_path / "quiet.jsonl"
    rec = q.gate(
        20.0,
        150.0,
        10.0,
        retries=3,
        wait_s=30.0,
        run_ps_fn=host.ps,
        loadavg_fn=host.load,
        clock=host.clock,
        sleep=lambda s: (waits.append(s), host.sleep(s)),
        log=log,
    )
    assert rec["passed"] is False
    assert len(rec["attempts"]) == 4
    assert [w for w in waits if w != 1.0] == [
        30.0,
        30.0,
        30.0,
    ]  # 1.0 = sampling interval
    assert len(log.read_text().splitlines()) == 4
    assert rec["holders"][0]["name"] == "hog"


def test_gate_stops_as_soon_as_a_try_is_quiet(monkeypatch):
    busy = steady({600: (1, 60.0, "hog")}, samples=11)
    quiet = steady({}, samples=11)
    frames = busy + quiet  # first attempt busy, second attempt quiet
    host = FakeHost(frames)
    waits = []
    rec = q.gate(
        20.0,
        150.0,
        10.0,
        retries=3,
        wait_s=30.0,
        run_ps_fn=host.ps,
        loadavg_fn=host.load,
        clock=host.clock,
        sleep=lambda s: (waits.append(s), host.sleep(s)),
    )
    assert rec["passed"] is True
    assert rec["attempt"] == 2
    assert [w for w in waits if w != 1.0] == [30.0]


def test_main_busy_exit_code_and_message_name_holders(monkeypatch, capsys):
    frames = steady({600: (1, 60.0, "hog")})
    host = FakeHost(frames)
    monkeypatch.setattr(q, "run_ps", host.ps)
    monkeypatch.setattr(q, "read_loadavg", host.load)
    monkeypatch.setattr(q.time, "monotonic", host.clock)
    monkeypatch.setattr(q.time, "sleep", host.sleep)
    rc = q.main(["gate", "--retries", "0", "--window-s", "10"])
    err = capsys.readouterr().err
    assert rc == q.GATE_EXIT_BUSY
    assert "BUSY" in err and "hog" in err and "pid 600" in err


def test_main_measurement_failure_exits_one(monkeypatch, capsys):
    def broken():
        raise q.QuietError("ps failed")

    monkeypatch.setattr(q, "run_ps", broken)
    monkeypatch.setattr(q, "read_loadavg", lambda: 1.0)
    monkeypatch.setattr(q.time, "sleep", lambda s: None)
    assert q.main(["check", "--window-s", "1"]) == 1
    assert "measurement failed" in capsys.readouterr().err


def test_module_has_no_kill_path():
    src = Path(q.__file__).read_text()
    assert "os.kill" not in src and "killpg" not in src and "send_signal" not in src
    assert os.path.basename(q.__file__) == "fullcmp_quiet.py"
