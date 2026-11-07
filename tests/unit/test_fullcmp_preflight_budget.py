"""The swap-stability preflight job must outlive its own window (lead 2026-11-03, budget check).

gpuq sends SIGINT to a job when its --timeout (or the --short cap) expires or when its log stops
growing for STALL_S. The swap gate prints nothing until its window ends, so it must have: a timeout
that covers the window plus startup, a short-lane cap that does not cut it below the window, and a
stall threshold above the window. CPU only: reads constants, starts nothing.
"""

from __future__ import annotations

import inspect
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "research"))

import fullcmp_run as fr  # noqa: E402

GPUQ_SRC = (
    ROOT / "scripts" / "dev" / "gpuq.py"
).read_text()  # the repo under test (not the main checkout)
STARTUP_MARGIN_S = 60


def _gpuq_const(name: str) -> float:
    m = re.search(
        rf"^{name} = (?:float\(os\.environ\.get\([^,]+,\s*)?([0-9.]+)", GPUQ_SRC, re.M
    )
    assert m, f"gpuq.py constant {name} not found"
    return float(m.group(1))


def test_swap_gate_timeout_covers_window_and_startup():
    default_window = 300.0  # fullcmp_preflight.py --window-s default (argparse)
    assert float(fr.PREFLIGHT_TIMEOUT_MIN) * 60 >= default_window + STARTUP_MARGIN_S


def test_swap_gate_timeout_is_not_cut_by_the_short_lane_cap():
    # the job is submitted --short: gpuq caps it at SHORT_MIN, so the requested timeout must fit
    assert float(fr.PREFLIGHT_TIMEOUT_MIN) * 60 <= _gpuq_const("SHORT_MIN") * 60


def test_stall_threshold_is_above_the_silent_window():
    # the preflight prints only at the end: a shorter stall threshold would SIGINT it mid-window
    assert _gpuq_const("STALL_S") > 300.0 + 10.0


def test_swap_gate_argv_requests_the_short_lane_with_that_timeout():
    src = inspect.getsource(fr.swap_gate)
    assert '"--short"' in src
    assert "PREFLIGHT_TIMEOUT_MIN" in src


def test_job_wrapper_removes_the_job_homes_on_every_exit(tmp_path, monkeypatch):
    """Engine disk caches in a job's HOME filled the result disk (2026-11-08): the wrapper traps EXIT and
    the termination signals, and a retry (own label) gets its own directories."""
    import os
    import shutil
    import signal
    import subprocess

    monkeypatch.setattr(fr, "TFOUT_ROOT", tmp_path / "tfout")
    monkeypatch.setattr(fr, "HOME_ROOT", tmp_path / "home")
    for label in ("jobA", "jobA-qretry1"):
        trap = fr.job_cleanup_trap(label, ["yunshu"])
        assert str(tmp_path / "tfout" / "home" / label) in trap
        assert str(tmp_path / "home" / "yunshu" / label) in trap
    assert fr.job_cleanup_trap("jobA", ["yunshu"]) != fr.job_cleanup_trap(
        "jobA-qretry1", ["yunshu"]
    )
    for body, sig in (("true", None), ("false", None), ("sleep 30", "TERM")):
        homes = [tmp_path / "tfout" / "home" / "jobB", tmp_path / "home" / "e" / "jobB"]
        for h in homes:
            (h / ".cache" / "apc").mkdir(parents=True)
            (h / ".cache" / "apc" / "x").write_text("cache")
        script = "set -e\n" + fr.job_cleanup_trap("jobB", ["e"]) + body + "\n"
        proc = subprocess.Popen(["/bin/sh", "-c", script], start_new_session=True)
        if sig:
            import time

            time.sleep(0.5)
            os.killpg(proc.pid, signal.SIGTERM)  # gpuq signals the job's process group
        proc.wait(timeout=20)
        assert not any(h.exists() for h in homes), (body, sig)
    shutil.rmtree(tmp_path, ignore_errors=True)


def test_tfbench_removes_only_a_labelled_fullcmp_job_home(tmp_path):
    import tfbench as tb

    keep = {"TFB_FULLCMP": "1"}  # no label: the per-tag home is not the job's to remove
    (tmp_path / "home" / "t").mkdir(parents=True)
    assert tb.cleanup_job_home(tmp_path, keep) is False
    assert (tmp_path / "home" / "t").exists()
    env = {"TFB_FULLCMP": "1", "TFB_JOB_LABEL": "L1"}
    (tmp_path / "home" / "L1" / ".cache").mkdir(parents=True)
    (tmp_path / "home" / "L2").mkdir()
    assert tb.cleanup_job_home(tmp_path, env) is True
    assert not (tmp_path / "home" / "L1").exists()
    assert (tmp_path / "home" / "L2").exists()  # a sibling job's home is untouched
    assert tb.cleanup_job_home(tmp_path, {"TFB_JOB_LABEL": "L2"}) is False
