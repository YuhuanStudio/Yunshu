"""CPU-only tests for the sudoless power adapter (scripts/research/fullcmp_power.py).

Parsers run on the macmon and mactop pipe-mode output captured on this machine (tests/unit/fixtures/
fullcmp_power). The child-process tests use a tiny fake shell script in tmp_path, never a real reader.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "research"))

import fullcmp_power as pw  # noqa: E402

FIX = Path(__file__).resolve().parent / "fixtures" / "fullcmp_power"


def _read(name: str) -> str:
    return (FIX / name).read_text()


def _first_int(text: str) -> int:
    import re

    return int(re.search(r"\d+", text).group(0))


# ---- parsers on captured output ----------------------------------------------
def test_macmon_fixture_parses_every_sample():
    rows, bad = pw.parse_stream(_read("macmon_pipe_idle_10s.jsonl"), "macmon")
    assert bad == 0
    assert len(rows) == 10
    ts = [r["t"] for r in rows]
    assert ts == sorted(ts)
    assert all(
        0.9 < b - a < 1.1 for a, b in zip(ts, ts[1:], strict=False)
    )  # 1000 ms interval
    r = rows[0]
    assert r["source"] == "macmon" and r["part"] == "power"
    assert r["sys_w"] == pytest.approx(36.006256103515625)
    assert r["gpu_w"] == pytest.approx(0.02834651991724968)
    assert r["gpu_mhz"] == 338
    assert r["gpu_active"] == pytest.approx(0.03166108950972557)
    assert r["cpu_temp_c"] == pytest.approx(50.37543487548828)


def test_macmon_swap_matches_sysctl_at_capture():
    # sysctl vm.swapusage at the macmon run read "used = 3551.06M"; macmon's swap_usage bytes agree.
    rows, _ = pw.parse_stream(_read("macmon_pipe_idle_10s.jsonl"), "macmon")
    assert rows[0]["swap_used"] == 3723558912
    assert round(rows[0]["swap_used"] / 2**20, 2) == 3551.06
    assert "used = 3551.06M" in _read("sysctl_swap_macmon_time.txt")


def test_mactop_fixture_parses_and_normalises_percent():
    rows, bad = pw.parse_stream(
        _read("mactop_headless_idle_10s.jsonl"), "mactop", receipt_t=5.0
    )
    assert bad == 0
    assert len(rows) == 10
    r = rows[0]
    assert r["source"] == "mactop"
    assert (
        r["t"] == 5.0
    )  # mactop timestamps have 1 s resolution, so the receipt time is used
    assert r["sys_w"] == pytest.approx(27.412328394216797)
    assert r["all_w"] == pytest.approx(27.446544647216797)
    assert r["gpu_active"] == pytest.approx(3.6252756609957797 / 100)
    assert r["ram_used"] == 22414409728
    assert r["swap_used"] == 3698393088
    assert "used = 3527.06M" in _read("sysctl_swap_mactop_time.txt")
    assert round(r["swap_used"] / 2**20, 2) == 3527.06


def test_mactop_cpu_power_reported_zero_is_kept_as_reported():
    rows, _ = pw.parse_stream(
        _read("mactop_headless_idle_10s.jsonl"), "mactop", receipt_t=5.0
    )
    assert all(r["cpu_w"] == 0 for r in rows)


def test_ioreg_device_utilization_idle_agrees_with_macmon_gpu_active():
    util = _first_int(_read("ioreg_device_util_before.txt")) / 100
    rows, _ = pw.parse_stream(_read("macmon_pipe_idle_10s.jsonl"), "macmon")
    # ioreg is a single instantaneous read (0 %), macmon is a 1 s average (~3 %): both say idle.
    assert util == 0
    assert all(r["gpu_active"] < 0.05 for r in rows)


def test_garbage_and_powerless_lines_are_counted_not_invented():
    text = (
        "not json\n"
        + '{"unrelated": 1}\n'
        + _read("macmon_pipe_idle_10s.jsonl").splitlines()[0]
    )
    rows, bad = pw.parse_stream(text, "macmon")
    assert len(rows) == 1
    assert bad == 2


def test_macmon_missing_timestamp_falls_back_to_receipt_time():
    row = pw.parse_macmon({"sys_power": 5.0, "gpu_power": 1.0}, receipt_t=42.0)
    assert row["t"] == 42.0
    assert row["sys_w"] == 5.0
    assert row["cpu_w"] is None  # absent, not zero


# ---- summaries -----------------------------------------------------------------
def _rows(powers, t0=100.0, dt=1.0, key="all_w"):
    return [{"part": "power", "t": t0 + i * dt, key: p} for i, p in enumerate(powers)]


def test_energy_is_left_rectangle_clipped_to_window():
    rows = _rows([10.0, 20.0, 30.0])  # samples at 100, 101, 102
    # window [100.5, 102.0]: 20 W from 101 to 102 = 20 J; 10 W part is before 101 so 10 W * 0.5 s = 5 J
    assert pw.energy_j(rows, 100.5, 102.0) == pytest.approx(5.0 + 20.0)


def test_energy_none_without_samples():
    assert pw.energy_j(_rows([10.0]), 200.0, 201.0) is None


def test_summarize_mean_peak_energy_and_j_per_token():
    rows = _rows([10.0, 20.0, 30.0, 40.0])
    s = pw.summarize(rows, 100.0, 103.0, tokens=50)
    assert s["stats"]["all_w"]["mean"] == 25.0
    assert s["stats"]["all_w"]["peak"] == 40.0
    # left rule over [100, 103]: 10 + 20 + 30 J, the last sample holds for 0 s
    assert s["energy_j"]["all_w"] == pytest.approx(60.0)
    assert s["j_per_token"] == pytest.approx(60.0 / 50)


def test_summarize_empty_window_reports_error_not_zero():
    s = pw.summarize(_rows([10.0]), 500.0, 501.0, tokens=10)
    assert "error" in s and "energy_j" not in s


# ---- child process ------------------------------------------------------------
def _fake(tmp_path: Path, body: str, name: str = "fake_reader") -> str:
    p = tmp_path / name
    p.write_text("#!/bin/sh\n" + body)
    p.chmod(0o755)
    return str(p)


_MACMON_LINE = (
    '{"timestamp":"2026-11-03T01:57:06.235457+00:00","sys_power":36.0,"gpu_power":0.03,'
    '"cpu_power":13.0,"ane_power":0.0,"all_power":13.0,"gpu_usage":[338,0.006],'
    '"memory":{"ram_total":1,"ram_usage":2,"swap_total":3,"swap_usage":4}}'
)


def test_missing_binary_is_unavailable_with_reason(tmp_path):
    s = pw.PowerSampler(tool="macmon", binary=str(tmp_path / "nope")).start()
    assert s.status.startswith("unavailable (binary missing")
    res = s.stop()
    assert res["rows"] == [] and res["killed"] is False


def test_unknown_tool_is_unavailable(tmp_path):
    s = pw.PowerSampler(tool="powermetrics").start()
    assert s.status.startswith("unavailable (unknown tool")


def test_fake_child_rows_then_clean_terminate(tmp_path):
    binary = _fake(tmp_path, f"while :; do echo '{_MACMON_LINE}'; sleep 0.05; done\n")
    s = pw.PowerSampler(tool="macmon", interval_ms=50, binary=binary).start()
    deadline = time.time() + 5
    while len(s.snapshot()) < 3 and time.time() < deadline:
        time.sleep(0.05)
    res = s.stop()
    assert len(res["rows"]) >= 3
    assert res["rows"][0]["sys_w"] == 36.0
    assert res["killed"] is False
    assert res["status"] == "stopped"
    assert s._proc.poll() is not None  # child is gone


def test_child_ignoring_sigterm_is_killed_after_grace(tmp_path):
    binary = _fake(
        tmp_path, f"trap '' TERM\nwhile :; do echo '{_MACMON_LINE}'; sleep 0.05; done\n"
    )
    s = pw.PowerSampler(
        tool="macmon", interval_ms=50, binary=binary, grace_s=0.5
    ).start()
    deadline = time.time() + 5
    while not s.snapshot() and time.time() < deadline:
        time.sleep(0.05)
    res = s.stop()
    assert res["killed"] is True
    assert s._proc.poll() is not None


def test_child_exiting_early_reports_unavailable(tmp_path):
    binary = _fake(tmp_path, "exit 3\n")
    s = pw.PowerSampler(tool="macmon", binary=binary, grace_s=0.5).start()
    deadline = time.time() + 5
    while s.status == "running" and time.time() < deadline:
        time.sleep(0.05)
    assert s.status.startswith("unavailable (child exited rc=3, no rows")
    s.stop()


def test_mactop_child_streams_array_format(tmp_path):
    mt = _read("mactop_headless_idle_10s.jsonl").splitlines()
    body = "".join(f"cat <<'EOF'\n{line}\nEOF\n" for line in mt[:3]) + "sleep 5\n"
    binary = _fake(tmp_path, body, name="fake_mactop")
    s = pw.PowerSampler(tool="mactop", binary=binary, grace_s=0.5).start()
    deadline = time.time() + 5
    while len(s.snapshot()) < 3 and time.time() < deadline:
        time.sleep(0.05)
    res = s.stop()
    assert len(res["rows"]) == 3
    assert res["rows"][0]["source"] == "mactop"
    assert os.path.basename(s.binary) == "fake_mactop"
