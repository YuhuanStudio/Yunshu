"""CPU-only tests for the telemetry sampler (scripts/research/fullcmp_telemetry.py).

Parsers are tested on command output captured on this machine (tests/unit/fixtures/fullcmp_telemetry,
files named after the command). Only the throttle and powermetrics strings are synthetic and say so.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "research"))

from datetime import UTC

import fullcmp_telemetry as tm  # noqa: E402

FIX = Path(__file__).resolve().parent / "fixtures" / "fullcmp_telemetry"


def _read(name: str) -> str:
    return (FIX / name).read_text()


# ---- parsers on captured output ----------------------------------------------
def test_vm_stat_captured():
    v = tm.parse_vm_stat(_read("vm_stat.txt"))
    assert v["page_bytes"] == 16384
    assert v["free_bytes"] == 3126255 * 16384
    assert v["wired_bytes"] == 2305952 * 16384
    assert v["compressed_bytes"] == 54823 * 16384  # "Pages occupied by compressor"
    assert v["swapouts"] == 11150583


def test_vm_stat_without_header_raises():
    with pytest.raises(tm.ProbeError):
        tm.parse_vm_stat("Pages free: 3.\n")


def test_swapusage_captured():
    s = tm.parse_swapusage(_read("swapusage.txt"))
    assert s["total_bytes"] == 5120 * 2**20
    assert s["used_bytes"] == pytest.approx(3671.06 * 2**20, abs=1)
    assert s["free_bytes"] == pytest.approx(1448.94 * 2**20, abs=1)


def test_swapusage_unparseable_raises():
    with pytest.raises(tm.ProbeError):
        tm.parse_swapusage("vm.swapusage: nonsense")


def test_pressure_captured():
    p = tm.parse_pressure(_read("pressure_sysctl.txt"), _read("memory_pressure_q.txt"))
    assert p == {"level": 1, "level_name": "normal", "free_pct": 71}


def test_pressure_levels_named():
    assert (
        tm.parse_pressure("kern.memorystatus_vm_pressure_level: 4")["level_name"]
        == "critical"
    )
    assert (
        tm.parse_pressure("kern.memorystatus_vm_pressure_level: 2")["level_name"]
        == "warn"
    )


def test_ioreg_captured_reads_exact_keys_not_driver_variant():
    g = tm.parse_ioreg_accel(_read("ioreg_ioaccel.txt"))
    # captured line: "Device Utilization %"=99 and "In use system memory"=32745897984
    # the "(driver)" variant is 0 in the same output and must not be read
    assert g["device_util_pct"] == 99
    assert g["in_use_sys_mem_bytes"] == 32745897984
    assert g["alloc_sys_mem_bytes"] == 34315567104


def test_ioreg_without_stats_raises():
    with pytest.raises(tm.ProbeError):
        tm.parse_ioreg_accel("+-o AGXAccelerator  <class x>\n")


def test_footprint_captured():
    f = tm.parse_footprint(_read("footprint_sleep.txt"))
    assert f["phys_footprint_bytes"] == 1200 * 1024  # "Footprint: 1200 KB"


def test_footprint_units():
    assert tm.parse_footprint("x [1]: 64-bit Footprint: 2.5 GB (16384 bytes per page)")[
        "phys_footprint_bytes"
    ] == int(2.5 * 2**30)
    assert (
        tm.parse_footprint("Footprint: 512 MB")["phys_footprint_bytes"] == 512 * 2**20
    )


def test_thermal_captured_is_not_throttled():
    t = tm.parse_thermal(_read("pmset_therm.txt"))
    assert t == {
        "cpu_speed_limit_pct": None,
        "thermal_warning_recorded": False,
        "performance_warning_recorded": False,
        "throttled": False,
    }


def test_thermal_synthetic_speed_limit_is_throttled():
    # synthetic: format of the CPU_Speed_Limit line, not captured on this machine
    t = tm.parse_thermal(
        "CPU_Speed_Limit = 80\nNote: No thermal warning level has been recorded\n"
    )
    assert t["cpu_speed_limit_pct"] == 80 and t["throttled"] is True


def test_thermal_empty_raises():
    with pytest.raises(tm.ProbeError):
        tm.parse_thermal("")


def test_powermetrics_synthetic_parse():
    # synthetic sample text in the powermetrics line format; not captured (no sudo here)
    text = (
        "CPU Power: 1200 mW\nGPU Power: 8400 mW\nANE Power: 0 mW\n"
        "Combined Power (CPU + GPU + ANE): 9600 mW\nGPU HW active frequency: 1398 MHz\n"
        "GPU HW active residency: 87.5%\n"
    )
    p = tm.parse_powermetrics(text)
    assert (
        p["combined_power_mw"] == 9600
        and p["gpu_freq_mhz"] == 1398
        and p["gpu_active_residency_pct"] == 87.5
    )


def test_powermetrics_without_power_lines_raises():
    with pytest.raises(tm.ProbeError):
        tm.parse_powermetrics("nothing here")


# ---- degrade to null -------------------------------------------------------------
def test_run_failure_is_probe_error(monkeypatch):
    def boom(*a, **k):
        raise OSError("no such tool")

    monkeypatch.setattr(tm.subprocess, "run", boom)
    with pytest.raises(tm.ProbeError):
        tm._run(["vm_stat"])


def test_call_returns_none_and_counts_error():
    s = tm.Sampler(1, None, "unused")
    s._state["gpu"] = tm.ProbeState("gpu", True, None, 1000)

    def fail():
        raise tm.ProbeError("ioreg went away")

    assert s._call("gpu", fail) is None
    st = s._state["gpu"]
    assert st.errors == 1 and "ioreg went away" in st.last_error and st.calls == 1


def test_unavailable_probes_are_listed_with_reason(tmp_path, monkeypatch):
    # every shell probe fails at start; the header must say so, not silently drop them
    def fail(*a, **k):
        raise tm.ProbeError("tool missing")

    monkeypatch.setattr(tm, "_run", fail)
    monkeypatch.setattr(
        tm, "probe_sudo_nopass", lambda: (False, "sudo -n needs a password")
    )
    # macmon binary absent: the power probe must be listed unavailable with the reason
    monkeypatch.setitem(tm.pw.BINARIES, "macmon", str(tmp_path / "no-macmon"))
    import os

    s = tm.Sampler(
        os.getpid(),
        8000,
        str(tmp_path / "t.jsonl"),
        probes=("proc", "vmstat", "gpu", POWER),
    )
    s.start()
    footer = s.stop()
    header, _, _, _ = tm.read_telemetry(tmp_path / "t.jsonl")
    assert header["probes"]["proc"]["available"] is True
    assert header["probes"]["vmstat"]["available"] is False
    assert "tool missing" in header["probes"]["vmstat"]["reason"]
    pwr = header["probes"]["power"]
    assert pwr["available"] is False
    assert pwr["reason"].startswith("unavailable (binary missing")
    assert pwr["source"] == "macmon" and pwr["interval_ms"] == 1000
    assert header["power_sudo_note"] == "sudo -n needs a password"
    assert footer["n_samples"] >= 0


POWER = "power"


def test_power_is_unavailable_without_passwordless_sudo(monkeypatch):
    monkeypatch.setattr(
        tm.subprocess, "run", lambda *a, **k: type("C", (), {"returncode": 1})()
    )
    ok, reason = tm.probe_sudo_nopass()
    assert ok is False and "password" in reason


def test_marks_read_only_complete_lines(tmp_path):
    marks = tmp_path / "m.jsonl"
    marks.write_text('{"ev": "request_start", "req": "0", "t": 5.0}\n{"ev": "end"')
    s = tm.Sampler(1, None, str(tmp_path / "o.jsonl"), marks_file=str(marks))
    got = s._read_marks()
    assert got == [{"part": "mark", "ev": "request_start", "req": "0", "t": 5.0}]
    marks.write_text(marks.read_text() + ', "t": 6.0}\n')
    assert s._read_marks() == [{"part": "mark", "ev": "end", "t": 6.0}]


# ---- summarize -----------------------------------------------------------------
def _sample(t, rss, fp, util, swap=100, power=None):
    row = {
        "part": "sample",
        "t": t,
        "proc": {"rss_bytes": rss, "cpu_pct": 10.0},
        "footprint": {"phys_footprint_bytes": fp},
        "gpu": {"device_util_pct": util},
        "swap": {"used_bytes": swap},
        "thermal": {"throttled": False},
    }
    if power is not None:
        row["power"] = {"combined_power_mw": power}
    return row


def test_summarize_window_peak_mean_after_idle():
    rows = [
        _sample(0.0, 100, 90, 0),
        _sample(1.0, 300, 280, 80),
        _sample(2.0, 500, 450, 100),
        _sample(3.0, 400, 380, 60),
        _sample(
            10.0, 120, 110, 1
        ),  # after-idle sample (end 3.0 + idle 5 = 8 -> first at/after 8)
    ]
    marks = [
        {"part": "mark", "ev": "request_start", "req": "a", "t": 0.5},
        {"part": "mark", "ev": "first_token", "req": "a", "t": 1.2},
        {"part": "mark", "ev": "end", "req": "a", "t": 3.0, "tokens": 128},
    ]
    s = tm.summarize(rows, marks, idle_s=5.0)
    w = s["windows"][0]
    assert w["req"] == "a"
    assert w["peak_rss_bytes"] == 500 and w["peak_footprint_bytes"] == 450
    assert w["gpu_util_peak_pct"] == 100
    assert w["gpu_util_mean_pct"] == pytest.approx((80 + 100 + 60) / 3, abs=0.01)
    assert w["ttft_s"] == pytest.approx(0.7, abs=1e-3)
    assert w["after_idle_rss_bytes"] == 120
    assert w["after_idle_gpu_util_pct"] == 1
    assert w["energy_j"] is None and "power unavailable" in w["energy_note"]


def _power_row(t, sys_w, source="macmon", **extra):
    row = {
        "part": "power",
        "source": source,
        "t": t,
        "gpu_w": 1.0,
        "cpu_w": 2.0,
        "ane_w": 0.5,
        "sys_w": sys_w,
        "all_w": 3.5,
    }
    row.update(extra)
    return row


def test_summarize_energy_per_token_from_power_rows_in_window():
    # 10 W system power held for the 2 s window = 20 J; 100 output tokens -> 0.2 J/token
    rows = [_power_row(float(t), 10.0) for t in range(3)] + [
        _sample(0.0, 1, 1, 1),
        _sample(2.0, 1, 1, 1),
    ]
    marks = [
        {"part": "mark", "ev": "request_start", "req": "0", "t": 0.0},
        {"part": "mark", "ev": "end", "req": "0", "t": 2.0, "tokens": 100},
    ]
    w = tm.summarize(rows, marks)["windows"][0]
    assert w["energy_j"]["sys_w"] == pytest.approx(20.0)
    assert w["energy_j"]["gpu_w"] == pytest.approx(2.0)  # 1 W for 2 s
    assert w["energy_j"]["cpu_w"] == pytest.approx(4.0)
    assert w["energy_j"]["ane_w"] == pytest.approx(1.0)
    assert w["energy_j_per_token"]["sys_w"] == pytest.approx(0.2)
    assert w["energy_j_per_token"]["gpu_w"] == pytest.approx(0.02)
    assert w["power_w"]["sys_w"] == {"mean": 10.0, "peak": 10.0}
    assert w["power_source"] == ["macmon"] and w["power_samples"] == 3


def test_summarize_energy_window_excludes_power_outside_marks():
    rows = [_power_row(float(t), 10.0) for t in range(10)]
    marks = [
        {"part": "mark", "ev": "request_start", "req": "0", "t": 4.0},
        {"part": "mark", "ev": "end", "req": "0", "t": 6.0, "tokens": 4},
    ]
    w = tm.summarize(rows + [_sample(0.0, 1, 1, 1)], marks)["windows"][0]
    # the window is [4, 6] at 10 W throughout: 20 J; power rows outside it do not add energy
    assert w["energy_j"]["sys_w"] == pytest.approx(20.0)
    assert w["energy_j_per_token"]["sys_w"] == pytest.approx(5.0)


def test_summarize_power_secondary_rows_are_not_integrated():
    rows = [
        _power_row(0.0, 10.0),
        _power_row(2.0, 10.0),
        {**_power_row(0.0, 999.0, source="mactop"), "part": "power_secondary"},
    ]
    marks = [
        {"part": "mark", "ev": "request_start", "req": "0", "t": 0.0},
        {"part": "mark", "ev": "end", "req": "0", "t": 2.0, "tokens": 10},
    ]
    w = tm.summarize(rows, marks)["windows"][0]
    assert w["energy_j"]["sys_w"] == pytest.approx(20.0)


def test_summarize_no_power_rows_is_null_not_zero():
    marks = [
        {"part": "mark", "ev": "request_start", "req": "0", "t": 0.0},
        {"part": "mark", "ev": "end", "req": "0", "t": 2.0, "tokens": 10},
    ]
    w = tm.summarize([_sample(0.0, 1, 1, 1), _sample(2.0, 1, 1, 1)], marks)["windows"][
        0
    ]
    assert w["energy_j"] is None and w["energy_j_per_token"] is None
    assert w["power_w"] is None and w["energy_note"] == "power unavailable"


def test_powermetrics_row_converts_mw_to_w():
    row = tm.powermetrics_row(
        {
            "cpu_power_mw": 1200.0,
            "gpu_power_mw": 8400.0,
            "ane_power_mw": 0.0,
            "combined_power_mw": 9600.0,
            "gpu_freq_mhz": 1398.0,
            "gpu_active_residency_pct": 87.5,
        },
        7.0,
    )
    assert row["part"] == "power" and row["source"] == "powermetrics"
    assert row["gpu_w"] == pytest.approx(8.4) and row["all_w"] == pytest.approx(9.6)
    assert row["sys_w"] is None  # not in the powermetrics text: null, not derived
    assert row["gpu_active"] == pytest.approx(0.875)


def test_tool_records_checks_sha256_against_tools_json(tmp_path):
    binary = tmp_path / "macmon"
    binary.write_bytes(b"not really macmon")
    import hashlib

    good = hashlib.sha256(b"not really macmon").hexdigest()
    tools = tmp_path / "TOOLS.json"
    tools.write_text(
        json.dumps(
            {
                "tools": {
                    "macmon": {
                        "version": "0.10.0",
                        "binary": str(binary),
                        "sudoless": True,
                        "binary_sha256": good,
                    },
                    "mactop": {
                        "version": "0.2.7",
                        "binary": str(tmp_path / "x"),
                        "binary_sha256": "0" * 64,
                    },
                }
            }
        )
    )
    rec = tm.tool_records(str(tools))
    assert rec["macmon"]["version"] == "0.10.0"
    assert (
        rec["macmon"]["sha256_actual"] == good and rec["macmon"]["sha256_match"] is True
    )
    assert (
        rec["mactop"]["sha256_actual"] is None
        and rec["mactop"]["sha256_match"] is False
    )
    assert tm.tool_records(str(tmp_path / "missing.json")) == {
        "error": "TOOLS.json unreadable: FileNotFoundError"
    }


def test_sampler_writes_power_rows_from_fake_macmon_child(tmp_path, monkeypatch):
    # sudo -n is forced to fail, so the sudoless macmon reader is the power source
    import os

    line = (
        '{"timestamp":"2026-11-03T01:57:06.235457+00:00","sys_power":36.0,"gpu_power":0.03,'
        '"cpu_power":13.0,"ane_power":0.0,"all_power":13.0,"gpu_usage":[338,0.006],'
        '"memory":{"ram_total":1,"ram_usage":2,"swap_total":3,"swap_usage":4}}'
    )
    fake = tmp_path / "fake_macmon"
    fake.write_text(f"#!/bin/sh\nwhile :; do echo '{line}'; sleep 0.05; done\n")
    fake.chmod(0o755)
    monkeypatch.setitem(tm.pw.BINARIES, "macmon", str(fake))
    monkeypatch.setattr(tm, "probe_sudo_nopass", lambda: (False, "no sudo"))
    out = tmp_path / "t.jsonl"
    s = tm.Sampler(os.getpid(), None, str(out), probes=("proc", POWER))
    s.start()
    deadline = time.time() + 5
    while (
        s._power is not None and len(s._power.snapshot()) < 3 and time.time() < deadline
    ):
        time.sleep(0.05)
    footer = s.stop()
    header, samples, _, _ = tm.read_telemetry(out)
    assert header["probes"]["power"]["available"] is True
    assert header["probes"]["power"]["source"] == "macmon"
    assert header["probes"]["power"]["reason"] is None
    rows = tm.read_telemetry_rows(out)
    power = [r for r in rows if r.get("part") == "power"]
    assert len(power) >= 3
    assert power[0]["sys_w"] == 36.0 and power[0]["source"] == "macmon"
    # macmon's own ISO timestamp is the row time (same epoch clock as the marks)
    from datetime import datetime

    expected = datetime(2026, 11, 3, 1, 57, 6, 235457, tzinfo=UTC).timestamp()
    assert power[0]["t"] == pytest.approx(expected, abs=1e-6)
    assert all("power" not in s_ for s_ in samples)  # power is never inside a tick row
    assert footer["power_stops"]["power"]["status"] == "stopped"
    assert "total_bytes" in header["swap_start"]
    assert set(header["tools"]) == {"macmon", "mactop"}


def test_sampler_power_unavailable_when_macmon_missing(tmp_path, monkeypatch):
    import os

    monkeypatch.setitem(tm.pw.BINARIES, "macmon", str(tmp_path / "absent"))
    monkeypatch.setattr(tm, "probe_sudo_nopass", lambda: (False, "no sudo"))
    out = tmp_path / "t.jsonl"
    s = tm.Sampler(os.getpid(), None, str(out), probes=("proc", POWER))
    s.start()
    s.stop()
    header, _, _, _ = tm.read_telemetry(out)
    assert header["probes"]["power"]["available"] is False
    assert header["probes"]["power"]["reason"].startswith("unavailable (binary missing")
    assert not [r for r in tm.read_telemetry_rows(out) if r.get("part") == "power"]


def test_summarize_swap_growth_and_throttle_flags():
    rows = [_sample(0.0, 1, 1, 1, swap=100), _sample(1.0, 1, 1, 1, swap=4096)]
    rows[1]["thermal"] = {"throttled": True}
    overall = tm.summarize(rows)["overall"]
    assert overall["swap_used_delta_bytes"] == 3996
    assert overall["throttled"] is True


def test_summarize_incomplete_marks_are_reported_not_dropped():
    marks = [{"part": "mark", "ev": "request_start", "req": "x", "t": 0.0}]
    windows = tm.summarize([_sample(0.0, 1, 1, 1)], marks)["windows"]
    assert windows == [{"req": "x", "error": "incomplete marks"}]


def test_read_telemetry_roundtrip(tmp_path):
    p = tmp_path / "x.jsonl"
    rows = [
        {"part": "telemetry_header", "probes": {}},
        _sample(0.0, 1, 1, 1),
        {"part": "telemetry_footer"},
    ]
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    h, samples, marks, f = tm.read_telemetry(p)
    assert (
        h["part"] == "telemetry_header"
        and len(samples) == 1
        and marks == []
        and f["part"] == "telemetry_footer"
    )
