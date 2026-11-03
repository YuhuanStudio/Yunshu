#!/usr/bin/env python3
"""Per-cell telemetry sampler for the full comparison (FULLCMP_SPEC.md, "Telemetry").

Runs beside one server process and writes a JSONL time series:

    line 1      {"part": "telemetry_header", ...}   probe availability + reasons, thermal at start
    lines 2..n  {"part": "sample", "t": epoch, ...}  one row per tick (only the fields due that tick)
    marks       {"part": "mark", "ev": ..., "t": ...} request_start / first_token / end, from the harness
    last line   {"part": "telemetry_footer", ...}  thermal at end, measured cost per probe, error counts

Every probe degrades independently. A probe that is unavailable at start is listed in the header with
"available": false and a reason; a probe that fails at runtime writes null for that tick and bumps an
error counter in the footer. Nothing is estimated or filled in: a missing value is null.

Probes and their source (macOS):
    proc      psutil.Process: RSS, CPU user/sys seconds, thread count, context switches     every tick (100 ms)
    disk      psutil.disk_io_counters(): SYSTEM-WIDE read/write bytes (per-process io counters are
              not available on macOS, so disk I/O cannot be attributed to the server)           every tick
    net_lo0   psutil.net_io_counters(pernic=True)["lo0"]: loopback bytes. AGGREGATE over all loopback
              ports (no per-port split is available without nettop); the header records the port     every tick
    vmstat    `vm_stat`: free / wired / compressed (pages occupied by compressor) x page size       1000 ms
    swap      `sysctl vm.swapusage`: total/used/free MB                                             1000 ms
    pressure  `sysctl kern.memorystatus_vm_pressure_level` (1 normal, 2 warn, 4 critical) and
              `memory_pressure -Q` free percentage                                                  1000 ms
    gpu       `ioreg -r -d 1 -w0 -c IOAccelerator` PerformanceStatistics: "Device Utilization %",
              "In use system memory", "Alloc system memory" (~46 KB output: the expensive probe)  1000 ms
    footprint `footprint -p PID`: header "Footprint: N KB|MB|GB" = phys_footprint of the process   1000 ms
    thermal   `pmset -g therm`: CPU_Speed_Limit and whether a thermal/performance warning is recorded
              (also read at start and at stop)                                                      1000 ms
    power     CPU/GPU/ANE/system power (W) as separate `part: "power"` lines (not per tick).
              Source: `sudo -n powermetrics` when `sudo -n true` succeeds, else sudoless
              `macmon pipe -i 1000` (fullcmp_power.PowerSampler). Optional secondary: `mactop -headless`
              (`--power-secondary mactop`, rows as `part: "power_secondary"`). Never faked: a reader that
              fails leaves the probe "unavailable" with a reason and no power lines.

Header also records swap used at start (`swap_start`) and the tool versions and sha256 from TOOLS.json,
next to the sha256 of each binary on disk (`tools`). The footer records `swap_end`.

Cost: the cheap probes (proc, disk, net) run every tick; the subprocess probes run at 1000 ms because
ioreg and footprint are the largest per-call costs. Measured per-probe cost is written to the footer.

Usage:
    python scripts/research/fullcmp_telemetry.py --pid PID --port P --interval-ms 100 \
        [--out FILE | --out-dir DIR] [--marks marks.jsonl] [--duration S] [--idle-s 5]
    (default --out-dir /Volumes/Micron/yunshu-fullcmp/raw; the run log is written next to the output)

The harness can also append mark lines to the --marks file: {"ev": "request_start", "req": "0"}.
Function API: Sampler(pid, port, out).start(); .mark("first_token", req="0"); .stop().
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field

import fullcmp_power as pw
import psutil

FAST_PROBES = ("proc", "disk", "net_lo0")
SLOW_PROBES = ("vmstat", "swap", "pressure", "gpu", "footprint", "thermal")
POWER_PROBE = "power"
ALL_PROBES = FAST_PROBES + SLOW_PROBES + (POWER_PROBE,)
SLOW_INTERVAL_MS = 1000
PAGE_BYTES_DEFAULT = 16384
MARK_EVENTS = ("request_start", "first_token", "end")
DEFAULT_OUT_DIR = "/Volumes/Micron/yunshu-fullcmp/raw"
TOOLS_JSON = "/Volumes/P5Plus/yunshu-build/tools/TOOLS.json"
TOOL_NAMES = ("macmon", "mactop")
ENERGY_KEYS = ("gpu_w", "cpu_w", "ane_w", "sys_w", "all_w")
POWER_PART_NAMES = ("power", "power_secondary")


# ---------------------------------------------------------------------------
# Pure parsers (unit-tested on captured command output)
# ---------------------------------------------------------------------------
class ProbeError(RuntimeError):
    """A probe could not produce a value (missing output, unparseable text, exited process)."""


_UNIT = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3, "TB": 1024**4}


def _to_bytes(num: str, unit: str) -> int:
    return int(float(num) * _UNIT[unit.upper()])


def parse_vm_stat(text: str) -> dict:
    """`vm_stat` -> free/wired/compressed/active/inactive bytes (page size from the header)."""
    m = re.search(r"page size of (\d+) bytes", text)
    if not m:
        raise ProbeError("vm_stat: no page size header")
    page = int(m.group(1))
    counts = {}
    for line in text.splitlines():
        mm = re.match(r'\s*"?([^:"]+?)"?:\s+(\d+)\.\s*$', line)
        if mm:
            counts[mm.group(1).strip().lower()] = int(mm.group(2))
    if "pages free" not in counts or "pages wired down" not in counts:
        raise ProbeError("vm_stat: missing free/wired counters")
    return {
        "page_bytes": page,
        "free_bytes": counts["pages free"] * page,
        "wired_bytes": counts["pages wired down"] * page,
        "compressed_bytes": counts.get("pages occupied by compressor", 0) * page,
        "active_bytes": counts.get("pages active", 0) * page,
        "inactive_bytes": counts.get("pages inactive", 0) * page,
        "swapins": counts.get("swapins", 0),
        "swapouts": counts.get("swapouts", 0),
    }


def parse_swapusage(text: str) -> dict:
    """`sysctl vm.swapusage` -> total/used/free bytes."""
    m = re.search(
        r"total = ([\d.]+)([BKMG])\s+used = ([\d.]+)([BKMG])\s+free = ([\d.]+)([BKMG])",
        text,
    )
    if not m:
        raise ProbeError("swapusage: unparseable")
    t, tu, u, uu, f, fu = m.groups()
    unit = {"B": "B", "K": "KB", "M": "MB", "G": "GB"}
    return {
        "total_bytes": _to_bytes(t, unit[tu]),
        "used_bytes": _to_bytes(u, unit[uu]),
        "free_bytes": _to_bytes(f, unit[fu]),
    }


def parse_pressure(sysctl_text: str, mpressure_text: str | None = None) -> dict:
    """`sysctl kern.memorystatus_vm_pressure_level` (+ optional `memory_pressure -Q`)."""
    m = re.search(r"kern\.memorystatus_vm_pressure_level:\s*(\d+)", sysctl_text)
    if not m:
        raise ProbeError("pressure: no level")
    level = int(m.group(1))
    out = {
        "level": level,
        "level_name": {1: "normal", 2: "warn", 4: "critical"}.get(level, "unknown"),
    }
    if mpressure_text is not None:
        mm = re.search(r"free percentage:\s*(\d+)%", mpressure_text)
        out["free_pct"] = int(mm.group(1)) if mm else None
    return out


def parse_ioreg_accel(text: str) -> dict:
    """`ioreg -r -d 1 -w0 -c IOAccelerator` -> GPU device utilisation and in-use memory.

    Only the exact keys are read ("In use system memory" is not the "(driver)" variant).
    """
    util = re.search(r'"Device Utilization %"\s*=\s*(\d+)', text)
    in_use = re.search(r'"In use system memory"\s*=\s*(\d+)', text)
    alloc = re.search(r'"Alloc system memory"\s*=\s*(\d+)', text)
    if not util and not in_use:
        raise ProbeError("ioreg: no PerformanceStatistics keys")
    return {
        "device_util_pct": int(util.group(1)) if util else None,
        "in_use_sys_mem_bytes": int(in_use.group(1)) if in_use else None,
        "alloc_sys_mem_bytes": int(alloc.group(1)) if alloc else None,
    }


_FOOT = re.compile(r"Footprint:\s*([\d.]+)\s*(B|KB|MB|GB|TB)\b")


def parse_footprint(text: str) -> dict:
    """`footprint -p PID` -> phys_footprint in bytes (header line)."""
    m = _FOOT.search(text)
    if not m:
        raise ProbeError("footprint: no Footprint header")
    return {"phys_footprint_bytes": _to_bytes(m.group(1), m.group(2))}


def parse_thermal(text: str) -> dict:
    """`pmset -g therm` -> CPU speed limit and whether a warning level is recorded.

    throttled is True only when the speed limit is below 100 or a warning level is recorded.
    """
    if not text.strip():
        raise ProbeError("pmset: empty output")
    lim = re.search(r"CPU_Speed_Limit\s*=\s*(\d+)", text)
    speed = int(lim.group(1)) if lim else None
    thermal_recorded = bool(
        re.search(r"thermal warning level", text, re.I)
    ) and not re.search(r"No thermal warning level has been recorded", text, re.I)
    perf_recorded = bool(
        re.search(r"performance warning level", text, re.I)
    ) and not re.search(r"No performance warning level has been recorded", text, re.I)
    throttled = bool(
        (speed is not None and speed < 100) or thermal_recorded or perf_recorded
    )
    return {
        "cpu_speed_limit_pct": speed,
        "thermal_warning_recorded": thermal_recorded,
        "performance_warning_recorded": perf_recorded,
        "throttled": throttled,
    }


_PM = {
    "cpu_power_mw": r"CPU Power:\s*([\d.]+)\s*mW",
    "gpu_power_mw": r"GPU Power:\s*([\d.]+)\s*mW",
    "ane_power_mw": r"ANE Power:\s*([\d.]+)\s*mW",
    "combined_power_mw": r"Combined Power \(CPU \+ GPU \+ ANE\):\s*([\d.]+)\s*mW",
    "gpu_freq_mhz": r"GPU HW active frequency:\s*([\d.]+)\s*MHz",
    "gpu_active_residency_pct": r"GPU HW active residency:\s*([\d.]+)%",
}


def parse_powermetrics(text: str) -> dict:
    """`powermetrics --samplers cpu_power,gpu_power` sample text -> mW / MHz / %.

    Only reached when `sudo -n` works. Keys absent from the text are None.
    """
    out = {k: None for k in _PM}
    for k, pat in _PM.items():
        m = re.search(pat, text)
        if m:
            out[k] = float(m.group(1))
    if out["combined_power_mw"] is None and out["cpu_power_mw"] is None:
        raise ProbeError("powermetrics: no power lines")
    return out


# ---------------------------------------------------------------------------
# Command helpers
# ---------------------------------------------------------------------------
def _run(argv: list[str], timeout: float = 5.0) -> str:
    try:
        cp = subprocess.run(
            argv, capture_output=True, text=True, timeout=timeout, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ProbeError(f"{argv[0]}: {type(exc).__name__}") from exc
    if cp.returncode != 0 and not cp.stdout.strip():
        raise ProbeError(f"{argv[0]}: rc={cp.returncode}")
    return cp.stdout


def probe_sudo_nopass() -> tuple[bool, str | None]:
    """True only if `sudo -n true` succeeds (no password prompt)."""
    try:
        cp = subprocess.run(
            ["sudo", "-n", "true"], capture_output=True, timeout=5, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"sudo probe failed: {type(exc).__name__}"
    if cp.returncode == 0:
        return True, None
    return False, "sudo -n needs a password (no NOPASSWD); powermetrics requires root"


def sha256_file(path: str) -> str | None:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
    except OSError:
        return None
    return h.hexdigest()


def tool_records(tools_json: str = TOOLS_JSON) -> dict:
    """Version, expected sha256 (TOOLS.json) and the sha256 of each binary on disk, per power tool.

    `sha256_match` is True only when the file on disk hashes to the TOOLS.json value.
    """
    try:
        with open(tools_json) as fh:
            tools = json.load(fh)["tools"]
    except (OSError, ValueError, KeyError) as exc:
        return {"error": f"TOOLS.json unreadable: {type(exc).__name__}"}
    out = {}
    for name in TOOL_NAMES:
        t = tools.get(name)
        if not isinstance(t, dict):
            out[name] = {"error": "not in TOOLS.json"}
            continue
        actual = sha256_file(t.get("binary", ""))
        out[name] = {
            "version": t.get("version"),
            "binary": t.get("binary"),
            "sudoless": t.get("sudoless"),
            "sha256_expected": t.get("binary_sha256"),
            "sha256_actual": actual,
            "sha256_match": actual is not None and actual == t.get("binary_sha256"),
        }
    return out


def swap_record() -> dict:
    """`sysctl vm.swapusage` as a record (bytes), or {"error": ...} when unreadable."""
    try:
        return parse_swapusage(_run(["sysctl", "vm.swapusage"]))
    except ProbeError as exc:
        return {"error": str(exc)}


def powermetrics_row(p: dict, t: float) -> dict:
    """parse_powermetrics() output (mW / MHz / %) -> the power row shape of fullcmp_power (W)."""

    def w(v):
        return None if v is None else v / 1000.0

    combined = p.get("combined_power_mw")
    return {
        "part": "power",
        "source": "powermetrics",
        "t": t,
        "gpu_w": w(p.get("gpu_power_mw")),
        "cpu_w": w(p.get("cpu_power_mw")),
        "ane_w": w(p.get("ane_power_mw")),
        "sys_w": None,  # powermetrics has no whole-system figure in this text
        "all_w": w(combined),
        "ram_w": None,
        "gpu_mhz": p.get("gpu_freq_mhz"),
        "gpu_active": None
        if p.get("gpu_active_residency_pct") is None
        else p["gpu_active_residency_pct"] / 100.0,
        "cpu_active": None,
        "ram_used": None,
        "swap_used": None,
        "cpu_temp_c": None,
        "gpu_temp_c": None,
    }


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------
@dataclass
class ProbeState:
    name: str
    available: bool
    reason: str | None
    interval_ms: int
    calls: int = 0
    errors: int = 0
    last_error: str | None = None
    cost_ms_total: float = 0.0
    cost_ms_max: float = 0.0
    source: str | None = None


@dataclass
class Sampler:
    pid: int
    port: int | None
    out: str
    interval_ms: int = 100
    marks_file: str | None = None
    probes: tuple = ALL_PROBES
    power_secondary: str | None = None  # None or "mactop" (sudoless, 1000 ms)
    _state: dict = field(default_factory=dict)
    _thread: threading.Thread | None = None
    _stop: threading.Event = field(default_factory=threading.Event)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _fh: object = None
    _marks_pos: int = 0
    _proc = None
    _prev_cpu: tuple | None = None
    _prev_t: float | None = None
    _power_proc = None  # sudo -n powermetrics child (only when sudo -n works)
    _pm_rows: list = field(default_factory=list)
    _power: pw.PowerSampler | None = None  # macmon (sudoless)
    _power2: pw.PowerSampler | None = None  # mactop (optional secondary)
    _flushed: dict = field(default_factory=dict)
    _power_stops: dict = field(default_factory=dict)
    _sudo_note: str | None = None
    _swap_start: dict | None = None
    _thermal_start: dict | None = None
    _thermal_end: dict | None = None
    _n_samples: int = 0

    # ---- lifecycle -------------------------------------------------------
    def start(self) -> Sampler:
        try:
            self._proc = psutil.Process(self.pid)
        except psutil.Error as exc:
            raise ProbeError(f"pid {self.pid} not found: {exc}") from exc
        self._fh = open(self.out, "w", buffering=1)  # noqa: SIM115 - closed in stop()
        self._init_probes()
        self._thermal_start = self._safe_thermal()
        self._swap_start = swap_record()
        self._write(
            {
                "part": "telemetry_header",
                "pid": self.pid,
                "port": self.port,
                "interval_ms": self.interval_ms,
                "slow_interval_ms": SLOW_INTERVAL_MS,
                "t0": time.time(),
                "pid_name": self._proc.name(),
                "page_bytes_note": "vmstat page size is read from each vm_stat output",
                "probes": {
                    n: {
                        "available": s.available,
                        "reason": s.reason,
                        "interval_ms": s.interval_ms,
                        **({"source": s.source} if s.source else {}),
                    }
                    for n, s in self._state.items()
                },
                "power_sudo_note": self._sudo_note,
                "tools": tool_records(),
                "swap_start": self._swap_start,
                "thermal_start": self._thermal_start,
                "net_scope": "lo0 aggregate (all loopback ports)",
                "disk_scope": "system-wide (macOS has no per-process io counters)",
                "footprint_source": "`footprint -p PID` header (phys_footprint)",
            }
        )
        if self.marks_file:
            open(self.marks_file, "a").close()
        self._thread = threading.Thread(
            target=self._loop, name="telemetry", daemon=True
        )
        self._thread.start()
        return self

    def stop(self) -> dict:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=10)
        for name, ps in (("power", self._power), ("power_secondary", self._power2)):
            if ps is not None:
                res = ps.stop()
                self._power_stops[name] = {
                    k: res[k] for k in ("status", "bad_lines", "exit_code", "killed")
                }
        if self._power_proc is not None:
            with contextlib.suppress(OSError):
                self._power_proc.terminate()
        self._flush_power()  # every power row the readers produced, after they stopped
        self._thermal_end = self._safe_thermal()
        footer = {
            "part": "telemetry_footer",
            "t1": time.time(),
            "n_samples": self._n_samples,
            "thermal_end": self._thermal_end,
            "swap_end": swap_record(),
            "power_stops": self._power_stops,
            "probe_stats": {
                n: {
                    "calls": s.calls,
                    "errors": s.errors,
                    "last_error": s.last_error,
                    "cost_ms_mean": round(s.cost_ms_total / s.calls, 3)
                    if s.calls
                    else None,
                    "cost_ms_max": round(s.cost_ms_max, 3),
                }
                for n, s in self._state.items()
            },
        }
        self._write(footer)
        if self._fh:
            self._fh.close()
            self._fh = None
        return footer

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()

    def mark(self, ev: str, **fields) -> None:
        """Record a harness event (request_start / first_token / end) at the current time."""
        row = {"part": "mark", "ev": ev, "t": time.time(), **fields}
        self._write(row)

    # ---- internals -------------------------------------------------------
    def _write(self, row: dict) -> None:
        with self._lock:
            if self._fh:
                self._fh.write(json.dumps(row, separators=(",", ":")) + "\n")

    def _avail(self, name: str) -> bool:
        st = self._state.get(name)
        return bool(st and st.available)

    def _init_probes(self) -> None:
        checks = {
            "proc": lambda: self._proc.memory_info(),
            "disk": lambda: psutil.disk_io_counters(),
            "net_lo0": lambda: psutil.net_io_counters(pernic=True)["lo0"],
            "vmstat": lambda: parse_vm_stat(_run(["vm_stat"])),
            "swap": lambda: parse_swapusage(_run(["sysctl", "vm.swapusage"])),
            "pressure": lambda: parse_pressure(
                _run(["sysctl", "kern.memorystatus_vm_pressure_level"])
            ),
            "gpu": lambda: parse_ioreg_accel(
                _run(["ioreg", "-r", "-d", "1", "-w0", "-c", "IOAccelerator"])
            ),
            "footprint": lambda: parse_footprint(
                _run(["footprint", "-p", str(self.pid)], timeout=10)
            ),
            "thermal": lambda: parse_thermal(_run(["pmset", "-g", "therm"])),
        }
        for name in self.probes:
            interval = self.interval_ms if name in FAST_PROBES else SLOW_INTERVAL_MS
            if name == POWER_PROBE:
                self._init_power()
                continue
            try:
                checks[name]()
                self._state[name] = ProbeState(name, True, None, interval)
            except Exception as exc:  # noqa: BLE001 - reason recorded, probe disabled
                self._state[name] = ProbeState(
                    name, False, f"start check failed: {exc}", interval
                )
        if POWER_PROBE in self.probes and self.power_secondary:
            self._init_power_secondary()

    def _init_power(self) -> None:
        """powermetrics only when `sudo -n` works; otherwise the sudoless macmon reader."""
        ok, why = probe_sudo_nopass()
        if ok:
            self._state[POWER_PROBE] = ProbeState(
                POWER_PROBE, True, None, SLOW_INTERVAL_MS, source="powermetrics"
            )
            self._start_powermetrics()
            return
        self._power = pw.PowerSampler(
            tool="macmon", interval_ms=SLOW_INTERVAL_MS
        ).start()
        ok = self._power.status == "running"
        self._state[POWER_PROBE] = ProbeState(
            POWER_PROBE,
            ok,
            None if ok else self._power.status,
            SLOW_INTERVAL_MS,
            source="macmon",
        )
        self._sudo_note = why

    def _init_power_secondary(self) -> None:
        self._power2 = pw.PowerSampler(
            tool=self.power_secondary, interval_ms=SLOW_INTERVAL_MS
        ).start()
        ok = self._power2.status == "running"
        self._state["power_secondary"] = ProbeState(
            "power_secondary",
            ok,
            None if ok else self._power2.status,
            SLOW_INTERVAL_MS,
            source=self.power_secondary,
        )

    def _flush_power(self) -> None:
        """Write power rows not yet written, as their own lines (the rows carry their own `t`)."""
        sources = []  # (key, part, snapshot): snapshots are taken without holding self._lock
        if self._power is not None:
            sources.append(("macmon", "power", self._power.snapshot))
        if self._power2 is not None:
            sources.append(("mactop", "power_secondary", self._power2.snapshot))
        sources.append(("powermetrics", "power", self._pm_snapshot))
        for key, part, snap in sources:
            rows = snap()
            done = self._flushed.get(key, 0)
            self._flushed[key] = len(rows)
            for r in rows[done:]:
                self._write({**r, "part": part})

    def _pm_snapshot(self) -> list[dict]:
        with self._lock:
            return list(self._pm_rows)

    def _start_powermetrics(self) -> None:
        st = self._state[POWER_PROBE]
        try:
            self._power_proc = subprocess.Popen(
                [
                    "sudo",
                    "-n",
                    "powermetrics",
                    "--samplers",
                    "cpu_power,gpu_power",
                    "-i",
                    "1000",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
        except OSError as exc:
            st.available, st.reason = False, f"powermetrics spawn failed: {exc}"
            return

        def reader():
            buf: list[str] = []
            for line in self._power_proc.stdout:
                if line.startswith("**** Processor usage") and buf:
                    self._consume_power("".join(buf))
                    buf = []
                buf.append(line)

        threading.Thread(target=reader, name="powermetrics", daemon=True).start()

    def _consume_power(self, text: str) -> None:
        with contextlib.suppress(ProbeError):
            row = powermetrics_row(parse_powermetrics(text), time.time())
            with self._lock:
                self._pm_rows.append(row)

    def _safe_thermal(self) -> dict | None:
        try:
            return parse_thermal(_run(["pmset", "-g", "therm"]))
        except ProbeError as exc:
            return {"error": str(exc)}

    def _call(self, name: str, fn):
        st = self._state[name]
        t0 = time.perf_counter()
        try:
            value = fn()
        except Exception as exc:  # noqa: BLE001 - a failed probe is null for this tick
            st.errors += 1
            st.last_error = f"{type(exc).__name__}: {exc}"
            value = None
        cost = (time.perf_counter() - t0) * 1000
        st.calls += 1
        st.cost_ms_total += cost
        st.cost_ms_max = max(st.cost_ms_max, cost)
        return value

    def _read_marks(self) -> list[dict]:
        if not self.marks_file:
            return []
        out = []
        try:
            with open(self.marks_file) as fh:
                fh.seek(self._marks_pos)
                for line in fh:
                    if not line.endswith("\n"):
                        break  # partial line: read it on the next tick
                    self._marks_pos += len(line.encode())
                    try:
                        m = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(m, dict) and m.get("ev") in MARK_EVENTS:
                        m.setdefault("t", time.time())
                        out.append({"part": "mark", **m})
        except OSError:
            pass
        return out

    def _sample_proc(self) -> dict:
        p = self._proc
        if not p.is_running():
            raise ProbeError("process exited")
        mi = p.memory_info()
        ct = p.cpu_times()
        ctx = p.num_ctx_switches()
        return {
            "rss_bytes": mi.rss,
            "cpu_user_s": ct.user,
            "cpu_sys_s": ct.system,
            "threads": p.num_threads(),
            "ctx_vol": ctx.voluntary,
            "ctx_invol": ctx.involuntary,
        }

    def _tick(self, now: float, due: dict) -> dict:
        row: dict = {}
        if self._avail("proc"):
            row["proc"] = self._call("proc", self._sample_proc)
            if (
                row["proc"] is not None
                and self._prev_cpu is not None
                and self._prev_t is not None
            ):
                dt = now - self._prev_t
                if dt > 0:
                    d = (row["proc"]["cpu_user_s"] + row["proc"]["cpu_sys_s"]) - (
                        self._prev_cpu[0] + self._prev_cpu[1]
                    )
                    row["proc"]["cpu_pct"] = round(100.0 * d / dt, 2)
            if row["proc"] is not None:
                self._prev_cpu = (row["proc"]["cpu_user_s"], row["proc"]["cpu_sys_s"])
                self._prev_t = now
        if self._avail("disk"):

            def disk():
                c = psutil.disk_io_counters()
                return {"read_bytes": c.read_bytes, "write_bytes": c.write_bytes}

            row["disk"] = self._call("disk", disk)
        if self._avail("net_lo0"):

            def net():
                c = psutil.net_io_counters(pernic=True)["lo0"]
                return {"recv_bytes": c.bytes_recv, "sent_bytes": c.bytes_sent}

            row["net_lo0"] = self._call("net_lo0", net)
        for name in SLOW_PROBES:
            if name in due and self._state[name].available:
                row[name] = self._call(name, due[name])
        row["t"] = (
            now  # power rows are written separately (_flush_power), on their own clock
        )
        return row

    def _slow_fns(self) -> dict:
        def pressure():
            sysctl = _run(["sysctl", "kern.memorystatus_vm_pressure_level"])
            try:
                mp = _run(["memory_pressure", "-Q"])
            except ProbeError:
                mp = None
            return parse_pressure(sysctl, mp)

        def footprint():
            return parse_footprint(_run(["footprint", "-p", str(self.pid)], timeout=10))

        return {
            "vmstat": lambda: parse_vm_stat(_run(["vm_stat"])),
            "swap": lambda: parse_swapusage(_run(["sysctl", "vm.swapusage"])),
            "pressure": pressure,
            "gpu": lambda: parse_ioreg_accel(
                _run(["ioreg", "-r", "-d", "1", "-w0", "-c", "IOAccelerator"])
            ),
            "footprint": footprint,
            "thermal": lambda: parse_thermal(_run(["pmset", "-g", "therm"])),
        }

    def _loop(self) -> None:
        slow_fns = self._slow_fns()
        start = time.monotonic()
        tick = 0
        next_tick = start
        while not self._stop.is_set():
            now = time.time()
            due = {}
            for name in SLOW_PROBES:
                st = self._state.get(name)
                if st is None:
                    continue
                if (
                    self._avail(name)
                    and (tick * self.interval_ms) % st.interval_ms == 0
                ):
                    due[name] = slow_fns[name]
            row = self._tick(now, due)
            for m in self._read_marks():
                self._write(m)
            self._flush_power()
            self._write({"part": "sample", **row})
            self._n_samples += 1
            if self._proc is not None and not self._proc.is_running():
                break  # the server exited: stop sampling it (footer still written)
            tick += 1
            next_tick += self.interval_ms / 1000.0
            delay = next_tick - time.monotonic()
            if delay > 0:
                self._stop.wait(delay)
            else:
                next_tick = time.monotonic()


# ---------------------------------------------------------------------------
# Summary (per request window and overall)
# ---------------------------------------------------------------------------
def _get(row: dict, path: str):
    cur = row
    for key in path.split("."):
        if not isinstance(cur, dict) or cur.get(key) is None:
            return None
        cur = cur[key]
    return cur


def _vals(rows, path):
    out = []
    for r in rows:
        v = _get(r, path)
        if v is not None:
            out.append(v)
    return out


def _power_window(power: list[dict], t0: float, t1: float, tokens: int | None) -> dict:
    """Power W (mean/peak) and energy J per source field over [t0, t1], plus J per output token.

    `power` is the power rows of the primary reader. Energy uses fullcmp_power.energy_j (left rule,
    carry-in at most 5 s old). Without any power row the values are None, never zero.
    """
    if not power:
        return {
            "power_w": None,
            "energy_j": None,
            "energy_j_per_token": None,
            "energy_note": "power unavailable",
        }
    win = [r for r in power if t0 <= r["t"] <= t1]
    watts = {}
    for k in ("gpu_w", "cpu_w", "ane_w", "sys_w"):
        vals = [r[k] for r in win if r.get(k) is not None]
        if vals:
            watts[k] = {
                "mean": round(sum(vals) / len(vals), 4),
                "peak": round(max(vals), 4),
            }
    energy = {k: pw.energy_j(power, t0, t1, k) for k in ENERGY_KEYS}
    per_tok = None
    if tokens:
        per_tok = {
            k: round(e / tokens, 6) if e is not None else None
            for k, e in energy.items()
        }
    note = None if win else "no power samples in window"
    return {
        "power_w": watts or None,
        "energy_j": energy,
        "energy_j_per_token": per_tok,
        "power_samples": len(win),
        "power_source": sorted({r.get("source") for r in win if r.get("source")})
        or None,
        **({"energy_note": note} if note else {}),
    }


def _window_stats(
    samples: list[dict],
    idle_row: dict | None,
    tokens: int | None,
    power: list[dict] | None = None,
    t0: float | None = None,
    t1: float | None = None,
) -> dict:
    stats: dict = {"n": len(samples)}
    rss = _vals(samples, "proc.rss_bytes")
    fp = _vals(samples, "footprint.phys_footprint_bytes")
    util = _vals(samples, "gpu.device_util_pct")
    cpu = _vals(samples, "proc.cpu_pct")
    stats["peak_rss_bytes"] = max(rss) if rss else None
    stats["peak_footprint_bytes"] = max(fp) if fp else None
    stats["gpu_util_peak_pct"] = max(util) if util else None
    stats["gpu_util_mean_pct"] = round(sum(util) / len(util), 2) if util else None
    stats["proc_cpu_mean_pct"] = round(sum(cpu) / len(cpu), 2) if cpu else None
    stats["threads_max"] = max(_vals(samples, "proc.threads"), default=None)
    stats["swap_used_delta_bytes"] = _delta(samples, "swap.used_bytes")
    stats["disk_read_delta_bytes"] = _delta(samples, "disk.read_bytes")
    stats["disk_write_delta_bytes"] = _delta(samples, "disk.write_bytes")
    stats["net_lo0_bytes"] = _delta(samples, "net_lo0.recv_bytes")
    stats["throttled"] = any(_get(r, "thermal.throttled") for r in samples) or None
    if idle_row is not None:
        stats["after_idle_rss_bytes"] = _get(idle_row, "proc.rss_bytes")
        stats["after_idle_footprint_bytes"] = _get(
            idle_row, "footprint.phys_footprint_bytes"
        )
        stats["after_idle_gpu_util_pct"] = _get(idle_row, "gpu.device_util_pct")
    if t0 is None or t1 is None:
        ts = [r["t"] for r in samples]
        t0, t1 = (min(ts), max(ts)) if ts else (0.0, 0.0)
    stats.update(_power_window(power or [], t0, t1, tokens))
    return stats


def _delta(samples, path):
    vs = _vals(samples, path)
    return (vs[-1] - vs[0]) if len(vs) >= 2 else None


def summarize(
    telemetry_rows: list[dict], marks: list[dict] | None = None, idle_s: float = 5.0
) -> dict:
    """Peak / after-idle / mean per request window, plus overall.

    telemetry_rows: "sample" rows and "power" rows (the primary reader; "power_secondary" is ignored
    for energy). marks: mark rows with ev in request_start / first_token / end and an optional "req"
    id (default "0") and "tokens" on end. A window is [request_start.t, end.t]: its power W, energy J
    per source field and J per token come from the power rows in that window. after-idle = the first
    sample at or after end.t + idle_s (the last sample if the series ends earlier). Without marks the
    whole series is one window.
    """
    samples = sorted(
        (r for r in telemetry_rows if r.get("part", "sample") == "sample" and "t" in r),
        key=lambda r: r["t"],
    )
    power = sorted(
        (
            r
            for r in telemetry_rows
            if r.get("part") == "power" and r.get("t") is not None
        ),
        key=lambda r: r["t"],
    )
    marks = marks or []
    windows = []
    by_req: dict[str, dict] = {}
    for m in marks:
        if m.get("ev") in MARK_EVENTS:
            by_req.setdefault(str(m.get("req", "0")), {}).setdefault(m["ev"], m)
    for req, evs in sorted(by_req.items()):
        if "request_start" not in evs or "end" not in evs:
            windows.append({"req": req, "error": "incomplete marks"})
            continue
        t0, t1 = evs["request_start"]["t"], evs["end"]["t"]
        win = [r for r in samples if t0 <= r["t"] <= t1]
        after = [r for r in samples if r["t"] >= t1 + idle_s]
        idle_row = after[0] if after else (samples[-1] if samples else None)
        tokens = evs["end"].get("tokens")
        w = {
            "req": req,
            "t_start": t0,
            "t_first_token": evs.get("first_token", {}).get("t"),
            "t_end": t1,
            "ttft_s": round(evs["first_token"]["t"] - t0, 4)
            if "first_token" in evs
            else None,
            **_window_stats(win, idle_row, tokens, power, t0, t1),
        }
        windows.append(w)
    overall = _window_stats(samples, samples[-1] if samples else None, None, power)
    if not marks:
        windows = [{"req": None, **overall}]
    return {"windows": windows, "overall": overall, "n_samples": len(samples)}


def read_telemetry_rows(path) -> list[dict]:
    """Every row of a telemetry jsonl, in file order (samples, power, marks, header, footer)."""
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def read_telemetry(path) -> tuple[dict | None, list[dict], list[dict], dict | None]:
    """Read a telemetry jsonl -> (header, samples, marks, footer). Power rows: read_telemetry_rows."""
    header = footer = None
    samples, marks = [], []
    with open(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            part = row.get("part")
            if part == "telemetry_header":
                header = row
            elif part == "telemetry_footer":
                footer = row
            elif part == "mark":
                marks.append(row)
            elif part == "sample":
                samples.append(row)
    return header, samples, marks, footer


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _cli(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Sample one server process into a telemetry jsonl."
    )
    p.add_argument("--pid", type=int, default=None)
    p.add_argument(
        "--spawn-sleep",
        type=float,
        default=None,
        metavar="S",
        help="self-test: sample a spawned `sleep S` (an idle target) instead of --pid",
    )
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--interval-ms", type=int, default=100)
    p.add_argument(
        "--out",
        default=None,
        help="telemetry jsonl (default: <out-dir>/<label>-<stamp>.jsonl)",
    )
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument(
        "--label", default="telemetry", help="file-name prefix for --out-dir"
    )
    p.add_argument("--log", default=None, help="run log (default: next to --out)")
    p.add_argument(
        "--marks", default=None, help="jsonl the harness appends mark events to"
    )
    p.add_argument("--duration", type=float, default=None, help="stop after S seconds")
    p.add_argument("--idle-s", type=float, default=5.0)
    p.add_argument(
        "--power-secondary",
        choices=("none", "mactop"),
        default="none",
        help="optional second sudoless power reader (1000 ms), rows as power_secondary",
    )
    a = p.parse_args(argv)
    child = None
    pid = a.pid
    if a.spawn_sleep is not None:
        child = subprocess.Popen(["sleep", str(a.spawn_sleep)])
        pid = child.pid
    if pid is None:
        p.error("need --pid or --spawn-sleep")
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = a.out or os.path.join(a.out_dir, f"{a.label}-{stamp}-pid{pid}.jsonl")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    log_path = a.log or os.path.splitext(out)[0] + ".log"
    secondary = None if a.power_secondary == "none" else a.power_secondary
    s = Sampler(pid, a.port, out, a.interval_ms, a.marks, power_secondary=secondary)
    try:
        s.start()
    except ProbeError as exc:
        print(f"telemetry: {exc}", file=sys.stderr)
        if child is not None:
            child.kill()
        return 2
    signal.signal(signal.SIGTERM, lambda *_: s._stop.set())
    t_end = time.monotonic() + a.duration if a.duration else None
    while not s._stop.is_set():
        if t_end is not None and time.monotonic() >= t_end:
            break
        if s._thread is not None and not s._thread.is_alive():
            break
        time.sleep(0.05)
    footer = s.stop()
    if child is not None:
        child.kill()
        child.wait()
    rows = read_telemetry_rows(out)
    header = next((r for r in rows if r.get("part") == "telemetry_header"), {})
    marks = [r for r in rows if r.get("part") == "mark"]
    summ = summarize(rows, marks, a.idle_s)
    text = json.dumps(
        {
            "out": out,
            "n_samples": footer["n_samples"],
            "probes": header.get("probes"),
            "tools": header.get("tools"),
            "swap_start": header.get("swap_start"),
            "swap_end": footer.get("swap_end"),
            "power_stops": footer.get("power_stops"),
            "overall": summ["overall"],
        },
        indent=1,
        default=str,
    )
    with open(log_path, "w") as fh:
        fh.write(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
