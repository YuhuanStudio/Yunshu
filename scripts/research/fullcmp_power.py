#!/usr/bin/env python3
"""Sudoless power sampler adapter for the full comparison (macmon / mactop pipe mode).

The telemetry module (fullcmp_telemetry.py) reads CPU/GPU/ANE power only through `sudo -n powermetrics`,
which is unavailable on the measurement machine. This adapter runs a sudoless IOReport reader as a child
process in JSON/pipe mode and turns its stream into power rows on the same clock as the telemetry marks
(`time.time()` epoch seconds, the `t` of a "mark" row).

Tools (binaries live under /Volumes/P5Plus/yunshu-build/tools, see TOOLS.json):
    macmon  `macmon pipe -i MS` prints one JSON object per interval (no root).
    mactop  `mactop -headless -interval MS` prints JSON samples (no root).

Rows produced (all power in watts, frequency in MHz, memory in bytes, temperature in deg C; None = absent):
    {"part": "power", "source": tool, "t": epoch, "gpu_w", "cpu_w", "ane_w", "sys_w", "all_w", "ram_w",
     "gpu_mhz", "gpu_active", "cpu_active", "ram_used", "swap_used", "cpu_temp_c", "gpu_temp_c"}
`t` is the tool's own sample timestamp when it has one (macmon emits ISO-8601), else the receipt time.

Degrades to a status of "unavailable (reason)" when the binary is missing, the child exits early, or its
output is not parseable; it never invents a value.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime

TOOLS_DIR = "/Volumes/P5Plus/yunshu-build/tools"
BINARIES = {
    "macmon": f"{TOOLS_DIR}/macmon-v0.10.0/macmon",
    "mactop": f"{TOOLS_DIR}/mactop-0.2.7/mactop",
}

FIELDS = (
    "gpu_w",
    "cpu_w",
    "ane_w",
    "sys_w",
    "all_w",
    "ram_w",
    "gpu_mhz",
    "gpu_active",
    "cpu_active",
    "ram_used",
    "swap_used",
    "cpu_temp_c",
    "gpu_temp_c",
)
POWER_FIELDS = ("gpu_w", "cpu_w", "ane_w", "sys_w", "all_w", "ram_w")


def _num(v):
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def _epoch(ts) -> float | None:
    if not isinstance(ts, str):
        return None
    try:
        return datetime.fromisoformat(ts).timestamp()
    except ValueError:
        return None


def parse_macmon(obj: dict, receipt_t: float | None = None) -> dict | None:
    """One `macmon pipe` JSON object -> power row, or None if it has no power fields at all."""
    if not isinstance(obj, dict):
        return None
    gpu = obj.get("gpu_usage")
    mem = obj.get("memory") if isinstance(obj.get("memory"), dict) else {}
    temp = obj.get("temp") if isinstance(obj.get("temp"), dict) else {}
    row = {
        "gpu_w": _num(obj.get("gpu_power")),
        "cpu_w": _num(obj.get("cpu_power")),
        "ane_w": _num(obj.get("ane_power")),
        "sys_w": _num(obj.get("sys_power")),
        "all_w": _num(obj.get("all_power")),
        "ram_w": _num(obj.get("ram_power")),
        "gpu_mhz": _num(gpu[0]) if isinstance(gpu, list) and len(gpu) == 2 else None,
        "gpu_active": _num(obj.get("gpu_active_ratio")),
        "cpu_active": _num(obj.get("cpu_active_ratio")),
        "ram_used": _num(mem.get("ram_usage")),
        "swap_used": _num(mem.get("swap_usage")),
        "cpu_temp_c": _num(temp.get("cpu_temp_avg")),
        "gpu_temp_c": _num(temp.get("gpu_temp_avg")),
    }
    if all(row[k] is None for k in POWER_FIELDS):
        return None
    t = _epoch(obj.get("timestamp"))
    if t is None:
        t = receipt_t
    return {"part": "power", "source": "macmon", "t": t, **row}


def parse_mactop(obj: dict, receipt_t: float | None = None) -> dict | None:
    """One `mactop -headless` sample -> power row, or None without power fields.

    mactop timestamps have one-second resolution, so `t` is the receipt time. Its cpu_power reads 0
    on this M5 Max, so the value is kept as reported (it is not a measurement of CPU power here).
    """
    if not isinstance(obj, dict) or not isinstance(obj.get("soc_metrics"), dict):
        return None
    soc = obj["soc_metrics"]
    mem = obj.get("memory") if isinstance(obj.get("memory"), dict) else {}

    def pct(v):
        v = _num(v)
        return None if v is None else v / 100.0

    row = {
        "gpu_w": _num(soc.get("gpu_power")),
        "cpu_w": _num(soc.get("cpu_power")),
        "ane_w": _num(soc.get("ane_power")),
        "sys_w": _num(soc.get("system_power")),
        "all_w": _num(soc.get("total_power")),
        "ram_w": _num(soc.get("dram_power")),
        "gpu_mhz": _num(soc.get("gpu_freq_mhz")),
        "gpu_active": pct(obj.get("gpu_usage")),
        "cpu_active": pct(obj.get("cpu_usage")),
        "ram_used": _num(mem.get("used")),
        "swap_used": _num(mem.get("swap_used")),
        "cpu_temp_c": _num(soc.get("cpu_temp")),
        "gpu_temp_c": _num(soc.get("gpu_temp")),
    }
    if all(row[k] is None for k in POWER_FIELDS):
        return None
    return {"part": "power", "source": "mactop", "t": receipt_t, **row}


PARSERS = {"macmon": parse_macmon, "mactop": parse_mactop}


def _decode(line: str, tool: str):
    """One stdout line -> JSON object, or None. mactop streams a JSON array: '[{..}', ',{..}', ']'."""
    s = line.strip()
    if tool == "mactop":
        s = s.lstrip("[,").rstrip("]").strip()
    if not s:
        return None
    return json.loads(s)


def parse_stream(
    text: str, tool: str = "macmon", receipt_t: float | None = None
) -> tuple[list[dict], int]:
    """Parse a captured pipe-mode stream -> (power rows, number of lines rejected).

    receipt_t is the time given to rows whose tool has no usable timestamp (mactop).
    """
    rows, bad = [], 0
    for line in text.splitlines():
        try:
            obj = _decode(line, tool)
        except json.JSONDecodeError:
            bad += 1
            continue
        if obj is None:
            continue
        row = PARSERS[tool](obj, receipt_t)
        if row is None or row["t"] is None:
            bad += 1
            continue
        rows.append(row)
    return rows, bad


# ---------------------------------------------------------------------------
# Child process
# ---------------------------------------------------------------------------
def _argv(tool: str, interval_ms: int, binary: str) -> list[str]:
    if tool == "macmon":
        return [binary, "pipe", "-i", str(int(interval_ms))]
    if tool == "mactop":
        return [binary, "-headless", "-interval", str(int(interval_ms))]
    raise ValueError(f"unknown power tool: {tool}")


@dataclass
class PowerSampler:
    """Runs one sudoless reader as a child and collects its rows.

    start() never raises: check `.status` ("running", "stopped", or "unavailable (reason)").
    The child is started by this object only; stop() terminates it and kills it (kill -9) only if it
    does not exit within `grace_s`.
    """

    tool: str = "macmon"
    interval_ms: int = 1000
    binary: str | None = None
    grace_s: float = 2.0
    rows: list[dict] = field(default_factory=list)
    status: str = "stopped"
    bad_lines: int = 0
    _proc: subprocess.Popen | None = None
    _stopping: bool = False
    _thread: threading.Thread | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def start(self) -> PowerSampler:
        if self.tool not in PARSERS:
            self.status = f"unavailable (unknown tool {self.tool})"
            return self
        binary = self.binary or BINARIES.get(self.tool, "")
        if not os.path.isfile(binary) or not os.access(binary, os.X_OK):
            self.status = f"unavailable (binary missing: {binary})"
            return self
        try:
            self._proc = subprocess.Popen(
                _argv(self.tool, self.interval_ms, binary),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
            )
        except OSError as exc:
            self.status = f"unavailable (spawn failed: {type(exc).__name__})"
            return self
        self.status = "running"
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()
        return self

    def _read(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        parser = PARSERS[self.tool]
        for line in self._proc.stdout:
            try:
                obj = _decode(line, self.tool)
            except json.JSONDecodeError:
                with self._lock:
                    self.bad_lines += 1
                continue
            if obj is None:
                continue
            row = parser(obj, time.time())
            with self._lock:
                if row is None:
                    self.bad_lines += 1
                else:
                    self.rows.append(row)
        # stdout closed: the child exited on its own
        if self.status == "running" and not self._stopping:
            try:
                rc = self._proc.wait(timeout=self.grace_s)
            except subprocess.TimeoutExpired:
                rc = None
            with self._lock:
                no_rows = "" if self.rows else ", no rows"
                self.status = f"unavailable (child exited rc={rc}{no_rows})"

    def mark(self) -> float:
        return time.time()

    def snapshot(self) -> list[dict]:
        with self._lock:
            return list(self.rows)

    def stop(self) -> dict:
        """Stop the child. Returns {"status", "rows", "bad_lines", "exit_code", "killed"}."""
        killed = False
        rc = None
        p = self._proc
        self._stopping = True
        if p is not None:
            if p.poll() is None:
                p.terminate()
                try:
                    rc = p.wait(timeout=self.grace_s)
                except subprocess.TimeoutExpired:
                    p.kill()  # SIGKILL: only the child this object started
                    killed = True
                    rc = p.wait()
            else:
                rc = p.returncode
        if self._thread is not None:
            self._thread.join(timeout=self.grace_s)
        if self.status == "running":
            self.status = "stopped"
        return {
            "status": self.status,
            "rows": self.snapshot(),
            "bad_lines": self.bad_lines,
            "exit_code": rc,
            "killed": killed,
        }

    def __enter__(self) -> PowerSampler:
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------
def _window(rows: list[dict], t0: float, t1: float) -> list[dict]:
    return sorted((r for r in rows if t0 <= r["t"] <= t1), key=lambda r: r["t"])


def energy_j(
    rows: list[dict], t0: float, t1: float, key: str = "all_w", max_carry_s: float = 5.0
) -> float | None:
    """Integral of power over [t0, t1] (joules).

    Left rule: a sample's power holds until the next sample. The last sample before t0 carries into
    the window when it is at most max_carry_s old, so [t0, first sample] is not dropped. The last
    sample holds to t1 only
    when the window has no later sample, which is the same rule applied to the window end. Returns
    None when there is no sample at or before the window end (nothing is estimated)."""
    pts = sorted(
        ((r["t"], r[key]) for r in rows if r.get(key) is not None and r["t"] <= t1),
        key=lambda x: x[0],
    )
    # keep the last sample before t0 (it carries into the window) and every sample inside it
    before = [x for x in pts if x[0] < t0]
    inside = [x for x in pts if t0 <= x[0]]
    carry = before[-1] if before and t0 - before[-1][0] <= max_carry_s else None
    seq = ([carry] if carry else []) + inside
    if not seq:
        return None
    total = 0.0
    for i, (t, p) in enumerate(seq):
        end = seq[i + 1][0] if i + 1 < len(seq) else t1
        total += p * max(0.0, min(end, t1) - max(t, t0))
    return round(total, 3)


def summarize(
    rows: list[dict], t0: float, t1: float, tokens: int | None = None
) -> dict:
    """Mean / peak per power and utilisation field over [t0, t1], energy (J) per power field,
    and energy per output token (J/token, all_w) when `tokens` > 0."""
    win = _window(rows, t0, t1)
    out: dict = {"n_samples": len(win), "window_s": round(t1 - t0, 3)}
    if not win:
        out["error"] = "no power samples in window"
        return out
    stats = {}
    for k in FIELDS:
        vals = [r[k] for r in win if r.get(k) is not None]
        if vals:
            stats[k] = {
                "mean": round(sum(vals) / len(vals), 4),
                "peak": round(max(vals), 4),
            }
    out["stats"] = stats
    out["energy_j"] = {k: energy_j(rows, t0, t1, k) for k in POWER_FIELDS if k in stats}
    if tokens:
        e = out["energy_j"].get("all_w")
        out["j_per_token"] = round(e / tokens, 6) if e is not None else None
    return out


def unavailable_summary(reason: str) -> dict:
    return {"error": f"unavailable ({reason})"}
