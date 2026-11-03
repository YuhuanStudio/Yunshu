#!/usr/bin/env python3
"""Quiet-machine gate for the full fair comparison (FULLCMP_SPEC.md; host CPU contention).

Yunshu decode is host-bound, so a foreign process that takes CPU during a cell can make the same
server run ~3x slower per round. Evidence (2026-11-03, preflight.md): one A/A rep and one
sampler_ab run were slow; the slow server had no sampler, and a stray ~25% CPU process appeared
only in those runs. mds_stores spikes reached 43%.

What this module does (read-only, never kills anything):
  * samples `ps -axo pid,ppid,pcpu,etime,time,command` once per interval over a window and reads
    `sysctl -n vm.loadavg` at each sample;
  * excludes the harness's own processes: this process, its ancestors (except pid 1; the gpuq job
    shell and worker are ancestors), its descendants (tfbench's server and its helpers), and any
    command line naming gpuq / fullcmp / tfbench;
  * measures each foreign process's CPU from its cumulative CPU time (`time` column) between its
    first and last sighting, so a bursty process is not diluted by ps's lifetime `pcpu` average.
    A process that starts and exits between two samples is not seen (interval 1 s by default);
  * passes only when BOTH hold (percent of one core):
      - no single foreign holder sustains >= `max_single_pct` (default 20). macOS system noise
        (WindowServer, kernel_task, mds_stores) is exempt from this rule only: it is still listed
        and still counted in the total;
      - total foreign CPU is < `max_total_pct` (default 150).
    The per-holder rule catches the slow run's ~25% stray process; the total rule catches many
    small holders adding up to a busy host.

The record lists EVERY foreign holder with a nonzero mean (`holders`, sorted by mean, name/pid/pct),
plus `violators` (holders that broke the single-holder rule) and the top five (`top`).

CLI:
    fullcmp_quiet.py check [--max-single-pct 20] [--max-total-pct 150] [--window-s 10]  one snapshot
    fullcmp_quiet.py gate  [--log FILE] [--retries 3] [--wait-s 30]     retries inside a job;
                                                                        exit 3 naming holders
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_MAX_SINGLE_PCT = 20.0
DEFAULT_MAX_TOTAL_PCT = 150.0
# macOS noise exempt from the single-holder rule only (still recorded and counted in total). Spotlight
# (mds / mds_stores) is NOT exempt: at 49% on a core it contaminated a host-bound decode (wire-1107
# kapitsa mixed quiet_pre, 2026-11-04). WindowServer and kernel_task are small and not controllable.
SYSTEM_NOISE = frozenset({"WindowServer", "kernel_task"})
DEFAULT_WINDOW_S = 10.0
DEFAULT_INTERVAL_S = 1.0
DEFAULT_RETRIES = 3
DEFAULT_WAIT_S = 30.0
TOP_N = 5
# Command lines of the harness's own infrastructure (never counted as foreign).
HARNESS_MARKERS = ("gpuq", "fullcmp", "tfbench")
PS_FORMAT = "pid=,ppid=,pcpu=,etime=,time=,command="
GATE_EXIT_BUSY = 3


class QuietError(RuntimeError):
    """The quiet-machine measurement itself failed (ps or sysctl unreadable)."""


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def parse_cputime(text: str) -> float:
    """ps `time` column ([DD-]hh:mm:ss[.ss] or mm:ss[.ss]) -> seconds."""
    text = text.strip()
    days = 0.0
    if "-" in text:
        d, text = text.split("-", 1)
        days = float(d)
    seconds = 0.0
    for part in text.split(":"):
        seconds = seconds * 60 + float(part)
    return days * 86400 + seconds


def parse_ps(text: str) -> dict[int, dict]:
    """`ps -axo PS_FORMAT` output -> {pid: row}. Malformed lines fail (no partial samples)."""
    procs: dict[int, dict] = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        fields = line.split(None, 5)
        if len(fields) < 6:
            raise QuietError(f"unparseable ps line: {line!r}")
        pid, ppid, pcpu, etime, cputime, command = fields
        procs[int(pid)] = {
            "pid": int(pid),
            "ppid": int(ppid),
            "pcpu": float(pcpu),
            "etime": etime,
            "cpu_s": parse_cputime(cputime),
            "command": command.strip(),
        }
    return procs


def parse_loadavg(text: str) -> float:
    """`sysctl -n vm.loadavg` ('{ 1.23 1.10 0.98 }') -> 1-minute load average."""
    nums = re.findall(r"\d+(?:\.\d+)?", text)
    if not nums:
        raise QuietError(f"unparseable vm.loadavg: {text!r}")
    return float(nums[0])


# ---------------------------------------------------------------------------
# Process classification
# ---------------------------------------------------------------------------


def harness_pids(procs: dict[int, dict], me: int | None = None) -> set[int]:
    """This process, its ancestors below pid 1 (gpuq job shell and worker), its descendants
    (tfbench's server and helpers), and every process whose command names the harness."""
    me = os.getpid() if me is None else me
    keep: set[int] = set()
    pid = me
    while pid in procs and pid not in keep and pid > 1:
        keep.add(pid)
        pid = procs[pid]["ppid"]
    keep.add(me)
    children: dict[int, list[int]] = {}
    for row in procs.values():
        children.setdefault(row["ppid"], []).append(row["pid"])
    stack = [me]
    while stack:
        for child in children.get(stack.pop(), []):
            if child not in keep:
                keep.add(child)
                stack.append(child)
    for row in procs.values():
        if any(marker in row["command"] for marker in HARNESS_MARKERS):
            keep.add(row["pid"])
    return keep


def display_name(command: str) -> str:
    first = command.split()[0] if command.split() else command
    return os.path.basename(first) or first


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


def run_ps() -> dict[int, dict]:
    res = subprocess.run(
        ["ps", "-axo", PS_FORMAT], capture_output=True, text=True, timeout=20
    )
    if res.returncode != 0:
        raise QuietError(f"ps failed (rc {res.returncode}): {res.stderr.strip()}")
    return parse_ps(res.stdout)


def read_loadavg() -> float:
    res = subprocess.run(
        ["sysctl", "-n", "vm.loadavg"], capture_output=True, text=True, timeout=10
    )
    if res.returncode != 0:
        raise QuietError(f"sysctl vm.loadavg failed (rc {res.returncode})")
    return parse_loadavg(res.stdout)


def check(
    max_single_pct: float = DEFAULT_MAX_SINGLE_PCT,
    max_total_pct: float = DEFAULT_MAX_TOTAL_PCT,
    window_s: float = DEFAULT_WINDOW_S,
    interval_s: float = DEFAULT_INTERVAL_S,
    *,
    run_ps_fn=None,
    loadavg_fn=None,
    clock=None,
    sleep=None,
) -> dict:
    """Sample the machine over `window_s` and return one record (never raises on a busy host).

    The record: ok, limit_pct, total_foreign_pct (sum of foreign process means), holders (>= 5%,
    by mean, name/pid/pct), top (top 5 foreign processes), load_avg {start, end, mean}, window_s,
    samples, at (wall clock). Raises QuietError only when the measurement itself cannot run."""
    if window_s <= 0 or interval_s <= 0:
        raise ValueError("window_s and interval_s must be positive")
    run_ps_fn = run_ps_fn or run_ps
    loadavg_fn = loadavg_fn or read_loadavg
    clock = clock or time.monotonic
    sleep = sleep or time.sleep
    steps = max(1, int(round(window_s / interval_s)))
    first: dict[int, float] = {}
    last: dict[int, float] = {}
    rows: dict[int, dict] = {}
    loads: list[float] = []
    t0 = clock()
    at = time.time()
    for i in range(steps + 1):
        procs = run_ps_fn()
        excluded = harness_pids(procs)
        for pid, row in procs.items():
            if pid in excluded:
                continue
            first.setdefault(pid, row["cpu_s"])
            last[pid] = row["cpu_s"]
            rows[pid] = row
        loads.append(loadavg_fn())
        if i < steps:
            sleep(interval_s)
    elapsed = clock() - t0
    if elapsed <= 0:
        raise QuietError("non-positive measurement window")
    pct = {pid: max(0.0, last[pid] - first[pid]) / elapsed * 100.0 for pid in first}
    total = sum(pct.values())
    ranked = sorted(pct, key=lambda p: (-pct[p], p))
    holders = [
        {"pid": p, "name": display_name(rows[p]["command"]), "pct": round(pct[p], 2)}
        for p in ranked
        if pct[p] > 0.0
    ]
    violators = [
        h
        for h in holders
        if h["pct"] >= max_single_pct and h["name"] not in SYSTEM_NOISE
    ]
    top = [
        {
            "pid": p,
            "name": display_name(rows[p]["command"]),
            "pct": round(pct[p], 2),
            "cmd": rows[p]["command"][:80],
        }
        for p in ranked[:TOP_N]
    ]
    return {
        "ok": total < max_total_pct and not violators,
        "max_single_pct": max_single_pct,
        "max_total_pct": max_total_pct,
        "total_foreign_pct": round(total, 2),
        "holders": holders,
        "violators": violators,
        "top": top,
        "load_avg": {
            "start": loads[0],
            "end": loads[-1],
            "mean": round(sum(loads) / len(loads), 2),
        },
        "window_s": round(elapsed, 2),
        "samples": steps + 1,
        "at": at,
    }


def snapshot(
    max_single_pct=DEFAULT_MAX_SINGLE_PCT,
    max_total_pct=DEFAULT_MAX_TOTAL_PCT,
    window_s=DEFAULT_WINDOW_S,
    **kw,
) -> dict:
    """check() for a per-cell record (quiet_pre / quiet_post): the same fields."""
    return check(max_single_pct, max_total_pct, window_s, **kw)


def describe_holders(record: dict) -> str:
    """Violators first (the reason for a BUSY verdict), then the top foreign holders."""
    named = record.get("violators") or []
    rest = [h for h in (record.get("holders") or [])[:TOP_N] if h not in named]
    shown = named + rest
    if not shown:
        return "no foreign holder"
    return ", ".join(f"{h['name']} (pid {h['pid']}, {h['pct']:.1f}%)" for h in shown)


def gate(
    max_single_pct: float = DEFAULT_MAX_SINGLE_PCT,
    max_total_pct: float = DEFAULT_MAX_TOTAL_PCT,
    window_s: float = DEFAULT_WINDOW_S,
    retries: int = DEFAULT_RETRIES,
    wait_s: float = DEFAULT_WAIT_S,
    *,
    log=None,
    sleep=None,
    **kw,
) -> dict:
    """Up to 1 + `retries` checks, `wait_s` apart (a bounded loop, not shell polling). Returns the
    final record with `attempts` (list of per-attempt totals) and `passed`. Every attempt is
    appended to `log` (jsonl) when given."""
    sleep = sleep or time.sleep
    attempts = []
    final = None
    for attempt in range(retries + 1):
        if attempt:
            sleep(wait_s)
        final = check(max_single_pct, max_total_pct, window_s, sleep=sleep, **kw)
        final["attempt"] = attempt + 1
        attempts.append(final["total_foreign_pct"])
        if log is not None:
            Path(log).parent.mkdir(parents=True, exist_ok=True)
            with open(log, "a") as fh:
                fh.write(json.dumps(final) + "\n")
        if final["ok"]:
            break
    final["attempts"] = attempts
    final["passed"] = bool(final["ok"])
    return final


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("check", "gate"):
        p = sub.add_parser(name)
        p.add_argument("--max-single-pct", type=float, default=DEFAULT_MAX_SINGLE_PCT)
        p.add_argument("--max-total-pct", type=float, default=DEFAULT_MAX_TOTAL_PCT)
        p.add_argument("--window-s", type=float, default=DEFAULT_WINDOW_S)
    sub.choices["gate"].add_argument("--retries", type=int, default=DEFAULT_RETRIES)
    sub.choices["gate"].add_argument("--wait-s", type=float, default=DEFAULT_WAIT_S)
    sub.choices["gate"].add_argument("--log", type=Path, default=None)
    a = ap.parse_args(argv)
    try:
        if a.cmd == "check":
            rec = check(a.max_single_pct, a.max_total_pct, a.window_s)
            print(json.dumps(rec, indent=2))
            return 0 if rec["ok"] else GATE_EXIT_BUSY
        rec = gate(
            a.max_single_pct,
            a.max_total_pct,
            a.window_s,
            retries=a.retries,
            wait_s=a.wait_s,
            log=a.log,
        )
    except QuietError as exc:
        print(f"quiet machine: measurement failed: {exc}", file=sys.stderr)
        return 1
    if rec["passed"]:
        print(
            f"quiet machine: foreign CPU {rec['total_foreign_pct']:.1f}% < {rec['max_total_pct']:.0f}%, "
            f"no holder >= {rec['max_single_pct']:.0f}% "
            f"(attempt {rec['attempt']})",
            flush=True,
        )
        return 0
    print(
        f"quiet machine: BUSY after {len(rec['attempts'])} attempts: foreign CPU "
        f"{rec['total_foreign_pct']:.1f}% (limit {rec['max_total_pct']:.0f}%), single-holder limit "
        f"{rec['max_single_pct']:.0f}%; holders: {describe_holders(rec)}",
        file=sys.stderr,
        flush=True,
    )
    return GATE_EXIT_BUSY


if __name__ == "__main__":
    raise SystemExit(main())
