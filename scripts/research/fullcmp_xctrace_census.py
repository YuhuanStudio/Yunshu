#!/usr/bin/env python3
"""Op census and host-sync counts from the Metal System Trace tables of one profile request.

Input: the XML exports of a finished .trace (`xctrace export --xpath .../table[@schema=...]`), as written
by fullcmp_xctrace.py. Pure parsing, CPU only, nothing here starts a server or touches the GPU.

What a Metal System Trace carries (checked on the dev0 trace of the e67748f3c validation):
- metal-gpu-intervals: one row per GPU encoder execution (channel Compute), with start, duration, the
  command buffer id and the encoder id. MLX does not label its encoders, so the label column is empty:
  there are no kernel names in this template. The census is therefore per encoder, grouped into classes by
  duration, with the count and the summed GPU ms of each class (op_census), not per kernel name.
- metal-application-command-buffer-submissions: one row per command buffer the application committed, with
  the committing thread and the number of encoders in it.

A "host sync" is the host waiting for GPU results before it can enqueue the next work. The trace has no wait
event, so it is read from the timeline: a GPU idle gap between consecutive encoder executions of the server
(gap >= IDLE_GAP_US) means the GPU ran dry and waited for the host. host_sync_count is those gaps per decode
step (a proxy, labelled as such in the row; the gap histogram is kept in the trace block).

Everything is reported per decode step (the profile request's step count comes from the engine log), over
the window that starts at the first command buffer after the prefill and ends at the last one.
"""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict

# duration classes of one GPU encoder execution, in microseconds: [lo, hi)
OP_CLASSES = (
    ("lt_20us", 0, 20),
    ("20_100us", 20, 100),
    ("100_500us", 100, 500),
    ("500us_2ms", 500, 2000),
    ("ge_2ms", 2000, float("inf")),
)
# a gap this long between two encoder executions of one server is a host stall (the host was not feeding)
IDLE_GAP_US = 50.0


def _pool(root) -> dict:
    return {el.get("id"): el for el in root.iter() if el.get("id") is not None}


def _value(el, pool: dict):
    """The text of an element (its fmt when it has no text), following ref= back-references."""
    if el is None:
        return None
    if el.get("ref") is not None:
        el = pool.get(el.get("ref"))
        if el is None:
            return None
    text = (el.text or "").strip()
    return text if text else el.get("fmt")


def read_table(xml_text: str) -> list[dict]:
    """Rows of an exported table as {mnemonic: value}, columns taken from the schema in order.

    A cell element whose subtree has a process or thread carries its pid / tid as extra keys
    (`<mnemonic>.pid`, `<mnemonic>.tid`) so rows can be filtered by pid without parsing the label."""
    root = ET.fromstring(xml_text)
    pool = _pool(root)
    schema = root.find(".//schema")
    mnemonics = [c.findtext("mnemonic") for c in schema.findall("col")] if schema is not None else []
    out = []
    for row in root.iter("row"):
        cells = list(row)
        rec = {}
        for i, cell in enumerate(cells):
            key = mnemonics[i] if i < len(mnemonics) else f"col{i}"
            rec[key] = _value(cell, pool)
            target = pool.get(cell.get("ref")) if cell.get("ref") is not None else cell
            if target is not None:
                pid = target.find("pid") if target.tag == "process" else None
                tid = target.find("tid") if target.tag == "thread" else None
                if pid is not None:
                    rec[f"{key}.pid"] = _value(pid, pool)
                if tid is not None:
                    rec[f"{key}.tid"] = _value(tid, pool)
        out.append(rec)
    return out


def _num(x) -> int | None:
    try:
        return int(str(x).replace(",", ""))
    except (TypeError, ValueError):
        return None


def server_encoders(gpu_rows: list[dict], pid: int) -> list[tuple[int, int, str]]:
    """(start_ns, duration_ns, cmdbuffer id) of every Compute encoder execution of the server pid."""
    out = []
    for r in gpu_rows:
        if _num(r.get("process.pid")) != pid or r.get("channel-name") != "Compute":
            continue
        s, d = _num(r.get("start")), _num(r.get("duration"))
        if s is None or d is None:
            continue
        out.append((s, d, str(r.get("cmdbuffer-id"))))
    out.sort()
    return out


def op_census(encoders: list[tuple[int, int, str]], steps: int | None) -> dict:
    """{class: {"count": n, "gpu_ms": ms, "count_per_step": x}} over the encoder executions given.
    count_per_step is None when the step count is unknown (never guessed)."""
    cls_count, cls_ns = Counter(), Counter()
    for _s, d, _cb in encoders:
        us = d / 1000.0
        for name, lo, hi in OP_CLASSES:
            if lo <= us < hi:
                cls_count[name] += 1
                cls_ns[name] += d
                break
    out = {}
    for name, _lo, _hi in OP_CLASSES:
        n = cls_count[name]
        out[name] = {
            "count": n,
            "gpu_ms": round(cls_ns[name] / 1e6, 3),
            "count_per_step": round(n / steps, 2) if steps else None,
        }
    return out


def server_commits(sub_rows: list[dict], pid: int) -> list[dict]:
    """Command buffer submissions of the server pid, time ordered: start, duration, thread, encoders, cb id."""
    out = []
    for r in sub_rows:
        if _num(r.get("process.pid")) != pid:
            continue
        s = _num(r.get("start"))
        if s is None:
            continue
        out.append(
            {
                "start": s,
                "duration": _num(r.get("duration")) or 0,
                "thread": r.get("thread.tid") or r.get("thread"),
                "encoders": _num(r.get("num-encoders")) or 0,
                "cb": str(r.get("cmdbuffer-id")),
            }
        )
    out.sort(key=lambda c: c["start"])
    return out


def idle_gaps_us(encoders: list[tuple[int, int, str]]) -> list[float]:
    """GPU idle gaps (microseconds) between consecutive encoder executions of the server: the time from the
    latest end so far to the next start, positive gaps only (overlapping encoders are concurrent work)."""
    gaps, latest_end = [], None
    for start, dur, _cb in encoders:
        if latest_end is not None and start > latest_end:
            gaps.append((start - latest_end) / 1000.0)
        latest_end = start + dur if latest_end is None else max(latest_end, start + dur)
    return gaps


def host_syncs(encoders: list[tuple[int, int, str]], threshold_us: float = IDLE_GAP_US) -> int:
    """Host syncs, read from the timeline: a GPU idle gap of at least threshold_us means the GPU ran dry and
    waited for the host (read-back, scheduling, the next eval) before more work arrived. The trace has no wait
    event, so this is a measured proxy, reported with its threshold."""
    return sum(1 for g in idle_gaps_us(encoders) if g >= threshold_us)


def decode_window(encoders: list[tuple[int, int, str]]) -> tuple[int, int] | None:
    """(start_ns, end_ns) of the decode phase of a cold request: from the first encoder after the longest GPU
    idle gap (the prefill/decode boundary: prefill runs as a few long encoders, then the host starts the
    round loop) to the last encoder start. None with fewer than two encoders."""
    if len(encoders) < 2:
        return None
    best, idx, latest_end = -1, 0, encoders[0][0] + encoders[0][1]
    for i in range(1, len(encoders)):
        gap = encoders[i][0] - latest_end
        if gap > best:
            best, idx = gap, i
        latest_end = max(latest_end, encoders[i][0] + encoders[i][1])
    return encoders[idx][0], encoders[-1][0]


def census_from_tables(
    gpu_xml: str,
    sub_xml: str,
    pid: int,
    steps: int | None,
    window: tuple[int, int] | None = None,
) -> dict:
    """The profile row fields: {"op_census": {...}, "host_sync_count": per step or None, "trace": {...}}.

    `steps` is the decode step count of the profile request (engine log); without it the per-step numbers
    stay None and the totals are still reported. `window` (ns) restricts to the decode phase; default: the
    decode window found by decode_window()."""
    enc = server_encoders(read_table(gpu_xml), pid)
    commits = server_commits(read_table(sub_xml), pid)
    if window is None:
        window = decode_window(enc)
    if window is not None:
        enc = [e for e in enc if window[0] <= e[0] <= window[1]]
        commits = [c for c in commits if window[0] <= c["start"] <= window[1]]
    gaps = idle_gaps_us(enc)
    hist = {f"ge_{int(t)}us": sum(1 for g in gaps if g >= t) for t in (50, 200, 500, 1000, 5000)}
    syncs = host_syncs(enc)
    return {
        "op_census": op_census(enc, steps),
        "host_sync_count": round(syncs / steps, 3) if steps else None,
        "trace": {
            "pid": pid,
            "encoder_executions": len(enc),
            "command_buffers": len(commits),
            "empty_command_buffers": sum(1 for c in commits if c["encoders"] == 0),
            "host_sync_threshold_us": IDLE_GAP_US,
            "idle_gaps": hist,
            "steps": steps,
            "window_ns": list(window) if window else None,
            "note": (
                "op_census is per encoder-duration class (this trace template carries no kernel names); "
                "host_sync_count is GPU idle gaps >= the threshold per decode step, a timeline proxy "
                "(the trace has no host wait event)"
            ),
        },
    }


def steps_from_log(text: str) -> int | None:
    """Decode steps of the profile request from the engine log: the 'round driver idle: N decode steps'
    line of the Phase B driver, or the dev0 'DFlash round profile: N rounds' line (largest N: the request
    that decoded the most)."""
    found = [int(n) for n in re.findall(r"round driver idle: (\d+) decode steps", text)]
    found += [int(n) for n in re.findall(r"DFlash round profile: (\d+) rounds", text)]
    return max(found) if found else None


if __name__ == "__main__":  # pragma: no cover - library; the capture driver calls census_from_tables
    import sys

    print(json.dumps({"usage": "import fullcmp_xctrace_census; census_from_tables(...)"}), file=sys.stderr)
    sys.exit(2)
