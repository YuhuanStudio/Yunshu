"""Metrics-only tfbench adapter and fail-closed receipt checks for ANE interference."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

_CONTENT_KEYS = {"text", "_text", "head", "lp_tokens", "messages", "prompt", "content"}


def metrics_only(value):
    if isinstance(value, dict):
        return {k: metrics_only(v) for k, v in value.items() if k not in _CONTENT_KEYS}
    if isinstance(value, list):
        return [metrics_only(v) for v in value]
    return value


def install_adapter(bench):
    send = bench.send

    def timed_send(*args, **kwargs):
        kwargs["capture_timestamps"] = True
        result = send(*args, **kwargs)
        result["_end_at_s"] = time.perf_counter()
        return result

    def emit(out, **record):
        record = metrics_only({**bench.META, **record})
        record["device"] = os.environ.get("GPUQ_DEVICE", "m5")
        record["contended"] = bench.was_contended()
        out.write(json.dumps(record) + "\n")
        out.flush()
        print(
            json.dumps({k: v for k, v in record.items() if k not in {"cmd", "env"}})[
                :300
            ],
            flush=True,
        )

    bench.send = timed_send
    bench.emit = emit


def read_receipt(path: Path):
    records = [
        json.loads(line) for line in path.read_text().splitlines() if line.strip()
    ]
    if not records or records[-1].get("complete") is not True:
        raise ValueError("missing final complete record")
    sessions = [r for r in records if r.get("part") == "session"]
    if not sessions or any(r.get("engaged_spec_mode") != "dflash" for r in sessions):
        raise ValueError("DFlash engagement not proven")
    rows = [r for r in records if r.get("part") == "decode"]
    if len(rows) != 2 or {r.get("kind") for r in rows} != {"prose", "code"}:
        raise ValueError("expected two complete code/prose cells")
    for row in rows:
        if row.get("cached") != 0 or row.get("phase") != "cold":
            raise ValueError("cached/non-cold timing row")
        if row.get("ct") != 512 or row.get("finish") != "length":
            raise ValueError("incomplete decode")
        if not row.get("sha") or not row.get("_first_at_s") or not row.get("_end_at_s"):
            raise ValueError("missing digest or decode window")
        if row["_end_at_s"] <= row["_first_at_s"]:
            raise ValueError("invalid decode window")
    return rows


def rate_in_window(samples, start, end):
    """Weighted calls/sec over overlapping one-second samples; None without coverage."""
    calls, seconds = 0.0, 0.0
    for sample in samples:
        lo, hi, count = sample["start_s"], sample["end_s"], sample["calls"]
        if hi <= lo or count < 0:
            raise ValueError("invalid load sample")
        overlap = max(0.0, min(end, hi) - max(start, lo))
        calls += count * overlap / (hi - lo)
        seconds += overlap
    return calls / seconds if seconds else None


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "check":
        p = argparse.ArgumentParser()
        p.add_argument("receipt", type=Path)
        p.add_argument("--base", type=Path)
        a = p.parse_args(sys.argv[2:])
        rows = read_receipt(a.receipt)
        if a.base:
            base = {r["kind"]: r["sha"] for r in read_receipt(a.base)}
            if any(r["sha"] != base[r["kind"]] for r in rows):
                raise ValueError("background ANE changed GPU output digest")
        print(
            json.dumps({"complete": True, "cells": len(rows), "cached": 0}), flush=True
        )
        return
    import tfbench

    install_adapter(tfbench)
    tfbench.main()


if __name__ == "__main__":
    main()
