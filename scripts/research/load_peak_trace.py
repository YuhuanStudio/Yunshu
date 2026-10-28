"""Load-phase MLX memory timeline: the app's start-up runs in this process while a sampler thread records
MLX active / cache / peak every SAMPLE_S, and every log line is stamped with the latest sample, so the step
that raises the peak is named by the log line it follows.

  python load_peak_trace.py --model PATH --out DIR [--env K=V ...]
Writes DIR/load_trace.json (samples, log marks, step table) and prints the step table.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
import time

SAMPLE_S = 0.02


def step_table(samples: list[dict], marks: list[dict]) -> list[dict]:
    """For each log mark: the sample just before it, and the highest active / peak seen since the
    previous mark (so a transient inside a step shows against the line that ends it)."""
    rows, prev_t = [], 0.0
    for m in marks:
        seg = [s for s in samples if prev_t < s["t"] <= m["t"]]
        rows.append(
            {
                "t": round(m["t"], 2),
                "msg": m["msg"],
                "active_gb": round(seg[-1]["active"] / 1e9, 2) if seg else None,
                "max_active_gb": round(
                    max((s["active"] for s in seg), default=0) / 1e9, 2
                ),
                "peak_gb": round(max((s["peak"] for s in seg), default=0) / 1e9, 2),
                "cache_gb": round(seg[-1]["cache"] / 1e9, 2) if seg else None,
            }
        )
        prev_t = m["t"]
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--env", action="append", default=[])
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for kv in a.env:
        k, v = kv.split("=", 1)
        os.environ[k] = v
    os.environ["YUNSHU_MODEL"] = a.model
    os.environ.setdefault("YUNSHU_AUTH_DISABLED", "true")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    tree = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, os.path.join(tree, "python"))
    import mlx.core as mx

    t0 = time.perf_counter()
    samples: list[dict] = []
    marks: list[dict] = []
    stop = threading.Event()

    def sampler():
        while not stop.is_set():
            samples.append(
                {
                    "t": time.perf_counter() - t0,
                    "active": mx.get_active_memory(),
                    "cache": mx.get_cache_memory(),
                    "peak": mx.get_peak_memory(),
                }
            )
            time.sleep(SAMPLE_S)

    class Mark(logging.Handler):
        def emit(self, record):
            marks.append(
                {"t": time.perf_counter() - t0, "msg": record.getMessage()[:140]}
            )

    threading.Thread(target=sampler, daemon=True).start()
    logging.getLogger().addHandler(Mark())
    logging.getLogger().setLevel(logging.INFO)
    from fastapi.testclient import TestClient

    from yunshu_gateway.main import app

    ok = False
    try:
        with TestClient(app):
            marks.append(
                {"t": time.perf_counter() - t0, "msg": "== startup complete =="}
            )
            time.sleep(0.5)
        ok = True
    finally:
        stop.set()
    table = step_table(samples, marks)
    with open(os.path.join(a.out, "load_trace.json"), "w") as f:
        json.dump({"table": table, "samples": samples}, f)
    prev = 0.0
    for r in table:
        if r["max_active_gb"] != prev or "==" in r["msg"]:
            print(
                r["t"],
                "active",
                r["active_gb"],
                "max_active",
                r["max_active_gb"],
                "peak",
                r["peak_gb"],
                "|",
                r["msg"],
            )
            prev = r["max_active_gb"]
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
