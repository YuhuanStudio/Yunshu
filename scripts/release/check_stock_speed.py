"""Speed floor: a running Yunshu server must decode at least as fast as stock mlx-vlm on the same checkpoint.

The gate runs this for families that only the generic layer serves (Gemma 4) and for Flash-Next, where a
slowdown against the upstream path would otherwise go unseen. Same prompt, greedy, thinking off, the same
streaming client against both servers (stock ``python -m mlx_vlm.server`` is started here on its own port and
killed on exit); one untimed warm-up per case, the median of ``--repeats``, the two sides alternating which goes
first. ratio = yunshu / stock decode tok/s per case; FAIL when the geometric mean is below ``--min-geomean`` or any
case is below ``--min-ratio``. Text agreement is recorded, not required: batch-invariant kernels may differ from
stock in near-ties.

    python scripts/release/check_stock_speed.py --url http://127.0.0.1:18764 --model <ckpt> --served-name <name> \\
        --output runs/stock-speed.json

Prints one JSON summary line; exit 0 = PASS, 1 = FAIL, 2 = could not measure, 3 = CONTENDED
(``--retry-contended`` waits for a quiet CPU window and retries once, as check_server_path does).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import signal
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for sub in ("scripts/dev", "scripts/release", "scripts/research"):
    sys.path.insert(0, str(ROOT / sub))

from check_server_path import geometric_mean, judge  # noqa: E402
from gpuq_contention import (  # noqa: E402
    server_path_attempts,
    wait_for_quiet,
    was_contended,
)

STOCK_PY = os.environ.get(
    "STOCK_MLXVLM_PY",
    "/Volumes/P5Plus/yunshu-build/competitors/venvs/mlx-vlm-0.7.11/bin/python",
)


def case_ratio(yunshu: list[float], stock: list[float]) -> float | None:
    """Median yunshu tok/s over median stock tok/s; None when either side never produced a rate."""
    ys, ss = [x for x in yunshu if x], [x for x in stock if x]
    if not ys or not ss:
        return None
    return statistics.median(ys) / statistics.median(ss)


def verdict(cases: list[dict], min_case: float, min_geomean: float) -> str:
    """PASS / FAIL / ERROR from the per-case ratios; a case with no rate on either side is an error."""
    ratios = [c["ratio"] for c in cases]
    if not cases or any(r is None for r in ratios):
        return "ERROR"
    return judge(ratios, True, True, min_case, min_geomean)


def start_stock(model: str, port: int, log: Path) -> subprocess.Popen:
    cmd = [STOCK_PY, "-m", "mlx_vlm.server", "--model", model, "--host", "127.0.0.1"]
    cmd += ["--port", str(port)]
    return subprocess.Popen(
        cmd, stdout=log.open("wb"), stderr=subprocess.STDOUT, start_new_session=True
    )


def stop(proc: subprocess.Popen) -> None:
    """kill -9 the stock server's whole process group: nothing may outlive the check."""
    with contextlib.suppress(ProcessLookupError):
        os.killpg(proc.pid, signal.SIGKILL)
    proc.wait()


def wait_ready(url: str, proc: subprocess.Popen, timeout: float = 900.0) -> None:
    end = time.time() + timeout
    while time.time() < end:
        if proc.poll() is not None:
            raise RuntimeError(f"stock server exited rc={proc.returncode}")
        try:
            urllib.request.urlopen(url + "/health", timeout=2).read()
            return
        except Exception:
            try:
                urllib.request.urlopen(url + "/v1/models", timeout=2).read()
                return
            except Exception:
                time.sleep(2)
    raise RuntimeError("stock server not ready")


def measure(a) -> dict:
    from probe_server_path import messages_for, run_http

    started = time.time()
    port = a.stock_port
    surl = f"http://127.0.0.1:{port}"
    proc = start_stock(a.model, port, a.stock_log)
    cases = []
    try:
        wait_ready(surl, proc)
        for ctx in a.contexts:
            for task in a.tasks:
                msgs = messages_for(task, ctx)
                warm_y = run_http(a.url, a.served_name, msgs, a.max_tokens)
                warm_s = run_http(surl, a.model, msgs, a.max_tokens)
                yr, sr = [], []
                for rep in range(a.repeats):
                    order = (
                        [("y", a.url, a.served_name), ("s", surl, a.model)]
                        if rep % 2 == 0
                        else [("s", surl, a.model), ("y", a.url, a.served_name)]
                    )
                    for who, url, name in order:
                        r = run_http(url, name, msgs, a.max_tokens)
                        (yr if who == "y" else sr).append(r["decode_tps"] or 0.0)
                ratio = case_ratio(yr, sr)
                cases.append(
                    {
                        "task": task,
                        "context": ctx,
                        "yunshu_tps": round(statistics.median(yr), 2),
                        "stock_tps": round(statistics.median(sr), 2),
                        "ratio": None if ratio is None else round(ratio, 3),
                        "same_text": warm_y["text"] == warm_s["text"],
                    }
                )
    finally:
        stop(proc)
    ratios = [c["ratio"] for c in cases if c["ratio"] is not None]
    g = geometric_mean(ratios)
    return {
        "status": verdict(cases, a.min_ratio, a.min_geomean),
        "contended": was_contended(started, time.time()),
        "started": started,
        "ended": time.time(),
        "model": a.model,
        "stock_python": STOCK_PY,
        "worst_ratio": min(ratios) if ratios else None,
        "min_ratio": a.min_ratio,
        "geomean_ratio": None if g is None else round(g, 4),
        "min_geomean": a.min_geomean,
        "same_text": all(c["same_text"] for c in cases),
        "cases": cases,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--url", required=True, help="the running Yunshu server")
    ap.add_argument("--model", required=True, help="checkpoint dir both engines load")
    ap.add_argument("--served-name", required=True)
    ap.add_argument("--stock-port", type=int, default=18999)
    ap.add_argument("--stock-log", type=Path, default=Path("/dev/null"))
    ap.add_argument("--tasks", nargs="+", default=["code", "prose"])
    ap.add_argument("--contexts", type=int, nargs="+", default=[0, 8192])
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--min-ratio", type=float, default=0.90)
    ap.add_argument("--min-geomean", type=float, default=0.97)
    ap.add_argument("--output", type=Path)
    ap.add_argument("--retry-contended", action="store_true")
    a = ap.parse_args(argv)
    try:
        if a.retry_contended:
            summary = server_path_attempts(
                lambda: measure(a),
                before_retry=lambda: wait_for_quiet(os.getpgrp()),
            )
        else:
            summary = measure(a)
    except Exception as exc:  # could not measure: never a pass
        summary = {"status": "ERROR", "error": f"{type(exc).__name__}: {exc}"}
    if a.output:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)
    return {"PASS": 0, "FAIL": 1, "CONTENDED": 3}.get(summary["status"], 2)


if __name__ == "__main__":
    sys.exit(main())
