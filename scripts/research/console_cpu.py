"""CPU and memory cost of the console process while it polls a real engine at 1 Hz (gpuq job).

    gpuq submit ... -- python scripts/research/console_cpu.py MODEL --mode idle|busy --minutes 10

Starts `yunshu serve -m MODEL` (the console process comes with it), waits for both, then samples the
CONSOLE process every 5 s for ``--minutes`` with psutil: CPU% (process cpu_times delta over wall time)
and RSS. ``busy`` keeps one streamed request running at all times against the engine, so every poll
sees a live decode. The engine's own CPU is sampled the same way, for context. Writes a JSON summary.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
from console_real_server import get, stop_tree  # noqa: E402


def summarize(samples: list[dict]) -> dict:
    """Pure: CPU% and RSS statistics of a list of {cpu, rss_mib} samples."""
    cpu = [s["cpu"] for s in samples]
    rss = [s["rss_mib"] for s in samples]
    if not cpu:
        return {"n": 0}
    return {
        "n": len(cpu),
        "cpu_mean_pct": round(statistics.fmean(cpu), 3),
        "cpu_median_pct": round(statistics.median(cpu), 3),
        "cpu_p95_pct": round(
            sorted(cpu)[int(len(cpu) * 0.95) - 1 if len(cpu) > 1 else 0], 3
        ),
        "cpu_max_pct": round(max(cpu), 3),
        "rss_mib_start": rss[0],
        "rss_mib_end": rss[-1],
        "rss_mib_max": max(rss),
        "rss_growth_mib": round(rss[-1] - rss[0], 2),
    }


def sampler(pid: int, minutes: float, every: float = 5.0) -> list[dict]:
    import psutil

    p = psutil.Process(pid)
    out: list[dict] = []
    end = time.time() + minutes * 60
    last_cpu = sum(p.cpu_times()[:2])
    last_t = time.time()
    while time.time() < end:
        time.sleep(every)
        now = time.time()
        cpu = sum(p.cpu_times()[:2])
        out.append(
            {
                "t": round(now, 1),
                "cpu": 100.0 * (cpu - last_cpu) / max(now - last_t, 1e-6),
                "rss_mib": round(p.memory_info().rss / 1048576, 1),
            }
        )
        last_cpu, last_t = cpu, now
        print(
            f"sample pid={pid} cpu={out[-1]['cpu']:.2f}% rss={out[-1]['rss_mib']} MiB",
            flush=True,
        )  # keeps the job log alive
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("model")
    ap.add_argument("--mode", choices=("idle", "busy"), required=True)
    ap.add_argument("--minutes", type=float, default=10.0)
    ap.add_argument("--engine-port", type=int, default=18996)
    ap.add_argument("--console-port", type=int, default=18997)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    home = Path(
        tempfile.mkdtemp(prefix="console-cpu-", dir=os.environ.get("CONSOLE_REAL_TMP"))
    )
    env = {
        **os.environ,
        "PYTHONPATH": str(ROOT / "python"),
        "YUNSHU_CONSOLE_DB": str(home / "history.sqlite"),
        "YUNSHU_CONSOLE_PORT": str(args.console_port),
    }
    engine_url = f"http://127.0.0.1:{args.engine_port}"
    console_url = f"http://127.0.0.1:{args.console_port}"
    log = (home / "engine.log").open("wb")
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "yunshu_cli",
            "serve",
            "-m",
            args.model,
            "--port",
            str(args.engine_port),
        ],
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        t0 = time.time()
        while time.time() - t0 < 600:
            if proc.poll() is not None:
                print("FAIL engine exited early")
                return 1
            try:
                if get(f"{engine_url}/health", 3).get("status") and get(
                    f"{console_url}/v1/yunshu/console", 3
                ).get("up"):
                    break
            except Exception:
                time.sleep(1.0)
        else:
            print("FAIL engine or console did not come up")
            return 1
        engine_pid = get(f"{engine_url}/v1/yunshu/status")["pid"]
        kids = subprocess.run(
            ["pgrep", "-P", str(engine_pid)], capture_output=True, text=True
        ).stdout.split()
        console_pid = int(kids[0])
        model_id = get(f"{engine_url}/v1/models")["data"][0]["id"]
        stop = threading.Event()

        def load() -> None:
            body = json.dumps(
                {
                    "model": model_id,
                    "max_tokens": 400,
                    "stream": True,
                    "messages": [
                        {
                            "role": "user",
                            "content": "Write a long story about a lighthouse.",
                        }
                    ],
                }
            ).encode()
            while not stop.is_set():
                try:
                    req = urllib.request.Request(
                        f"{engine_url}/v1/chat/completions",
                        body,
                        {"content-type": "application/json"},
                    )
                    with urllib.request.urlopen(req, timeout=120) as r:
                        for _ in r:
                            if stop.is_set():
                                break
                except Exception:
                    time.sleep(0.5)

        worker = None
        if args.mode == "busy":
            worker = threading.Thread(target=load, daemon=True)
            worker.start()
            time.sleep(3.0)
        time.sleep(5.0)  # settle
        results: dict = {}
        box: dict = {}
        threads = [
            threading.Thread(
                target=lambda: box.__setitem__(
                    "console", sampler(console_pid, args.minutes)
                )
            ),
            threading.Thread(
                target=lambda: box.__setitem__(
                    "engine", sampler(engine_pid, args.minutes)
                )
            ),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        stop.set()
        if worker:
            worker.join(timeout=10)
        recorded = get(f"{console_url}/v1/yunshu/metrics/history?since={t0}")
        results = {
            "model": args.model,
            "mode": args.mode,
            "minutes": args.minutes,
            "console": summarize(box["console"]),
            "engine": summarize(box["engine"]),
            "history_rows": len((recorded.get("series") or {}).get("t") or []),
            "poll_s": get(f"{console_url}/v1/yunshu/console").get("poll_s"),
            "complete": True,
        }
        Path(args.out).write_text(json.dumps(results, indent=1))
        print(json.dumps(results, indent=1))
        print("PASS console cpu measurement")
        return 0
    finally:
        stop_tree(proc)
        log.close()


if __name__ == "__main__":
    raise SystemExit(main())
