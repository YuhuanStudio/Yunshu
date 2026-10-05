"""Server-side mixed-length workload for one arm (run inside one gpuq job).

Starts its own server (kill -9 on exit), then for c in 2/4/8 sends c requests
with prompt lengths drawn from 1K-8K, alternating code / prose bodies, arrivals
staggered over STAGGER_S seconds (seeded, so both arms get the same workload).
Records per-request TTFT and decode tok/s and per-cell mean / p90 TTFT and
aggregate tok/s. Exits nonzero on any failed request.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import random
import signal
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
MODEL = "/Volumes/P5Plus/models/Jundot/Qwen3.8-27B-oQ4e-mtp"
CORPORA = Path(
    "/Users/yuhuan/Documents/YuhuanStudio/Yunshu/reference/omlx/omlx/admin/bench_corpora"
)
LENGTHS = [1024, 2048, 4096, 8192]
STAGGER_S = 1.5
CONCURRENCY = [2, 4, 8]
SERVE = (
    "import sys, uvicorn;"
    "uvicorn.run('yunshu_gateway.main:app', host='127.0.0.1', port=int(sys.argv[1]))"
)


def workload(c: int, rep: int) -> list[dict]:
    """Same for both arms: (length, kind, start offset) per request."""
    rng = random.Random(1000 * c + rep)
    reqs = [
        {
            "pp": rng.choice(LENGTHS),
            "kind": "code_python" if i % 2 == 0 else "novel_en",
            "at": round(rng.uniform(0, STAGGER_S), 3),
        }
        for i in range(c)
    ]
    return sorted(reqs, key=lambda r: r["at"])


def p90(xs: list[float]) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(0.9 * len(xs)))]


def summarize(rows: list[dict], wall: float) -> dict:
    ttft = [r["ttft_s"] for r in rows]
    dec = [r["decode_tps"] for r in rows if r.get("decode_tps")]
    total = sum(r["completion_tokens"] for r in rows)
    return {
        "n": len(rows),
        "mean_ttft_s": round(statistics.mean(ttft), 3),
        "p90_ttft_s": round(p90(ttft), 3),
        "aggregate_tps": round(total / wall, 1),
        "mean_decode_tps": round(statistics.mean(dec), 1) if dec else None,
        "wall_s": round(wall, 2),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["off", "routed"])
    ap.add_argument("--rep", type=int, default=0)
    ap.add_argument("--port", type=int, default=18992)
    ap.add_argument("--tg", type=int, default=128)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    root = HERE.parents[1]
    env = {
        **os.environ,
        "PYTHONPATH": str(root / "python"),
        "YUNSHU_MODEL": MODEL,
        "YUNSHU_AUTH_DISABLED": "1",
        "HF_HUB_OFFLINE": "1",
        "YUNSHU_ROUND_DRIVER": "0" if a.arm == "off" else "1",
    }
    log = a.out.with_suffix(f".{a.arm}.r{a.rep}.server.log")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    url = f"http://127.0.0.1:{a.port}"
    with log.open("w") as lf:
        srv = subprocess.Popen(
            [sys.executable, "-c", SERVE, str(a.port)],
            env=env, stdout=lf, stderr=subprocess.STDOUT, cwd=root,
        )  # fmt: skip
    try:
        for _ in range(300):
            if srv.poll() is not None:
                print("server exited early", flush=True)
                return 1
            try:
                if (
                    b'"ready":true'
                    in urllib.request.urlopen(url + "/health/ready", timeout=2).read()
                ):
                    break
            except Exception:
                pass
            time.sleep(2)
        else:
            print("server not ready", flush=True)
            return 1
        if ("Round driver:" in log.read_text()) != (a.arm == "routed"):
            print("driver engagement does not match arm", flush=True)
            return 1
        from bench_context_batch import INSTRUCTIONS, make_prompt, stream
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(MODEL)
        corpora = {k: (CORPORA / f"{k}.txt").read_text() for k in INSTRUCTIONS}
        stream(url, "Qwen3.8-27B", "Say hi.", 8, 600)  # warm-up
        for c in CONCURRENCY:
            reqs = workload(c, a.rep)
            prompts = [
                make_prompt(tok, corpora[r["kind"]], r["pp"], INSTRUCTIONS[r["kind"]])
                for r in reqs
            ]
            t0 = time.perf_counter()

            def go(i):
                time.sleep(max(0.0, reqs[i]["at"] - (time.perf_counter() - t0)))
                return stream(url, "Qwen3.8-27B", prompts[i], a.tg, 1800)

            with concurrent.futures.ThreadPoolExecutor(c) as pool:
                rows = list(pool.map(go, range(c)))
            wall = time.perf_counter() - t0
            bad = [r for r in rows if r.get("error") or not r.get("ttft_s")]
            if bad:
                print(f"c={c}: failed requests {bad[:2]}", flush=True)
                return 1
            rec = {
                "arm": a.arm, "rep": a.rep, "c": c, "workload": reqs,
                **summarize(rows, wall),
                "ttft_each": [r["ttft_s"] for r in rows],
            }  # fmt: skip
            with a.out.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            print(
                json.dumps({k: v for k, v in rec.items() if k != "workload"}),
                flush=True,
            )
        print("complete", flush=True)
        return 0
    finally:
        srv.send_signal(signal.SIGKILL)
        srv.wait()


if __name__ == "__main__":
    sys.exit(main())
