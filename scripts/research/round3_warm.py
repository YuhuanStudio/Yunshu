"""Warm long-context cell for one arm (run inside one gpuq job).

Agentic shape: c requests, each a distinct CTX-token prefix (primed first, so
it sits in the prefix cache) plus a TURN-token new turn, long reply (TG tokens).
--prime alone: prefixes primed one at a time (a lone request: the upstream path);
--prime conc: primed together (the round driver's path when it is on).
Records per request TTFT, cached tokens, decode tok/s over the whole reply, and the
server's memory footprint. Starts its own server (kill -9 on exit); nonzero exit on
any failed request or a driver-engagement mismatch.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
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
CORPUS = Path(
    "/Users/yuhuan/Documents/YuhuanStudio/Yunshu/reference/omlx/omlx/admin/bench_corpora/code_python.txt"
)
SERVE = (
    "import sys, uvicorn;"
    "import yunshu_engine.vlm_batch_runner as v;"
    "v.DRIVER_MAX_UNCACHED_TOKENS = int(sys.argv[2]) or v.DRIVER_MAX_UNCACHED_TOKENS;"
    "uvicorn.run('yunshu_gateway.main:app', host='127.0.0.1', port=int(sys.argv[1]))"
)
INSTRUCTION = "\n\nContinue this code with more functions. Output only code."


def spans(ctx: int, turn: int, c: int) -> list[tuple[int, int, int]]:
    """Per request (prefix start, prefix end, turn end) in the token stream: every
    request gets its own slice, so no two share a prefix."""
    out, at = [], 0
    for _ in range(c):
        out.append((at, at + ctx, at + ctx + turn))
        at += ctx + turn
    return out


def summarize(rows: list[dict], wall: float) -> dict:
    ttft = [r["ttft_s"] for r in rows]
    dec = [r["decode_tps"] for r in rows if r.get("decode_tps")]
    return {
        "mean_ttft_s": round(statistics.mean(ttft), 3),
        "max_ttft_s": max(ttft),
        "mean_decode_tps": round(statistics.mean(dec), 1) if dec else None,
        "min_decode_tps": min(dec) if dec else None,
        "completion_tokens": [r["completion_tokens"] for r in rows],
        "cached_tokens": [r["cached_tokens"] for r in rows],
        "wall_s": round(wall, 2),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["off", "routed"])
    ap.add_argument("--prime", choices=["alone", "conc"], default="alone")
    ap.add_argument("--ctx", type=int, required=True)
    ap.add_argument("--turn", type=int, default=2048)
    ap.add_argument("--c", type=int, default=2)
    ap.add_argument("--tg", type=int, default=1500)
    ap.add_argument("--max-uncached", type=int, default=0)
    ap.add_argument("--rep", type=int, default=0)
    ap.add_argument("--port", type=int, default=18993)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    root = HERE.parents[1]
    env = {
        **os.environ,
        "PYTHONPATH": str(root / "python"),
        "YUNSHU_MODEL": MODEL,
        "YUNSHU_AUTH_DISABLED": "1",
        "HF_HUB_OFFLINE": "1",
        "YUNSHU_DEBUG_ROUTES": "1",
        "YUNSHU_ROUND_DRIVER": "0" if a.arm == "off" else "1",
    }
    log = a.out.with_suffix(f".{a.arm}.{a.prime}.{a.ctx}.c{a.c}.r{a.rep}.server.log")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    url = f"http://127.0.0.1:{a.port}"
    with log.open("w") as lf:
        srv = subprocess.Popen(
            [sys.executable, "-c", SERVE, str(a.port), str(a.max_uncached)],
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
        from bench_context_batch import stream
        from process_memory import process_tree_memory
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(MODEL)
        corpus = CORPUS.read_text()
        need = a.c * (a.ctx + a.turn) + 64
        ids, text = [], corpus
        while len(ids) < need:
            ids = tok.encode(text)
            text += corpus
        spn = spans(a.ctx, a.turn, a.c)
        salt = f"WARM-{os.getpid()}-{a.rep}-"
        prefixes = [
            salt + f"{i} " + tok.decode(ids[s:e]) for i, (s, e, _) in enumerate(spn)
        ]
        turns = [
            p + tok.decode(ids[e:t]) + INSTRUCTION
            for p, (_, e, t) in zip(prefixes, spn, strict=True)
        ]

        def mem():
            return round(
                process_tree_memory(srv.pid)["physical_footprint_sum_bytes"] / 2**30, 2
            )

        def apc_metrics():
            out = {}
            try:
                text = (
                    urllib.request.urlopen(url + "/metrics", timeout=5).read().decode()
                )
            except Exception:
                return out
            for line in text.splitlines():
                if "apc_" in line and not line.startswith("#"):
                    name, _, val = line.rpartition(" ")
                    out[name.split("{")[0]] = float(val)
            return out

        def run(prompts, tg):
            t0 = time.perf_counter()
            with concurrent.futures.ThreadPoolExecutor(len(prompts)) as pool:
                rows = list(
                    pool.map(lambda p: stream(url, "Qwen3.8-27B", p, tg, 3000), prompts)
                )
            return rows, time.perf_counter() - t0

        stream(url, "Qwen3.8-27B", "Say hi.", 8, 600)  # warm-up
        t_prime = time.perf_counter()
        if a.prime == "alone":
            for p in prefixes:
                rows, _ = run([p], 8)
                if rows[0].get("error"):
                    print("prime failed", rows[0], flush=True)
                    return 1
        else:
            rows, _ = run(prefixes, 8)
            if any(r.get("error") for r in rows):
                print("prime failed", rows, flush=True)
                return 1
        prime_s = time.perf_counter() - t_prime
        mem_primed = mem()
        apc_primed = apc_metrics()
        rows, wall = run(turns, a.tg)
        bad = [r for r in rows if r.get("error") or not r.get("ttft_s")]
        if bad:
            print("request failed", bad[:2], flush=True)
            return 1
        busy = 0.0
        try:
            for line in (
                urllib.request.urlopen(url + "/metrics", timeout=5)
                .read()
                .decode()
                .splitlines()
            ):
                if "round_driver_busy_seconds" in line and not line.startswith("#"):
                    busy = float(line.split()[-1])
        except Exception:
            pass
        rec = {
            "arm": a.arm, "prime": a.prime, "ctx": a.ctx, "turn": a.turn, "c": a.c,
            "tg": a.tg, "rep": a.rep, "max_uncached": a.max_uncached,
            "apc_primed": apc_primed, "apc_end": apc_metrics(),
            "prime_s": round(prime_s, 1), "footprint_primed_gib": mem_primed,
            "footprint_end_gib": mem(), "driver_busy_s": busy,
            **summarize(rows, wall),
            "ttft_each": [r["ttft_s"] for r in rows],
            "decode_each": [r.get("decode_tps") for r in rows],
        }  # fmt: skip
        try:
            rec["census"] = json.loads(
                urllib.request.urlopen(
                    url + "/debug/memory-census?min_mib=128", timeout=120
                ).read()
            )
        except Exception as exc:
            rec["census"] = {"error": repr(exc)}
        with a.out.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)
        if (busy > 0) != (a.arm == "routed" and a.prime == "conc"):
            print("note: driver busy seconds", busy, flush=True)
        return 0
    finally:
        srv.send_signal(signal.SIGKILL)
        srv.wait()


if __name__ == "__main__":
    sys.exit(main())
