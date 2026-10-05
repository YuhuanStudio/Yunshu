"""One arm x one cell of the round-driver routing A/B (run inside one gpuq job).

arm off      YUNSHU_ROUND_DRIVER=0 (upstream shared batch + single-request lane)
arm always   driver for every text request (MIN_CONCURRENCY=1)
arm routed   driver only at concurrency >= 2 (MIN_CONCURRENCY=2)
cell 1k      single 1K, then b2/b4/b8 at pp1024, 128 new tokens
cell 32k     b2 and b4 cold at pp32768, 128 new tokens

Starts its own server (kill -9 on exit), exits nonzero on any failure.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

# the oMLX checkout the worktree has no copy of
CORPORA = "/Users/yuhuan/Documents/YuhuanStudio/Yunshu/reference/omlx/omlx/admin/bench_corpora"
MODEL = "/Volumes/P5Plus/models/Jundot/Qwen3.8-27B-oQ4e-mtp"


SERVE = (
    "import sys, uvicorn;"
    "import yunshu_engine.vlm_batch_runner as v;"
    "v.DRIVER_MAX_UNCACHED_TOKENS = int(sys.argv[2]) or v.DRIVER_MAX_UNCACHED_TOKENS;"
    "uvicorn.run('yunshu_gateway.main:app', host='127.0.0.1', port=int(sys.argv[1]))"
)


def arm_env(arm: str) -> dict[str, str]:
    if arm == "off":
        return {"YUNSHU_ROUND_DRIVER": "0"}
    if arm == "routed":
        return {"YUNSHU_ROUND_DRIVER": "1"}
    raise ValueError(arm)


def bench_args(cell: str) -> list[str]:
    if cell == "1k":
        return ["--lengths", "1024", "--batches", "2", "4", "8", "--batch-pp", "1024"]
    if cell == "8k":
        return ["--lengths", "--batches", "2", "4", "--batch-pp", "8192"]
    if cell == "32k":
        return ["--lengths", "--batches", "2", "4", "--batch-pp", "32768"]
    raise ValueError(cell)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["off", "routed"])
    ap.add_argument("--cell", required=True, choices=["1k", "8k", "32k"])
    ap.add_argument(
        "--max-uncached", type=int, help="override DRIVER_MAX_UNCACHED_TOKENS"
    )
    ap.add_argument("--rep", type=int, default=0)
    ap.add_argument("--port", type=int, default=18991)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    root = Path(__file__).resolve().parents[2]
    log = a.out.with_suffix(f".{a.arm}.{a.cell}.r{a.rep}.server.log")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    env = {
        **os.environ,
        "PYTHONPATH": str(root / "python"),
        "YUNSHU_MODEL": MODEL,
        "YUNSHU_AUTH_DISABLED": "1",
        "HF_HUB_OFFLINE": "1",
        **arm_env(a.arm),
    }
    url = f"http://127.0.0.1:{a.port}"
    with log.open("w") as lf:
        srv = subprocess.Popen(
            [sys.executable, "-c", SERVE, str(a.port), str(a.max_uncached or 0)],
            env=env, stdout=lf, stderr=subprocess.STDOUT, cwd=root,
        )  # fmt: skip
    try:
        for _ in range(300):
            if srv.poll() is not None:
                print("server exited early", flush=True)
                return 1
            try:
                body = urllib.request.urlopen(url + "/health/ready", timeout=2).read()
                if b'"ready":true' in body:
                    break
            except Exception:
                pass
            time.sleep(2)
        else:
            print("server not ready", flush=True)
            return 1
        text = log.read_text()
        engaged = "Round driver:" in text
        if engaged != (a.arm != "off"):
            print(f"engaged={engaged} does not match arm {a.arm}", flush=True)
            return 1
        before = len(a.out.read_text().splitlines()) if a.out.exists() else 0
        rc = subprocess.call(
            [sys.executable, str(root / "scripts/research/bench_context_batch.py"),
             "--url", url, "--model", "Qwen3.8-27B", "--tokenizer", MODEL,
             "--pid", str(srv.pid), "--tg", "256", "--corpora-dir", CORPORA, "--note", f"{a.arm}/{a.cell}/r{a.rep}",
             "--output", str(a.out), *bench_args(a.cell)],
            env=env, cwd=root,
        )  # fmt: skip
        if rc:
            print(f"bench rc={rc}", flush=True)
            return 1
        rows = [json.loads(x) for x in a.out.read_text().splitlines()[before:]]
        batches = [r for r in rows if r.get("kind") == "batch"]
        want = 3 if a.cell == "1k" else 2
        if (
            rc
            or len(batches) != want
            or any(b["ok"] != b["batch_size"] for b in batches)
        ):
            print(f"incomplete: rc={rc} batches={len(batches)}", flush=True)
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
        except Exception as exc:
            print("metrics unavailable", exc, flush=True)
        print("driver busy seconds:", busy, flush=True)
        # the 32k cell: prompts above DRIVER_MAX_UNCACHED_TOKENS keep the upstream
        # path by design, so the driver stays idle in both arms
        expect_driver = a.arm != "off" and (a.cell == "1k" or bool(a.max_uncached))
        if (busy > 0) != expect_driver:
            print("driver usage does not match arm", flush=True)
            return 1
        print("complete", flush=True)
        return 0
    finally:
        srv.send_signal(signal.SIGKILL)
        srv.wait()


if __name__ == "__main__":
    sys.exit(main())
