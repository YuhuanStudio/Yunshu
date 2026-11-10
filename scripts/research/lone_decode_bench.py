#!/usr/bin/env python3
"""Lone-row greedy decode tok/s of a served model on a repo tree (round driver generic layer A/B).

argv: --repo --model --label --out [--tokens N --reps R]. Starts a server on 18990-18999 (killed on
exit), one warmup, then R sequential /v1/chat/completions of N max tokens at temperature 0; records
text, completion tokens and decode tok/s. Exits nonzero when a request fails or yields too few tokens.
"""

import argparse
import json
import os
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path

PY = "/Users/yuhuan/Documents/YuhuanStudio/Yunshu/.venv/bin/python"
PROMPT = "Write a long, detailed story about a lighthouse keeper and the sea. Keep going without stopping."


def summarize(runs: list[dict]) -> dict:
    rates = [r["ct"] / r["s"] for r in runs]
    return {
        "tok_s_med": statistics.median(rates),
        "tok_s_all": rates,
        "texts_equal": len({r["text"] for r in runs}) == 1,
    }


def free_port() -> int:
    for port in range(18990, 19000):
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise SystemExit("no free port in 18990-18999")


def main() -> int:
    import httpx

    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokens", type=int, default=512)
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()
    repo, out = Path(a.repo).resolve(), Path(a.out).resolve()
    work = out.parent / a.label
    (work / "home").mkdir(parents=True, exist_ok=True)
    port = free_port()
    env = dict(
        os.environ,
        PYTHONPATH=f"{repo}/python",
        YUNSHU_MODEL=a.model,
        YUNSHU_AUTH_DISABLED="true",
        HOME=str(work / "home"),
        HF_HUB_OFFLINE="1",
        PYTHONUNBUFFERED="1",
        YUNSHU_CONSOLE="0",
    )
    with open(work / "server.log", "w") as log:
        proc = subprocess.Popen(
            [
                PY,
                "-m",
                "uvicorn",
                "--app-dir",
                f"{repo}/python",
                "yunshu_gateway.main:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
            ],
            cwd=str(repo),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            base = f"http://127.0.0.1:{port}"
            t0 = time.time()
            while True:
                if proc.poll() is not None or time.time() - t0 > 900:
                    raise SystemExit("server failed to come up")
                try:
                    if httpx.get(base + "/health", timeout=5).json().get("status") in (
                        "ok",
                        "healthy",
                        "ready",
                    ):
                        break
                except Exception:
                    pass
                time.sleep(1)

            def one() -> dict:
                s = time.perf_counter()
                r = httpx.post(
                    base + "/v1/chat/completions",
                    timeout=600,
                    json={
                        "model": os.path.basename(a.model),
                        "temperature": 0,
                        "max_tokens": a.tokens,
                        "messages": [{"role": "user", "content": PROMPT}],
                    },
                )
                dt = time.perf_counter() - s
                r.raise_for_status()
                b = r.json()
                return {
                    "text": b["choices"][0]["message"]["content"],
                    "ct": b["usage"]["completion_tokens"],
                    "s": dt,
                }

            one()
            runs = [one() for _ in range(a.reps)]
            if any(r["ct"] < a.tokens // 2 for r in runs):
                raise SystemExit(f"too few tokens: {[r['ct'] for r in runs]}")
            res = {
                "label": a.label,
                "repo": str(repo),
                "tokens": runs[0]["ct"],
                **summarize(runs),
                "text0": runs[0]["text"],
            }
            out.write_text(json.dumps(res, ensure_ascii=False, indent=2))
            print(
                json.dumps({k: v for k, v in res.items() if k != "text0"}), flush=True
            )
        finally:
            proc.kill()
            proc.wait()
    return 0


if __name__ == "__main__":
    sys.exit(main())
