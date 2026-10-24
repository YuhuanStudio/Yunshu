"""Verdict probes for text-only experimental settings (external LM draft, engine loop).

Starts one server (port 18990-18995) with the given env, runs a workload, kills it.

    python scripts/research/flagverdict_text.py spec  --model T --draft D --out o.json
    python scripts/research/flagverdict_text.py conc  --model T --label loop --env YUNSHU_ENGINE_LOOP=1 --out o.json

spec: one server with SPEC_UNVERIFIED=eagle + DRAFT_MODEL; every prompt is sent alternately with
spec_decode false (plain fast path) and true (draft), greedy non-stream, ``reps`` interleaved rounds.
Identity = texts equal; engagement = /v1 spec stats counters moved (checked from the server log).
conc: concurrency 1/2/4/8 streaming greedy requests; per-request TTFT, decode tok/s, aggregate tok/s,
and the greedy texts (for comparison with another run).
"""

import argparse
import concurrent.futures
import http.client
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

PROMPTS = [
    "Write a Python function that merges overlapping intervals, with a docstring and three unit tests.",
    "Explain how a B-tree keeps itself balanced, step by step, with a short worked example.",
    "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "List twelve practical tips for debugging a slow SQL query and explain each in two sentences.",
]


def identical(a: list[str], b: list[str]) -> bool:
    return len(a) == len(b) and all(x == y for x, y in zip(a, b, strict=True))


def first_diff(a: str, b: str) -> int | None:
    for i, (x, y) in enumerate(zip(a, b, strict=False)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def summarize(xs: list[float]) -> dict:
    xs = [x for x in xs if x is not None]
    if not xs:
        return {"n": 0}
    return {
        "n": len(xs),
        "mean": round(statistics.mean(xs), 2),
        "min": round(min(xs), 2),
        "max": round(max(xs), 2),
    }


def req(port, body, stream=False, timeout=600):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    body = {"temperature": 0.0, "enable_thinking": False, **body}
    if stream:
        body.update(stream=True, stream_options={"include_usage": True})
    t0 = time.perf_counter()
    c.request(
        "POST",
        "/v1/chat/completions",
        json.dumps(body),
        {"Content-Type": "application/json"},
    )
    r = c.getresponse()
    if r.status != 200:
        raise RuntimeError(f"HTTP {r.status}: {r.read()[:300]!r}")
    if not stream:
        d = json.loads(r.read())
        c.close()
        return {
            "text": d["choices"][0]["message"]["content"],
            "tokens": d["usage"]["completion_tokens"],
            "wall": time.perf_counter() - t0,
        }
    first = last = None
    text, usage, done, fin = "", None, False, None
    for line in r:
        if not line.startswith(b"data: "):
            continue
        if line[6:].strip() == b"[DONE]":
            done = True
            continue
        ev = json.loads(line[6:])
        usage = ev.get("usage") or usage
        for ch in ev.get("choices") or []:
            fin = ch.get("finish_reason") or fin
            p = (ch.get("delta") or {}).get("content")
            if p:
                now = time.perf_counter()
                first = first or now
                last = now
                text += p
    c.close()
    if not done or usage is None or first is None:
        raise RuntimeError("incomplete stream")
    n = usage["completion_tokens"]
    return {
        "text": text,
        "tokens": n,
        "ttft": first - t0,
        "wall": time.perf_counter() - t0,
        "dec_tps": (n - 1) / (last - first) if last > first and n > 1 else None,
    }


def start_server(port, model, env_extra, logpath):
    env = dict(
        os.environ,
        YUNSHU_MODEL=model,
        YUNSHU_AUTH_DISABLED="1",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        PYTHONPATH="python",
        **env_extra,
    )
    log = open(logpath, "w")  # noqa: SIM115
    p = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "yunshu_gateway.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    for _ in range(300):
        if p.poll() is not None:
            raise RuntimeError("server exited early")
        try:
            c = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
            c.request("GET", "/health/ready")
            if b'"ready":true' in c.getresponse().read():
                return p
        except OSError:
            pass
        time.sleep(2)
    p.kill()
    raise RuntimeError("server not ready")


def stop(p):
    p.send_signal(2)
    try:
        p.wait(30)
    except subprocess.TimeoutExpired:
        p.kill()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["spec", "conc"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--draft")
    ap.add_argument("--label", default="x")
    ap.add_argument("--env", nargs="*", default=[])
    ap.add_argument("--port", type=int, default=18990)
    ap.add_argument("--tokens", type=int, default=192)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", required=True)
    ap.add_argument("--log", required=True)
    a = ap.parse_args()
    env = dict(e.split("=", 1) for e in a.env)
    if a.mode == "spec":
        env.update(YUNSHU_SPEC_UNVERIFIED="eagle", YUNSHU_DRAFT_MODEL=a.draft)
    srv = start_server(a.port, a.model, env, a.log)
    res = {"mode": a.mode, "label": a.label, "env": env, "model": a.model}
    try:
        name = json.loads(_get(a.port, "/v1/models"))["data"][0]["id"]
        base = {"model": name, "max_tokens": a.tokens}
        msgs = [[{"role": "user", "content": p}] for p in PROMPTS]
        req(a.port, {**base, "messages": msgs[0], "max_tokens": 16})  # warm
        if a.mode == "spec":
            plain, spec = (
                {i: [] for i in range(len(msgs))},
                {i: [] for i in range(len(msgs))},
            )
            for _ in range(a.reps):
                for i, m in enumerate(msgs):
                    plain[i].append(req(a.port, {**base, "messages": m}))
                    spec[i].append(
                        req(a.port, {**base, "messages": m, "spec_decode": True})
                    )
            res["plain_deterministic"] = all(
                identical([r["text"] for r in plain[i]], [plain[i][0]["text"]] * a.reps)
                for i in plain
            )
            res["spec_equal_plain_all_reps"] = all(
                plain[i][k]["text"] == spec[i][k]["text"]
                for i in plain
                for k in range(a.reps)
            )
            res["first_diff_char"] = [
                first_diff(plain[i][0]["text"], spec[i][0]["text"]) for i in plain
            ]
            res["spec_texts"] = [spec[i][0]["text"] for i in spec]
            res["tokens"] = [plain[i][0]["tokens"] for i in plain]
            res["plain_tps"] = [
                [round(r["tokens"] / r["wall"], 2) for r in plain[i]] for i in plain
            ]
            res["spec_tps"] = [
                [round(r["tokens"] / r["wall"], 2) for r in spec[i]] for i in spec
            ]
            res["texts"] = [plain[i][0]["text"] for i in plain]
            res["stats"] = None
        else:
            res["texts_by_n"] = {}
            res["rows"] = {}
            for n in (1, 2, 4, 8):
                ms = [msgs[i % len(msgs)] for i in range(n)]
                rounds = []
                for _ in range(2):
                    t0 = time.perf_counter()
                    with concurrent.futures.ThreadPoolExecutor(n) as ex:
                        rs = list(
                            ex.map(
                                lambda m: req(
                                    a.port, {**base, "messages": m}, stream=True
                                ),
                                ms,
                            )
                        )
                    wall = time.perf_counter() - t0
                    rounds.append(
                        {
                            "agg_tps": sum(r["tokens"] for r in rs) / wall,
                            "ttft": [r["ttft"] for r in rs],
                            "dec": [r["dec_tps"] for r in rs],
                            "texts": [r["text"] for r in rs],
                        }
                    )
                res["rows"][n] = {
                    "agg_tps": summarize([r["agg_tps"] for r in rounds]),
                    "ttft_s": summarize([t for r in rounds for t in r["ttft"]]),
                    "per_req_tps": summarize([t for r in rounds for t in r["dec"]]),
                    "stable_texts": all(
                        r["texts"] == rounds[0]["texts"] for r in rounds
                    ),
                }
                res["texts_by_n"][n] = rounds[0]["texts"]
    finally:
        stop(srv)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(
        json.dumps(
            {
                k: v
                for k, v in res.items()
                if k not in ("texts", "texts_by_n", "spec_texts")
            },
            indent=1,
        )[:3000]
    )


def _get(port, path):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    c.request("GET", path)
    return c.getresponse().read().decode()


if __name__ == "__main__":
    main()
