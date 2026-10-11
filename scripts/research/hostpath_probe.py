"""Host-path probe: TTFT decomposition, concurrency (GIL) and thread QoS on a real server.

Run only through gpuq. One job = one server with one arm; the launcher patches the server
process (never the repo) with:

- ``qos``: ``none`` | ``mlx`` (MLX executor thread UI) | ``all`` (every Python thread + main UI)
- a monitor thread: event-loop lag percentiles, GC pause totals and per-thread CPU time,
  appended to ``<out>.mon.jsonl`` once a second

The client measures first header byte, first content chunk and the server's own
``x_yunshu.latency`` breakdown, for short and long prompts and for 1 vs N concurrent streams.

    python scripts/research/hostpath_probe.py --arm base --out runs/hp-base.jsonl
"""

from __future__ import annotations

import argparse
import contextlib
import http.client
import json
import os
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

ARMS = ("base", "mlx", "all", "sw", "allsw")

LAUNCHER_PATCH = r"""
import asyncio, ctypes, gc, json, os, sys, threading, time
CFG = json.loads(os.environ["HP_CFG"])
_lib = ctypes.CDLL("libSystem.B.dylib")
UI = 0x21
def set_qos(cls=UI):
    return _lib.pthread_set_qos_class_self_np(cls, 0)
if CFG.get("switch"):
    sys.setswitchinterval(CFG["switch"])
if CFG["qos"] in ("mlx", "all"):
    from yunshu_engine import mlx_executor as _me
    _orig_init = _me._init_mlx_thread
    def _init():
        set_qos()
        _orig_init()
    _me._init_mlx_thread = _init
if CFG["qos"] == "all":
    set_qos()
    _orig_run = threading.Thread.run
    def _run(self):
        set_qos()
        return _orig_run(self)
    threading.Thread.run = _run

_gc = {"n": 0, "pause": 0.0, "max": 0.0, "t0": 0.0}
def _gccb(phase, info):
    if phase == "start":
        _gc["t0"] = time.perf_counter()
    else:
        d = time.perf_counter() - _gc["t0"]
        _gc["n"] += 1; _gc["pause"] += d; _gc["max"] = max(_gc["max"], d)
gc.callbacks.append(_gccb)

_lag = []
async def _lagloop():
    _l = asyncio.get_running_loop()
    with open(CFG["mon"], "a") as f:
        f.write(json.dumps(dict(loop=type(_l).__module__ + "." + type(_l).__name__)) + "\n")
    while True:
        t = time.perf_counter()
        await asyncio.sleep(0.005)
        _lag.append((time.perf_counter() - t - 0.005) * 1000)

def _monitor():
    import psutil
    p = psutil.Process()
    prev = {t.id: (t.user_time + t.system_time) for t in p.threads()}
    out = open(CFG["mon"], "a")
    out.write(json.dumps(dict(start=True, gc_threshold=gc.get_threshold(), gc_frozen=gc.get_freeze_count())) + "\n")
    while True:
        time.sleep(1.0)
        lag, _lag[:] = _lag[:], []
        cur = {}
        names = {t.ident: t.name for t in threading.enumerate()}
        for t in p.threads():
            cur[t.id] = t.user_time + t.system_time
        busy = sorted(((cur[i] - prev.get(i, 0.0), i) for i in cur), reverse=True)[:6]
        prev = cur
        lag.sort()
        row = dict(t=time.time(), gc_n=_gc["n"], gc_pause_ms=_gc["pause"] * 1000, gc_max_ms=_gc["max"] * 1000,
                   lag_n=len(lag), lag_p50=lag[len(lag)//2] if lag else None,
                   lag_p99=lag[int(len(lag)*0.99)] if lag else None, lag_max=lag[-1] if lag else None,
                   cpu_top=[(round(c, 3), names.get(i, i)) for c, i in busy if c > 0.001])
        out.write(json.dumps(row) + "\n"); out.flush()

from fastapi import FastAPI
_orig_call = FastAPI.__call__
_started = []
async def _call(self, scope, receive, send):
    if not _started:
        _started.append(1)
        asyncio.get_running_loop().create_task(_lagloop())
        threading.Thread(target=_monitor, name="hp-monitor", daemon=True).start()
    return await _orig_call(self, scope, receive, send)
FastAPI.__call__ = _call

from yunshu_cli import main
main()
"""


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else None


def stream(port, body, t0_label=None):
    """One streaming request; returns client timestamps (ms from send) and server latency."""
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=900)
    payload = json.dumps(
        dict(body, stream=True, stream_options={"include_usage": True})
    ).encode()
    t0 = time.perf_counter()
    conn.request(
        "POST",
        "/v1/chat/completions",
        payload,
        {"Content-Type": "application/json", "Authorization": "Bearer k"},
    )
    resp = conn.getresponse()
    t_head = time.perf_counter()
    first = None
    stamps = []
    done = False
    xy = usage = None
    for line in resp:
        if not line.startswith(b"data:"):
            continue
        p = line[5:].strip()
        if p == b"[DONE]":
            done = True
            break
        d = json.loads(p)
        if d.get("error"):
            raise RuntimeError(f"SSE error {d['error']}")
        usage = d.get("usage") or usage
        xy = d.get("x_yunshu") or xy
        for ch in d.get("choices") or []:
            dl = ch.get("delta") or {}
            if dl.get("content") or dl.get("reasoning_content"):
                now = time.perf_counter()
                stamps.append(now)
                if first is None:
                    first = now
    t1 = time.perf_counter()
    conn.close()
    if not done or usage is None or first is None:
        raise RuntimeError("incomplete stream")  # fail closed
    ct = usage["completion_tokens"]
    lat = (xy or {}).get("latency") or {}
    return dict(
        head_ms=(t_head - t0) * 1000,
        ttft_ms=(first - t0) * 1000,
        total_s=t1 - t0,
        ct=ct,
        pt=usage["prompt_tokens"],
        dec_tps=(ct - 1) / (t1 - first) if ct > 1 and t1 > first else None,
        chunks=len(stamps),
        durations_ms=lat.get("durations_ms"),
        milestones_ms=lat.get("milestones_ms"),
        server_ttft_ms=(xy or {}).get("ttft_ms"),
    )


def body_for(model, text, mt, seed=1234):
    return dict(
        model=model,
        messages=[{"role": "user", "content": text}],
        max_tokens=mt,
        temperature=0,
        seed=seed,
        chat_template_kwargs={"enable_thinking": False},
    )


def salted(text, salt):
    return f"[run {salt}]\n" + text


def summarize(rows, key):
    xs = [r[key] for r in rows if r.get(key) is not None]
    if not xs:
        return None
    return dict(
        n=len(xs), med=round(statistics.median(xs), 2), p90=round(pct(xs, 0.9), 2)
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", choices=ARMS, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model")
    p.add_argument(
        "--load", type=int, default=0, help="background CPU burner processes"
    )
    p.add_argument("--short-n", type=int, default=12)
    p.add_argument("--long-n", type=int, default=3)
    p.add_argument("--conc", type=int, nargs="*", default=[1, 4])
    p.add_argument("--tokens", type=int, default=256)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    if a.dry_run:
        print(json.dumps(vars(a), default=str))
        return
    import tfbench as t

    if a.model:
        t.M = a.model
    a.out.parent.mkdir(parents=True, exist_ok=True)
    if a.out.exists():
        raise FileExistsError(a.out)
    t.OUT = a.out.parent / a.out.stem
    t.YUNSHU_SRC = str(Path(__file__).resolve().parents[2] / "python")
    mon = a.out.with_suffix(".mon.jsonl")
    launcher = a.out.with_suffix(".launcher.py")
    launcher.write_text(f"#!{sys.executable}\n" + LAUNCHER_PATCH)
    launcher.chmod(0o700)
    t.YUNSHU_BIN = str(launcher)
    qos = {"base": "none", "sw": "none", "allsw": "all"}.get(a.arm, a.arm)
    cfg = json.dumps(
        dict(qos=qos, switch=0.0005 if a.arm in ("sw", "allsw") else None, mon=str(mon))
    )
    burners = []
    server = None
    try:
        for _ in range(a.load):
            burners.append(subprocess.Popen([sys.executable, "-c", "while True: pass"]))
        server = t.Srv(
            "yunshu",
            {"YUNSHU_VLM_DRAFT": "mtp", "YUNSHU_VLM_APC_DISK": "0", "HP_CFG": cfg},
            a.out.stem,
        )
        short_text = "Explain in two sentences why the sky is blue."
        prose1k = t.load_prompt("prose-1024")
        prose8k = t.load_prompt("prose-8192")
        model = server.model
        rows = []

        def emit(**kw):
            row = dict(arm=a.arm, load=a.load, contended=t.was_contended(), **kw)
            rows.append(row)
            with a.out.open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(json.dumps(row)[:400], flush=True)

        # warm every path once, unmeasured
        stream(server.port, body_for(model, short_text, 16))
        for i in range(a.short_n):
            r = stream(server.port, body_for(model, salted(short_text, f"s{i}"), 16))
            emit(kind="short", i=i, **r)
        for i in range(a.long_n):  # cold 8K: unique salt defeats the prefix cache
            r = stream(server.port, body_for(model, salted(prose8k, f"c{i}"), 16))
            emit(kind="long8k_cold", i=i, **r)
        for i in range(a.long_n):  # warm 8K: same prompt as the last cold one
            r = stream(
                server.port,
                body_for(model, salted(prose8k, f"c{a.long_n - 1}"), 16),
            )
            emit(kind="long8k_warm", i=i, **r)
        for n in a.conc:
            for rep in range(2):
                results = [None] * n
                errs = []

                def work(k, n=n, rep=rep):
                    try:
                        results[k] = stream(
                            server.port,
                            body_for(
                                model,
                                salted(prose1k, f"n{n}r{rep}k{k}"),
                                a.tokens,
                            ),
                        )
                    except Exception as e:  # noqa: BLE001
                        errs.append(repr(e))

                ths = [threading.Thread(target=work, args=(k,)) for k in range(n)]
                t0 = time.perf_counter()
                for th in ths:
                    th.start()
                for th in ths:
                    th.join()
                wall = time.perf_counter() - t0
                if errs:
                    raise RuntimeError(f"concurrent stream failed: {errs}")
                emit(
                    kind="conc",
                    n=n,
                    rep=rep,
                    wall_s=wall,
                    agg_tps=sum(r["ct"] for r in results) / wall,
                    per_stream_tps=[r["dec_tps"] for r in results],
                    ttft_ms=[r["ttft_ms"] for r in results],
                )
        summary = dict(
            arm=a.arm,
            load=a.load,
            short_ttft=summarize([r for r in rows if r["kind"] == "short"], "ttft_ms"),
            short_head=summarize([r for r in rows if r["kind"] == "short"], "head_ms"),
            cold_ttft=summarize(
                [r for r in rows if r["kind"] == "long8k_cold"], "ttft_ms"
            ),
            warm_ttft=summarize(
                [r for r in rows if r["kind"] == "long8k_warm"], "ttft_ms"
            ),
        )
        with a.out.open("a") as f:
            f.write(json.dumps(dict(kind="summary", **summary)) + "\n")
        print(json.dumps(summary), flush=True)
    finally:
        for b in burners:
            b.kill()
        if server is not None:
            server.kill()
        with contextlib.suppress(OSError):
            launcher.unlink()


if __name__ == "__main__":
    main()
