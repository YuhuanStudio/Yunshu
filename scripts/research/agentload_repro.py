"""Agent-load reproducer: sampled long reasoning, a concurrent request, client timeout + identical
non-streaming retry, and fairness / recovery between concurrent decoders. Real model, real server.

Scenarios (one server per run, isolated HOME, port 18990-18999):
  retry  R1 long sampled stream; R2 concurrent at --r2-at; client gives up on R1 at --cut-at and
         re-sends the identical request with stream=false 2 s later (what Claude Code does).
  fair   A long stream; B short stream joins at --b-at; C joins at --c-at. Per-request tok/s per
         window shows who starves and whether A recovers when B/C end.

Records a 1 Hz poll of /v1/yunshu/status (what the console shows) and every SSE chunk time, writes
timeline.json + timeline.md (+ a PNG when matplotlib exists) into --out.  Exits nonzero on any
failure (server exit, a request error, no tokens).

    python scripts/research/agentload_repro.py MODEL --scenario retry --grace 0 --out DIR
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import subprocess
import sys
import time

TOKEN = "agentload-secret"
THINK_PROMPT = (
    "Design, in great depth, a complete migration plan for moving a 400-table PostgreSQL "
    "monolith to a sharded architecture. Reason step by step about every risk, then write "
    "at least 30 numbered sections, each with concrete examples and trade-offs."
)
SHORT_PROMPT = (
    "Explain, with a worked example, how a B-tree page split works. Be thorough."
)
THIRD_PROMPT = (
    "Write a long essay on the history of the Rust borrow checker, with code samples."
)


def rates(
    chunks: list[tuple[float, int]], t0: float, t1: float, step: float
) -> list[dict]:
    """Per-window tok/s from (time, cumulative_tokens) samples."""
    out = []
    t = t0
    while t < t1:
        lo = [n for (ts, n) in chunks if ts <= t]
        hi = [n for (ts, n) in chunks if ts <= t + step]
        a = lo[-1] if lo else 0
        b = hi[-1] if hi else 0
        out.append(
            {"t": round(t - t0, 1), "tps": round((b - a) / step, 1) if b >= a else 0.0}
        )
        t += step
    return out


def summarize(chunks: list[tuple[float, int]], t_start: float) -> dict:
    if not chunks:
        return {"tokens": 0}
    first, last = chunks[0], chunks[-1]
    span = last[0] - first[0]
    return {
        "tokens": last[1],
        "ttft_s": round(first[0] - t_start, 2),
        "end_s": round(last[0] - t_start, 2),
        "mean_tps": round((last[1] - 1) / span, 1) if span > 0 else None,
    }


def wait_ready(url, proc, secs=600):
    import urllib.request

    for _ in range(secs):
        if proc.poll() is not None:
            raise RuntimeError(f"server exited rc={proc.returncode}")
        try:
            urllib.request.urlopen(url + "/health/ready", timeout=2)
            return
        except Exception:  # noqa: BLE001
            time.sleep(1)
    raise RuntimeError("server not ready")


async def stream_request(client, url, body, name, rec, t_zero, cut_after=None):
    """Consume one SSE stream; chunks = (monotonic time, cumulative chunk count)."""
    chunks = rec.setdefault(name, {"chunks": [], "text": "", "start": time.monotonic()})
    n = 0
    headers = {"Authorization": f"Bearer {TOKEN}"}
    t_cut = time.monotonic() + cut_after if cut_after else None
    async with client.stream(
        "POST",
        url + "/v1/chat/completions",
        json={**body, "stream": True},
        headers=headers,
    ) as r:
        if r.status_code != 200:
            raise RuntimeError(
                f"{name}: HTTP {r.status_code} {(await r.aread())[:200]!r}"
            )
        async for line in r.aiter_lines():
            if t_cut and time.monotonic() >= t_cut:
                chunks["cut"] = time.monotonic()
                return  # closes the connection: the client's timeout
            if not line.startswith("data: ") or line.endswith("[DONE]"):
                continue
            try:
                ev = json.loads(line[6:])
            except ValueError:
                continue
            ch = (ev.get("choices") or [{}])[0].get("delta") or {}
            piece = (ch.get("content") or "") + (
                ch.get("reasoning_content") or ch.get("reasoning") or ""
            )
            if piece:
                n += 1
                chunks["text"] += piece
                chunks["chunks"].append((time.monotonic(), n))
            if ev.get("usage"):
                chunks["usage"] = ev["usage"]
    chunks["end"] = time.monotonic()


async def nonstream_request(client, url, body, name, rec):
    r0 = rec.setdefault(name, {"start": time.monotonic()})
    r = await client.post(
        url + "/v1/chat/completions",
        json={**body, "stream": False},
        headers={"Authorization": f"Bearer {TOKEN}"},
    )
    r0["end"] = time.monotonic()
    if r.status_code != 200:
        raise RuntimeError(f"{name}: HTTP {r.status_code} {r.text[:200]}")
    d = r.json()
    msg = d["choices"][0]["message"]
    r0["usage"] = d.get("usage")
    r0["text"] = (msg.get("reasoning_content") or msg.get("reasoning") or "") + (
        msg.get("content") or ""
    )
    r0["finish"] = d["choices"][0].get("finish_reason")


async def poll_status(client, url, rec, stop):
    rows = rec.setdefault("status", [])
    while not stop.is_set():
        t = time.monotonic()
        try:
            r = await client.get(
                url + "/v1/yunshu/status",
                headers={"Authorization": f"Bearer {TOKEN}"},
                timeout=5,
            )
            d = r.json()
            items = [
                {
                    k: it.get(k)
                    for k in (
                        "request_id",
                        "phase",
                        "elapsed_s",
                        "completion_tokens",
                        "tokens_per_second",
                    )
                }
                | {"first_token_ms": (it.get("offsets_ms") or {}).get("first_token")}
                for it in d["requests"]["items"]
            ]
            rows.append(
                {
                    "t": t,
                    "active": d["requests"]["active"],
                    "items": items,
                    "live_decode_tps": d["throughput"]["live_decode_tps"],
                    "gpu": d.get("gpu"),
                    "orphans": d.get("orphans"),
                }
            )
        except Exception as exc:  # noqa: BLE001
            rows.append({"t": t, "error": repr(exc)[:100]})
        await asyncio.sleep(1.0)


def gpu_busy(rows):
    vals = [
        (v.get("slices") or {}).get("busy_seconds")
        for r in rows
        for v in ((r.get("gpu") or {}).values())
        if isinstance(v, dict)
    ]
    vals = [v for v in vals if v is not None]
    return (vals[0], vals[-1]) if vals else (None, None)


async def scenario(args, url):
    import httpx

    rec: dict = {}
    stop = asyncio.Event()
    sampling = {"temperature": 0.7, "top_p": 0.95, "top_k": 20}
    limits = httpx.Limits(max_connections=20)
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(1800.0), limits=limits
    ) as client:
        poller = asyncio.create_task(poll_status(client, url, rec, stop))
        await asyncio.sleep(2)
        t_zero = time.monotonic()
        rec["t_zero"] = t_zero
        base = {"model": "m", **sampling}
        if args.scenario == "retry":
            b1 = {
                **base,
                "messages": [{"role": "user", "content": THINK_PROMPT}],
                "max_tokens": args.max_tokens,
            }
            b2 = {
                **base,
                "messages": [{"role": "user", "content": SHORT_PROMPT}],
                "max_tokens": 1200,
            }

            async def r2():
                await asyncio.sleep(args.r2_at)
                await stream_request(client, url, b2, "R2", rec, t_zero)

            async def r1_then_retry():
                await stream_request(
                    client, url, b1, "R1", rec, t_zero, cut_after=args.cut_at
                )
                await asyncio.sleep(2.0)
                await nonstream_request(client, url, b1, "R1_retry", rec)

            await asyncio.gather(r1_then_retry(), r2())
        else:
            ba = {
                **base,
                "messages": [{"role": "user", "content": THINK_PROMPT}],
                "max_tokens": args.max_tokens,
            }
            bb = {
                **base,
                "messages": [{"role": "user", "content": SHORT_PROMPT}],
                "max_tokens": 700,
            }
            bc = {
                **base,
                "messages": [{"role": "user", "content": THIRD_PROMPT}],
                "max_tokens": 900,
            }

            async def later(delay, body, name):
                await asyncio.sleep(delay)
                await stream_request(client, url, body, name, rec, t_zero)

            await asyncio.gather(
                later(0, ba, "A"), later(args.b_at, bb, "B"), later(args.c_at, bc, "C")
            )
        await asyncio.sleep(3)
        stop.set()
        await poller
    return rec


def report(rec, args) -> dict:
    t0 = rec["t_zero"]
    out: dict = {"scenario": args.scenario, "grace": args.grace, "requests": {}}
    for name, r in rec.items():
        if name in ("status", "t_zero") or not isinstance(r, dict):
            continue
        ch = r.get("chunks")
        if ch is not None:
            s = summarize(ch, r["start"])
            s["start_s"] = round(r["start"] - t0, 1)
            s["cut_s"] = round(r["cut"] - t0, 1) if r.get("cut") else None
            s["windows_5s"] = rates(ch, t0, ch[-1][0] + 5 if ch else t0, 5.0)
            out["requests"][name] = s
        else:
            u = r.get("usage") or {}
            out["requests"][name] = {
                "start_s": round(r["start"] - t0, 1),
                "end_s": round(r["end"] - t0, 1),
                "duration_s": round(r["end"] - r["start"], 1),
                "completion_tokens": u.get("completion_tokens"),
                "finish": r.get("finish"),
            }
    if "R1" in rec and "R1_retry" in rec:
        a, b = rec["R1"]["text"], rec["R1_retry"].get("text", "")
        out["retry_continues_same_generation"] = bool(a) and b.startswith(
            a[: max(1, len(a) - 8)]
        )
    lo, hi = gpu_busy(rec["status"])
    out["gpu_busy_s"] = round(hi - lo, 1) if lo is not None else None
    out["status_timeline"] = [
        {
            "t": round(s["t"] - t0, 1),
            "active": s.get("active"),
            "live_decode_tps": s.get("live_decode_tps"),
            "orphans": s.get("orphans"),
            "items": s.get("items"),
        }
        for s in rec["status"]
    ]
    return out


def markdown(res) -> str:
    L = [f"# agentload {res['scenario']} (orphan grace {res['grace']} s)", ""]
    L.append("| request | start s | TTFT s | end s | tokens | mean tok/s | cut s |")
    L.append("|---|---|---|---|---|---|---|")
    for n, r in res["requests"].items():
        L.append(
            f"| {n} | {r.get('start_s')} | {r.get('ttft_s', '')} | {r.get('end_s', '')} | "
            f"{r.get('tokens', r.get('completion_tokens'))} | {r.get('mean_tps', '')} | {r.get('cut_s', '')} |"
        )
    L += ["", f"GPU busy seconds over the run: {res.get('gpu_busy_s')}"]
    if "retry_continues_same_generation" in res:
        L.append(
            f"Retry text continues the first generation: {res['retry_continues_same_generation']}"
        )
    L += ["", "Per 5 s window tok/s:", ""]
    names = [n for n, r in res["requests"].items() if "windows_5s" in r]
    L.append("| t | " + " | ".join(names) + " |")
    L.append("|---|" + "---|" * len(names))
    width = max((len(res["requests"][n]["windows_5s"]) for n in names), default=0)
    for i in range(width):
        row = [str(i * 5)]
        for n in names:
            w = res["requests"][n]["windows_5s"]
            row.append(str(w[i]["tps"]) if i < len(w) else "")
        L.append("| " + " | ".join(row) + " |")
    L += [
        "",
        "Console view (status at 1 Hz, every 5th sample): active / live tok/s / items",
        "",
    ]
    for s in res["status_timeline"][::5]:
        items = "; ".join(
            f"{(it['request_id'] or '')[-6:]}:{it['phase']}:{it.get('completion_tokens')}t:{it.get('tokens_per_second')}tps:ft={it.get('first_token_ms')}"
            for it in (s.get("items") or [])
        )
        L.append(
            f"- t={s['t']} active={s['active']} live={s['live_decode_tps']} orphans={len((s['orphans'] or {}).get('orphans', []))} | {items}"
        )
    return "\n".join(L)


def start_console_shots(a, tree, url, env) -> list:
    """The resident-style console (isolated HOME) plus a headless browser capturing it."""
    import urllib.request

    if not 18990 <= a.console_port <= 18999:
        raise RuntimeError("console port outside 18990-18999")
    cenv = dict(env, HOME=os.path.join(a.home, "console-home"))
    os.makedirs(cenv["HOME"], exist_ok=True)
    cons = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "yunshu_console",
            "--engine",
            url,
            "--engine-token",
            TOKEN,
            "--port",
            str(a.console_port),
        ],
        cwd=tree,
        env=cenv,
        stdout=open(os.path.join(a.out, "console.log"), "w"),  # noqa: SIM115
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    side = [cons]
    for _ in range(60):
        try:
            urllib.request.urlopen(
                f"http://127.0.0.1:{a.console_port}/console/", timeout=2
            )
            break
        except Exception:  # noqa: BLE001
            time.sleep(1)
    else:
        raise RuntimeError("console did not start")
    shots = subprocess.Popen(
        [
            "node",
            "scripts/agentload-shots.mjs",
            "--base",
            f"http://127.0.0.1:{a.console_port}",
            "--out",
            a.shots_dir,
            "--seconds",
            "1500",
        ],
        cwd=os.path.join(tree, "frontend"),
        env=dict(os.environ),
        stdout=open(os.path.join(a.out, "shots.log"), "w"),  # noqa: SIM115
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    side.append(shots)
    return side


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--scenario", choices=("retry", "fair"), default="retry")
    ap.add_argument("--port", type=int, default=18991)
    ap.add_argument("--home", default="/Volumes/P5Plus/yunshu-build/agentload-home")
    ap.add_argument("--out", required=True)
    ap.add_argument("--grace", type=float, default=120.0)
    ap.add_argument("--draft", default=os.environ.get("YUNSHU_VLM_DRAFT", ""))
    ap.add_argument("--max-tokens", type=int, default=7000)
    ap.add_argument("--r2-at", type=float, default=25.0)
    ap.add_argument("--cut-at", type=float, default=60.0)
    ap.add_argument("--b-at", type=float, default=20.0)
    ap.add_argument("--c-at", type=float, default=40.0)
    ap.add_argument(
        "--shots-dir",
        default="",
        help="start the console on --console-port and screenshot it every 5 s",
    )
    ap.add_argument("--console-port", type=int, default=18997)
    ap.add_argument("--extra-env", default="", help="K=V,K=V extra server env")
    a = ap.parse_args()
    if not 18990 <= a.port <= 18999:
        print("port outside 18990-18999", file=sys.stderr)
        return 2
    os.makedirs(a.home, exist_ok=True)
    os.makedirs(a.out, exist_ok=True)
    tree = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    env = dict(
        os.environ,
        HOME=a.home,
        PYTHONPATH=os.path.join(tree, "python"),
        YUNSHU_AUTH_DISABLED="true",
        YUNSHU_ORPHAN_GRACE_SECONDS=str(a.grace),
        HF_HUB_OFFLINE="1",
    )
    env.pop("YUNSHU_AUTH_TOKEN", None)
    if a.draft:
        env["YUNSHU_VLM_DRAFT"] = a.draft
    for kv in filter(None, a.extra_env.split(",")):
        k, v = kv.split("=", 1)
        env[k] = v
    log = open(os.path.join(a.out, f"server-{a.scenario}-g{int(a.grace)}.log"), "w")  # noqa: SIM115
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "yunshu_cli",
            "serve",
            "--model",
            a.model,
            "--port",
            str(a.port),
        ],
        cwd=tree,
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    url = f"http://127.0.0.1:{a.port}"
    rc = 1
    side: list[subprocess.Popen] = []
    try:
        wait_ready(url, proc)
        if a.shots_dir:
            side = start_console_shots(a, tree, url, env)
        rec = asyncio.run(scenario(a, url))
        res = report(rec, a)
        if not any(
            r.get("tokens") or r.get("completion_tokens")
            for r in res["requests"].values()
        ):
            raise RuntimeError("no tokens received")
        stem = os.path.join(a.out, f"timeline-{a.scenario}-g{int(a.grace)}")
        with open(stem + ".json", "w") as f:
            json.dump(res, f, indent=1)
        with open(stem + ".md", "w") as f:
            f.write(markdown(res))
        print(markdown(res))
        rc = 0
    except BaseException as exc:  # noqa: BLE001
        print("FAIL", repr(exc), file=sys.stderr)
    finally:
        for sp in side:
            with __import__("contextlib").suppress(Exception):
                os.killpg(sp.pid, signal.SIGKILL)
            sp.wait()
        with __import__("contextlib").suppress(Exception):
            os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
    return rc


if __name__ == "__main__":
    sys.exit(main())
