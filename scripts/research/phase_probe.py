"""Phase timeline probe: does every request show queued -> prefill -> decode -> done as the engine
really goes through them? Real model, real server (isolated HOME, port 18990-18999).

Starts the server, sends N long-prompt streaming requests at once (scenario burst) or one then a
joiner (stagger), polls GET /v1/requests at 5 Hz, and also records the `: yunshu-progress` SSE
comments each stream carries. Prints, per request, the first time each phase was seen on each
channel next to the client's own send / first-token / end times, plus prefill progress samples.
Exits nonzero if a request never reported `prefill` although its first token came > 1.5 s after
it was sent.

    python scripts/research/phase_probe.py MODEL --scenario burst --out DIR [--ctx-chars 40000]
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def first_seen(samples: list[tuple[float, str]]) -> dict[str, float]:
    """phase -> time it was first seen (samples are (t, phase) in time order)."""
    out: dict[str, float] = {}
    for t, ph in samples:
        out.setdefault(ph, t)
    return out


def order_ok(seen: dict[str, float]) -> bool:
    """queued/starting <= prefill <= decode <= done wherever both ends were seen."""
    rank = {
        "starting": 0,
        "queued": 0,
        "prefill": 1,
        "decode": 2,
        "complete": 3,
        "done": 3,
    }
    items = sorted(((t, rank[p]) for p, t in seen.items() if p in rank))
    return all(a[1] <= b[1] for a, b in zip(items, items[1:], strict=False))


def missing_prefill(
    seen: dict[str, float], t_send: float, t_first: float, min_gap: float = 1.5
) -> bool:
    return (t_first - t_send) > min_gap and "prefill" not in seen


def load_conc():
    spec = importlib.util.spec_from_file_location("conc_probe", HERE / "conc_probe.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


async def run(url, model, plans, ctx_chars, tokens):
    import httpx

    conc = load_conc()
    t_zero = time.monotonic()
    rec: dict = {"poll": [], "reqs": {}}
    stop = asyncio.Event()

    async def poller(client):
        while not stop.is_set():
            t = time.monotonic() - t_zero
            try:
                r = await client.get(url + "/v1/requests", timeout=5)
                for it in r.json().get("data", r.json().get("requests", [])) or []:
                    rec["poll"].append((t, it))
            except Exception as exc:  # noqa: BLE001
                rec["poll"].append((t, {"error": repr(exc)[:80]}))
            try:
                d = (await client.get(url + "/debug/requests", timeout=5)).json()
                rec.setdefault("debug", []).append(
                    (t, d.get("active"), d.get("waiting"))
                )
            except Exception:  # noqa: BLE001
                rec.setdefault("debug", []).append((t, None, None))
            await asyncio.sleep(0.2)

    async def one(client, name, delay, salt):
        await asyncio.sleep(delay)
        b = conc.body(model, salt, tokens, ctx_chars, False)
        r = rec["reqs"].setdefault(
            name,
            {
                "send": time.monotonic() - t_zero,
                "progress": [],
                "first": None,
                "end": None,
                "id": None,
            },
        )
        async with client.stream("POST", url + "/v1/chat/completions", json=b) as resp:
            r["id"] = resp.headers.get("x-request-id")
            async for line in resp.aiter_lines():
                now = time.monotonic() - t_zero
                if line.startswith(": yunshu-progress"):
                    with contextlib.suppress(ValueError):
                        r["progress"].append((now, json.loads(line.split(" ", 2)[2])))
                elif (
                    line.startswith("data: ")
                    and line != "data: [DONE]"
                    and r["first"] is None
                ):
                    try:
                        d = json.loads(line[6:])
                    except ValueError:
                        continue
                    for ch in d.get("choices") or []:
                        dl = ch.get("delta") or {}
                        if (
                            dl.get("content")
                            or dl.get("reasoning_content")
                            or dl.get("reasoning")
                        ):
                            r["first"] = now
        r["end"] = time.monotonic() - t_zero

    async with httpx.AsyncClient(timeout=httpx.Timeout(900.0)) as client:
        pt = asyncio.create_task(poller(client))
        await asyncio.gather(*(one(client, n, d, s) for n, d, s in plans))
        await asyncio.sleep(1.0)
        stop.set()
        await pt
    return rec


def report(rec) -> tuple[str, int]:
    lines = [
        "| request | sent | polled: queued/starting | prefill | decode | stream: first prefill | first decode | first token | end | order | max prefill % seen |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    bad = 0
    by_id: dict[str, list] = {}
    for t, it in rec["poll"]:
        rid = it.get("request_id")
        if rid:
            by_id.setdefault(rid, []).append((t, it))
    for name, r in rec["reqs"].items():
        rid = r["id"]
        polled = [(t, it.get("phase", "?")) for t, it in by_id.get(rid, [])]
        pseen = first_seen(polled)
        sseen = first_seen([(t, p.get("phase", "?")) for t, p in r["progress"]])
        pct = max(
            [it.get("percent") or 0 for _, it in by_id.get(rid, [])]
            + [p.get("percent") or 0 for _, p in r["progress"]]
            + [0]
        )
        ok = order_ok(pseen) and order_ok(sseen)
        miss = missing_prefill(pseen, r["send"], r["first"] or r["send"])
        smiss = missing_prefill(sseen, r["send"], r["first"] or r["send"])
        bad += int(not ok) + int(miss)

        def f(d, k):
            return f"{d[k]:.1f}" if k in d else "-"

        q = pseen.get("queued", pseen.get("starting"))
        lines.append(
            f"| {name} | {r['send']:.1f} | {q if q is None else f'{q:.1f}'} | {f(pseen, 'prefill')} | {f(pseen, 'decode')} | "
            f"{f(sseen, 'prefill')} | {f(sseen, 'decode')} | {r['first'] and round(r['first'], 1)} | {r['end'] and round(r['end'], 1)} | "
            f"{'ok' if ok else 'BAD'}{' NO-PREFILL(poll)' if miss else ''}{' NO-PREFILL(stream)' if smiss else ''} | {pct} |"
        )
    # Per-second view: counts by phase from /v1/requests next to the engine's own /debug/requests counters.
    lines += [
        "",
        "| t | /v1/requests phases | /debug/requests active | waiting |",
        "|---|---|---|---|",
    ]
    dbg = {int(t): (a, w) for t, a, w in rec.get("debug", [])}
    per: dict[int, dict[str, int]] = {}
    seen_ids: dict[int, set] = {}
    for t, it in rec["poll"]:
        ph = it.get("phase")
        if ph:
            k = int(t)
            if it.get("request_id") in seen_ids.setdefault(k, set()):
                continue
            seen_ids[k].add(it.get("request_id"))
            per.setdefault(k, {}).setdefault(ph, 0)
            per[k][ph] += 1
    for k in sorted(per):
        a, w = dbg.get(k, (None, None))
        lines.append(f"| {k} | {per[k]} | {a} | {w} |")
    return "\n".join(lines), bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--scenario", choices=("burst", "stagger"), default="burst")
    ap.add_argument("--port", type=int, default=18995)
    ap.add_argument(
        "--home", default="/Volumes/P5Plus/yunshu-build/agentload-home-phase"
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--ctx-chars", type=int, default=40000)
    ap.add_argument("--tokens", type=int, default=200)
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument(
        "--tree",
        default="",
        help="serve from this source tree instead of this one (before/after)",
    )
    a = ap.parse_args()
    if not 18990 <= a.port <= 18999:
        return 2
    os.makedirs(a.home, exist_ok=True)
    os.makedirs(a.out, exist_ok=True)
    tree = a.tree or str(HERE.parent.parent)
    env = dict(
        os.environ,
        HOME=a.home,
        PYTHONPATH=os.path.join(tree, "python"),
        YUNSHU_AUTH_DISABLED="true",
        YUNSHU_DEBUG_ROUTES="1",
        HF_HUB_OFFLINE="1",
        YUNSHU_VLM_APC_DISK="0",
    )
    if a.debug:
        env["YUNSHU_LOG_LEVEL"] = "DEBUG"
    log = open(os.path.join(a.out, f"server-{a.scenario}.log"), "w")  # noqa: SIM115
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
    try:
        import urllib.request

        for _ in range(600):
            if proc.poll() is not None:
                raise RuntimeError("server exited")
            try:
                urllib.request.urlopen(url + "/health/ready", timeout=2)
                break
            except Exception:  # noqa: BLE001
                time.sleep(1)
        model = json.load(urllib.request.urlopen(url + "/v1/models"))["data"][0]["id"]
        plans = (
            [(chr(65 + i), 0.0, 11 + i) for i in range(a.n)]
            if a.scenario == "burst"
            else [("A", 0.0, 11), ("B", 6.0, 12)]
        )
        rec = asyncio.run(run(url, model, plans, a.ctx_chars, a.tokens))
        text, bad = report(rec)
        print(text)
        with open(os.path.join(a.out, f"phases-{a.scenario}.json"), "w") as f:
            json.dump({"reqs": rec["reqs"], "poll": rec["poll"]}, f)
        with open(os.path.join(a.out, f"phases-{a.scenario}.md"), "w") as f:
            f.write(text)
        rc = 0 if bad == 0 else 3
    except BaseException as exc:  # noqa: BLE001
        print("FAIL", repr(exc), file=sys.stderr)
    finally:
        with __import__("contextlib").suppress(Exception):
            os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
    return rc


if __name__ == "__main__":
    sys.exit(main())
