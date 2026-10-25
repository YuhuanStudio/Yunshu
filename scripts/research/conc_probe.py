"""Agent-shaped concurrency probe (lead, 2026-10-24).

Streams chat completions the way agent clients do (no sampling params -> the server's model-card defaults, thinking
on), records every token's arrival time per request, and writes per-request timelines so fairness, slowdown under
concurrency and recovery after a peer finishes are visible (1 s buckets). Scenarios:

  solo        one request alone (reference speed)
  burstN      N requests at once
  stagger     A starts; B joins at +JOIN s; B is shorter, so A's speed after B ends shows recovery
  greedy-*    the same with temperature 0 (the old benchmark shape), for contrast

Usage: conc_probe.py --url http://127.0.0.1:PORT --model NAME --out FILE.jsonl [--tokens 1500] [--ctx-chars 16000]
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from pathlib import Path

import httpx

CORPUS = Path("/Volumes/P5Plus/yunshu-build/snapshot014/corpus")


def filler(chars: int, salt: int) -> str:
    texts = sorted(
        p
        for p in CORPUS.rglob("*")
        if p.is_file() and p.suffix in (".txt", ".md", ".py")
    )
    out, i = [], salt
    while sum(len(x) for x in out) < chars and texts:
        out.append(texts[i % len(texts)].read_text(errors="ignore")[:4000])
        i += 7
    return "\n\n".join(out)[:chars]


def body(model: str, salt: int, tokens: int, ctx_chars: int, greedy: bool) -> dict:
    prompt = (
        filler(ctx_chars, salt)
        + f"\n\n[{salt}] Review the material above carefully. Reason step by step about its design, then write a "
        "detailed technical critique with concrete improvement proposals."
    )
    b = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if greedy:
        b["temperature"] = 0
    return b


def run_one(
    url: str, b: dict, name: str, t_zero: float, rows: list, lock: threading.Lock
) -> None:
    times: list[float] = []
    t0 = time.perf_counter()
    usage = None
    finish = None
    with httpx.Client(timeout=httpx.Timeout(1800.0)) as c:
        with c.stream("POST", url + "/v1/chat/completions", json=b) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                d = json.loads(line[6:])
                if d.get("usage"):
                    usage = d["usage"]
                for ch in d.get("choices") or []:
                    delta = ch.get("delta") or {}
                    if (
                        delta.get("content")
                        or delta.get("reasoning_content")
                        or delta.get("reasoning")
                    ):
                        times.append(time.perf_counter())
                    if ch.get("finish_reason"):
                        finish = ch["finish_reason"]
    end = time.perf_counter()
    if finish is None or usage is None:
        raise RuntimeError(
            f"{name}: incomplete stream (finish={finish}, usage={usage})"
        )
    with lock:
        rows.append(
            {
                "name": name,
                "start": round(t0 - t_zero, 3),
                "ttft": round(times[0] - t0, 3) if times else None,
                "end": round(end - t_zero, 3),
                "completion_tokens": usage.get("completion_tokens"),
                "finish": finish,
                "x_yunshu": usage.get("x_yunshu"),
                "token_times": [round(t - t_zero, 3) for t in times],
            }
        )


def scenario(url, model, out, label, plans, tokens, ctx_chars):
    """plans: list of (name, delay_s, tokens, greedy, salt)."""
    rows: list = []
    lock = threading.Lock()
    t_zero = time.perf_counter()
    threads = []
    for name, delay, ntok, greedy, salt in plans:
        b = body(model, salt, ntok or tokens, ctx_chars, greedy)

        def go(b=b, name=name, delay=delay):
            time.sleep(delay)
            run_one(url, b, name, t_zero, rows, lock)

        th = threading.Thread(target=go)
        th.start()
        threads.append(th)
    for th in threads:
        th.join()
    for r in rows:
        tt = r["token_times"]
        # 1 s buckets of tokens/s while the request ran
        buckets: dict[int, int] = {}
        for t in tt:
            buckets[int(t)] = buckets.get(int(t), 0) + 1
        r["tps_by_second"] = [
            buckets.get(s, 0) for s in range(int(r["start"]), int(r["end"]) + 1)
        ]
        span = (tt[-1] - tt[0]) if len(tt) > 1 else 0
        r["decode_tps"] = round((len(tt) - 1) / span, 1) if span else None
        del r["token_times"]
    with open(out, "a") as fh:
        fh.write(
            json.dumps(
                {"scenario": label, "requests": rows, "memory": status_memory(url)}
            )
            + "\n"
        )
    print(label, [(r["name"], r["decode_tps"], r["ttft"]) for r in rows], flush=True)


def status_memory(url: str) -> dict | None:
    """Engine memory block of /v1/yunshu/status (peak is cumulative since start), or None."""
    try:
        r = httpx.get(url + "/v1/yunshu/status", timeout=30.0)
        mem = r.json().get("memory")
        return mem if isinstance(mem, dict) else None
    except (httpx.HTTPError, ValueError):
        return None


def engine_observes(url: str) -> bool:
    """True when the engine at ``url`` reports observe mode (/v1/yunshu/status ``observe``)."""
    try:
        r = httpx.get(url + "/v1/yunshu/status", timeout=30.0)
        r.raise_for_status()
        return r.json().get("observe") is True
    except (httpx.HTTPError, ValueError):
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--allow-no-observe",
        action="store_true",
        help="run against an engine that is not in observe mode (a pure timing run)",
    )
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokens", type=int, default=1500)
    ap.add_argument("--ctx-chars", type=int, default=16000)
    ap.add_argument("--join", type=float, default=15.0)
    ap.add_argument(
        "--only", default="", help="comma list of scenario labels to run (default all)"
    )
    a = ap.parse_args()
    observe = engine_observes(a.url)
    if not observe and not a.allow_no_observe:
        raise SystemExit(
            "the engine is not in observe mode: start it with `yunshu serve --observe` "
            "(YUNSHU_OBSERVE=1) or pass --allow-no-observe"
        )
    with open(a.out, "a") as fh:
        fh.write(json.dumps({"meta": {"observe": observe}}) + "\n")
    T, C = a.tokens, a.ctx_chars
    only = {x for x in a.only.split(",") if x}

    def run(label, plans):
        if only and label not in only and label != "warmup":
            return
        scenario(a.url, a.model, a.out, label, plans, T, C)

    run("warmup", [("w", 0, 64, False, 99)])
    run("solo", [("A", 0, None, False, 1)])
    run("burst2", [("A", 0, None, False, 11), ("B", 0, None, False, 12)])
    run("burst4", [(n, 0, None, False, 20 + i) for i, n in enumerate("ABCD")])
    run("stagger", [("A", 0, None, False, 31), ("B", a.join, T // 3, False, 32)])
    run("greedy-solo", [("A", 0, None, True, 41)])
    run("greedy-burst2", [("A", 0, None, True, 51), ("B", 0, None, True, 52)])
    run("greedy-burst4", [(n, 0, None, True, 60 + i) for i, n in enumerate("ABCD")])
    run("burst8", [(n, 0, None, False, 70 + i) for i, n in enumerate("ABCDEFGH")])
    run("greedy-burst8", [(n, 0, None, True, 80 + i) for i, n in enumerate("ABCDEFGH")])
    print("complete: true", flush=True)


if __name__ == "__main__":
    main()
