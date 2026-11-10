"""Agent-shaped concurrency probe (lead, 2026-10-24).

Streams chat completions the way agent clients do (no sampling params -> the server's model-card defaults, thinking
on), records every token's arrival time per request, and writes per-request timelines so fairness, slowdown under
concurrency and recovery after a peer finishes are visible (1 s buckets). Scenarios:

  solo        one request alone (reference speed)
  burstN      N requests at once
  stagger     A starts; B joins at +JOIN s; B is shorter, so A's speed after B ends shows recovery
  greedy-*    the same with temperature 0 (the old benchmark shape), for contrast

Usage: conc_probe.py --url http://127.0.0.1:PORT --model NAME --out FILE.jsonl [--tokens 1500] [--ctx-tokens 4096]

The context length is the SERVED prompt length (usage.prompt_tokens of the request, template included; see
prompt_len.py): every request is checked against it and the run fails on a mismatch. --ctx-chars is a deprecated
alias (characters of filler), converted to tokens with the tokenizer and logged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import threading
import time
from pathlib import Path

import httpx

try:
    from . import prompt_len
except ImportError:  # run as a script
    import prompt_len

CORPUS = Path("/Volumes/P5Plus/yunshu-build/snapshot014/corpus")


def filler(chars: int, salt: int) -> str:
    texts = sorted(
        p
        for p in CORPUS.rglob("*")
        if p.is_file() and p.suffix in (".txt", ".md", ".py")
    )
    if not texts:
        raise FileNotFoundError(f"no filler corpus under {CORPUS}")
    out, i = [], salt
    while sum(len(x) for x in out) < chars and texts:
        out.append(texts[i % len(texts)].read_text(errors="ignore")[:4000])
        i += 7
    return "\n\n".join(out)[:chars]


def body(
    model: str,
    salt: int,
    tokens: int,
    ctx_chars: int | None,
    greedy: bool,
    ctx_tokens: int | None = None,
    tok=None,
) -> dict:
    """ctx_tokens (with a tokenizer) = served prompt length, exact; ctx_chars = legacy characters (callers that
    calibrate chars themselves, see tests/unit/test_prompt_len_guard.py)."""
    ask = (
        f"\n\n[{salt}] Review the material above carefully. Reason step by step about its design, then write a "
        "detailed technical critique with concrete improvement proposals."
    )
    if ctx_tokens is not None:
        prompt = prompt_len.build_text(
            tok or prompt_len.load_tokenizer(),
            filler(ctx_tokens * 6, salt),
            ctx_tokens,
            suffix=ask,
        )
    else:
        prompt = filler(ctx_chars, salt) + ask
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
    url: str,
    b: dict,
    name: str,
    t_zero: float,
    rows: list,
    lock: threading.Lock,
    label: int | None = None,
) -> None:
    times: list[float] = []
    t0 = time.perf_counter()
    usage = None
    finish = None
    digest = hashlib.sha256()
    queue_reasons: set = set()
    xy = None
    with httpx.Client(timeout=httpx.Timeout(1800.0)) as c:
        with c.stream("POST", url + "/v1/chat/completions", json=b) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if line.startswith(": yunshu-progress"):
                    try:
                        prog = json.loads(line.split("yunshu-progress", 1)[1])
                    except ValueError:
                        prog = {}
                    if prog.get("queue_reason"):
                        queue_reasons.add(prog["queue_reason"])
                    continue
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                d = json.loads(line[6:])
                if d.get("x_yunshu"):
                    xy = d["x_yunshu"]
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
                        digest.update(
                            (
                                delta.get("content")
                                or delta.get("reasoning_content")
                                or delta.get("reasoning")
                            ).encode()
                        )
                    if ch.get("finish_reason"):
                        finish = ch["finish_reason"]
    end = time.perf_counter()
    if finish is None or usage is None:
        raise RuntimeError(
            f"{name}: incomplete stream (finish={finish}, usage={usage})"
        )
    served = prompt_len.check_served_length(label, usage) if label is not None else None
    with lock:
        rows.append(
            {
                "name": name,
                "label_len": label,
                "served_len": served,
                "start": round(t0 - t_zero, 3),
                "ttft": round(times[0] - t0, 3) if times else None,
                "end": round(end - t_zero, 3),
                "completion_tokens": usage.get("completion_tokens"),
                "finish": finish,
                "text_sha": digest.hexdigest()[:16],
                "x_yunshu": usage.get("x_yunshu") or xy,
                "queue_reasons": sorted(queue_reasons),
                "token_times": [round(t - t_zero, 3) for t in times],
            }
        )


def scenario(url, model, out, label, plans, tokens, ctx_tokens, tok=None):
    """plans: list of (name, delay_s, tokens, greedy, salt)."""
    rows: list = []
    lock = threading.Lock()
    t_zero = time.perf_counter()
    threads = []
    for name, delay, ntok, greedy, salt in plans:
        b = body(model, salt, ntok or tokens, None, greedy, ctx_tokens, tok)

        def go(b=b, name=name, delay=delay):
            time.sleep(delay)
            run_one(url, b, name, t_zero, rows, lock, ctx_tokens)

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


def ctx_tokens_from_args(a, tok) -> int:
    """--ctx-tokens wins; deprecated --ctx-chars is converted (and logged); default 4096."""
    if a.ctx_tokens is not None and a.ctx_chars is not None:
        raise SystemExit("--ctx-chars is a deprecated alias of --ctx-tokens; give one")
    if a.ctx_chars is not None:
        n = prompt_len.tokens_from_chars(tok, a.ctx_chars)
        print(
            f"warning: --ctx-chars {a.ctx_chars} is deprecated; using --ctx-tokens {n}",
            flush=True,
        )
        return n
    return 4096 if a.ctx_tokens is None else a.ctx_tokens


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
    ap.add_argument(
        "--ctx-tokens",
        type=int,
        default=None,
        help="served prompt tokens (default 4096)",
    )
    ap.add_argument(
        "--ctx-chars",
        type=int,
        default=None,
        help="DEPRECATED alias: characters, converted to tokens",
    )
    ap.add_argument(
        "--tokenizer",
        default=prompt_len.DEFAULT_MODEL,
        help="tokenizer dir for the length fit",
    )
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
    tok = prompt_len.load_tokenizer(a.tokenizer)
    C = ctx_tokens_from_args(a, tok)
    T = a.tokens
    only = {x for x in a.only.split(",") if x}

    def run(label, plans):
        if only and label not in only and label != "warmup":
            return
        scenario(a.url, a.model, a.out, label, plans, T, C, tok)

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
