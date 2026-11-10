"""Newcomer prefill beside a decoding request, GPU-only vs the ANE FFN split (metrics only).

A runs a greedy stream; B (a long prompt) joins after --join seconds. Reports A's decode tok/s
before B, while B prefills, and after; B's TTFT. One server per job (tfbench Srv, port 18990+).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import threading
import time
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def stream(url, body, t0, rows, name, lock):
    times = []
    start = time.perf_counter()
    usage = finish = None
    with httpx.Client(timeout=httpx.Timeout(1800.0)) as c:
        with c.stream("POST", url + "/v1/chat/completions", json=body) as r:
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
                        times.append(time.perf_counter() - t0)
                    if ch.get("finish_reason"):
                        finish = ch["finish_reason"]
    if finish is None or usage is None:
        raise RuntimeError(f"{name}: incomplete stream")
    with lock:
        rows[name] = {
            "start": start - t0,
            "times": times,
            "tokens": usage.get("completion_tokens"),
            "prompt": usage.get("prompt_tokens"),
            "cached": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
        }


def served_ctx(tokens, chars, legacy_chars, tok, flag):
    """Served prompt tokens for one prompt: --*-ctx-tokens wins; the deprecated character flag is converted
    (and logged); with neither, the legacy character default is converted (and logged)."""
    import prompt_len

    if tokens is not None and chars is not None:
        raise SystemExit(
            f"{flag} is a deprecated alias; give one of it or its --*-ctx-tokens"
        )
    if tokens is not None:
        return tokens
    if chars is not None:
        n = prompt_len.tokens_from_chars(tok, chars)
        print(
            f"warning: {flag} {chars} is deprecated; using {n} served tokens",
            flush=True,
        )
        return n
    n = prompt_len.tokens_from_chars(tok, legacy_chars)
    print(
        f"note: default {flag} ({legacy_chars} legacy characters) is {n} served tokens",
        flush=True,
    )
    return n


def tps(times, lo, hi):
    window = [t for t in times if lo <= t < hi]
    if len(window) < 2:
        return None
    return round((len(window) - 1) / (window[-1] - window[0]), 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", choices=["off", "splash"], required=True)
    p.add_argument("--rep", type=int, default=0)
    p.add_argument("--share", type=float, default=None)
    p.add_argument("--min-rows", type=int, default=None)
    p.add_argument("--join", type=float, default=12.0)
    p.add_argument("--a-tokens", type=int, default=2000)
    p.add_argument(
        "--a-ctx-tokens",
        type=int,
        default=None,
        help="A's served prompt tokens (template included; default: the legacy 1400 characters, converted)",
    )
    p.add_argument(
        "--b-ctx-tokens",
        type=int,
        default=None,
        help="B's served prompt tokens (default: the legacy 60000 characters, converted)",
    )
    p.add_argument(
        "--a-chars",
        type=int,
        default=None,
        help="DEPRECATED alias: characters of A's prompt, converted to served tokens",
    )
    p.add_argument(
        "--b-chars",
        type=int,
        default=None,
        help="DEPRECATED alias: characters of B's prompt, converted to served tokens",
    )
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--scratch", type=Path, required=True)
    p.add_argument(
        "--cache",
        type=Path,
        default=Path("/Volumes/P5Plus/yunshu-build/codex/anenative-ffncache"),
    )
    a = p.parse_args()
    os.environ.update(
        TFB_YUNSHU_SRC=str(ROOT / "python"),
        TFB_OUT=str(a.scratch),
        TFB_PORT_LAST="18996",
    )
    sys.path.insert(0, str(HERE))
    import conc_probe
    import prompt_len
    import tfbench as tb

    tok = prompt_len.load_tokenizer(tb.M)
    a_ctx = served_ctx(a.a_ctx_tokens, a.a_chars, 1400, tok, "--a-chars")
    b_ctx = served_ctx(a.b_ctx_tokens, a.b_chars, 60000, tok, "--b-chars")
    (a.scratch / "tmp").mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(a.scratch / "tmp")
    env = {
        "YUNSHU_CONSOLE": "0",
        "YUNSHU_VLM_APC_DISK": "0",
        "YUNSHU_VLM_DRAFT": tb.D,
        "YUNSHU_ANE_FFN_SPLIT": a.arm,
        "YUNSHU_ANE_FFN_CACHE": str(a.cache),
    }
    if a.share:
        env["YUNSHU_ANE_FFN_SHARE"] = str(a.share)
    if a.min_rows:
        env["YUNSHU_ANE_FFN_MIN_ROWS"] = str(a.min_rows)
    s = tb.start_server(
        "yunshu-main",
        env,
        f"conc-{a.arm}-{a.rep}",
        model=tb.M,
        ctx_tokens=40000,
        parallel=1,
    )
    try:
        for _ in range(2):
            tb.send(s.url, tb.req(s.model, "Say hi.", 24))
        rows: dict = {}
        lock = threading.Lock()
        t0 = time.perf_counter()
        body_a = conc_probe.body(
            s.model, 5, a.a_tokens, None, True, ctx_tokens=a_ctx, tok=tok
        )
        body_b = conc_probe.body(s.model, 6, 32, None, True, ctx_tokens=b_ctx, tok=tok)
        threads = [
            threading.Thread(target=stream, args=(s.url, body_a, t0, rows, "A", lock)),
        ]
        threads[0].start()

        def late():
            time.sleep(a.join)
            stream(s.url, body_b, t0, rows, "B", lock)

        tb_thread = threading.Thread(target=late)
        tb_thread.start()
        threads.append(tb_thread)
        for th in threads:
            th.join()
        A, B = rows["A"], rows["B"]
        prompt_len.check_served_length(a_ctx, {"prompt_tokens": A["prompt"]})
        prompt_len.check_served_length(b_ctx, {"prompt_tokens": B["prompt"]})
        b_start = B["start"]
        b_first = B["times"][0]
        record = {
            "arm": a.arm,
            "rep": a.rep,
            "a_tokens": A["tokens"],
            "b_prompt_tokens": B["prompt"],
            "b_cached": B["cached"],
            "b_ttft_s": round(b_first - b_start, 3),
            "a_tps_before": tps(A["times"], 2.0, b_start),
            "a_tps_during_b_prefill": tps(A["times"], b_start, b_first),
            "a_tokens_during_b_prefill": sum(
                1 for t in A["times"] if b_start <= t < b_first
            ),
            "a_tps_after": tps(A["times"], b_first + 1.0, 1e9),
            "a_first_token_s": round(A["times"][0], 3),
            "a_end_s": round(A["times"][-1], 3),
            "b_start_s": round(b_start, 3),
            "ane_log": [
                line
                for line in s.log.read_text(errors="replace").splitlines()
                if "Neural Engine" in line
            ][-4:],
        }
        a.out.parent.mkdir(parents=True, exist_ok=True)
        with a.out.open("a") as f:
            f.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        _ = statistics
        print(json.dumps({"complete": True}))
    finally:
        s.kill()


if __name__ == "__main__":
    main()
