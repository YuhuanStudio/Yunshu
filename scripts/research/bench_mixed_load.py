"""Decode under concurrent prefill: does offloading prefill (e.g. to the ANE) help a busy GPU?

N background requests stream long generations. Once they are all decoding, one
long-prompt request (``--pp`` tokens, unique prefix so no cache hit) arrives.
Records, per background stream, the decode rate before the arrival, during the
long request's prefill (arrival -> its first token) and after; plus the long
request's TTFT and the aggregate tok/s over the whole run. A GPU-only engine
shows background decode collapsing during the prefill window; an engine that
moves prefill work to other hardware should keep more of it.

    python scripts/research/bench_mixed_load.py --url http://127.0.0.1:18764 \
        --model Qwen3.8-27B --tokenizer <ckpt> --streams 4 --pp 16384 \
        --output runs/mixed-load.jsonl --label yunshu
"""

import argparse
import http.client
import json
import sys
import threading
import time
import urllib.parse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench_context_batch import CORPUS, make_prompt  # noqa: E402


def stream(url, model, content, max_tokens, times, first_box):
    u = urllib.parse.urlparse(url)
    body = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "enable_thinking": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    conn = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=3600)
    t0 = time.perf_counter()
    conn.request(
        "POST",
        "/v1/chat/completions",
        json.dumps(body),
        {"Content-Type": "application/json"},
    )
    resp = conn.getresponse()
    for line in resp:
        if not line.startswith(b"data: ") or line[6:].strip() == b"[DONE]":
            continue
        ev = json.loads(line[6:])
        for ch in ev.get("choices") or []:
            if (ch.get("delta") or {}).get("content"):
                now = time.perf_counter()
                if first_box is not None and not first_box:
                    first_box.append(now - t0)
                times.append(now)
    conn.close()


def rate(times, a, b):
    n = sum(1 for t in times if a <= t < b)
    return round(n / (b - a), 1) if b > a else None


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--streams", type=int, default=4)
    ap.add_argument("--pp", type=int, default=16384)
    ap.add_argument("--gen", type=int, default=1500, help="background stream length")
    ap.add_argument(
        "--warm-s",
        type=float,
        default=8.0,
        help="decode time before the long prompt arrives",
    )
    ap.add_argument("--label", default="")
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    corpus = CORPUS.read_text()
    long_prompt = make_prompt(tok, corpus, a.pp)
    bg_times = [[] for _ in range(a.streams)]
    threads = [
        threading.Thread(
            target=stream,
            args=(
                a.url,
                a.model,
                f"Story {i}: write a very long, detailed story about a lighthouse keeper.",
                a.gen,
                bg_times[i],
                None,
            ),
        )
        for i in range(a.streams)
    ]
    t_start = time.perf_counter()
    for t in threads:
        t.start()
    time.sleep(a.warm_s)
    long_times, first = [], []
    t_arrive = time.perf_counter()
    long_t = threading.Thread(
        target=stream, args=(a.url, a.model, long_prompt, 128, long_times, first)
    )
    long_t.start()
    long_t.join()
    t_long_done = time.perf_counter()
    for t in threads:
        t.join()
    t_end = time.perf_counter()
    t_first = t_arrive + first[0] if first else t_long_done
    row = {
        "label": a.label,
        "streams": a.streams,
        "pp": a.pp,
        "long_ttft_s": round(first[0], 2) if first else None,
        "bg_rate_before": [rate(ts, t_start + 2, t_arrive) for ts in bg_times],
        "bg_rate_during_prefill": [rate(ts, t_arrive, t_first) for ts in bg_times],
        "bg_rate_after": [
            rate(ts, t_first, min(t_end, t_first + 10)) for ts in bg_times
        ],
        "aggregate_tps": round(
            (sum(len(ts) for ts in bg_times) + len(long_times)) / (t_end - t_start), 1
        ),
        "wall_s": round(t_end - t_start, 1),
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
