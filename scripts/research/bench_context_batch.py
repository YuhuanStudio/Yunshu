"""oMLX-style speed sweep over any OpenAI-compatible server: context x batch.

Mirrors oMLX's admin benchmark (omlx/admin/benchmark.py) over HTTP so every
engine gets the same test: single requests at prompt lengths 1K..200K tokens
(``pp``) generating 128 tokens (``tg``), then 2/4/8 concurrent requests at
pp1024. Every prompt starts with a unique ``BENCH-<uuid>`` prefix so no prefix
cache can hit; the body is oMLX's code_python corpus. Prompt text is sized with
the checkpoint's tokenizer; the server's reported prompt_tokens is recorded.

Per request: TTFT, prefill tok/s (prompt / TTFT), decode tok/s (tokens after
the first / time after the first). Per batch: aggregate decode tok/s (all
completion tokens / wall) and mean TTFT.

    python scripts/research/bench_context_batch.py --url http://127.0.0.1:18764 \
        --model Qwen3.8-27B --tokenizer /Volumes/.../Qwen3.8-27B-oQ4e-mtp \
        --pid 1234 --output runs/speed-yunshu.jsonl
"""

import argparse
import concurrent.futures
import http.client
import json
import sys
import time
import urllib.parse
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from process_memory import process_tree_memory  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "reference/omlx/omlx/admin/bench_corpora/code_python.txt"
LENGTHS = [1024, 4096, 8192, 16384, 32768, 65536, 131072, 200000]
BATCHES = [2, 4, 8]
INSTRUCTION = "\n\nContinue this code with more functions. Output only code."


def make_prompt(tokenizer, corpus, target):
    prefix = f"BENCH-{uuid.uuid4().hex} "
    body_tokens = target - len(tokenizer.encode(prefix + INSTRUCTION))
    ids = []
    text = corpus
    while len(ids) < body_tokens:
        ids = tokenizer.encode(text)
        text += corpus
    return prefix + tokenizer.decode(ids[:body_tokens]) + INSTRUCTION


def stream(url, model, prompt, max_tokens, timeout):
    u = urllib.parse.urlparse(url)
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "enable_thinking": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    conn = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=timeout)
    t0 = time.perf_counter()
    first = last = None
    chunks, usage = 0, None
    conn.request(
        "POST",
        "/v1/chat/completions",
        json.dumps(body),
        {"Content-Type": "application/json"},
    )
    resp = conn.getresponse()
    if resp.status != 200:
        return {"error": f"HTTP {resp.status}: {resp.read()[:200]!r}"}
    for line in resp:
        if not line.startswith(b"data: ") or line[6:].strip() == b"[DONE]":
            continue
        ev = json.loads(line[6:])
        usage = ev.get("usage") or usage
        for ch in ev.get("choices") or []:
            d = ch.get("delta") or {}
            if d.get("content") or d.get("reasoning_content") or d.get("reasoning"):
                now = time.perf_counter()
                first = first or now
                last = now
                chunks += 1
    conn.close()
    end = time.perf_counter()
    usage = usage or {}
    n = usage.get("completion_tokens") or chunks
    ttft = (first - t0) if first else None
    dec = (
        (n - 1) / (last - first) if first and last and last > first and n > 1 else None
    )
    pt = usage.get("prompt_tokens")
    return {
        "prompt_tokens": pt,
        "cached_tokens": (usage.get("prompt_tokens_details") or {}).get(
            "cached_tokens"
        ),
        "completion_tokens": n,
        "ttft_s": round(ttft, 3) if ttft else None,
        "prefill_tps": round(pt / ttft, 1) if pt and ttft else None,
        "decode_tps": round(dec, 1) if dec else None,
        "t_start": t0,
        "t_end": end,
    }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument(
        "--tokenizer", required=True, help="checkpoint dir with the tokenizer"
    )
    ap.add_argument("--pid", type=int)
    ap.add_argument("--lengths", type=int, nargs="*", default=LENGTHS)
    ap.add_argument("--batches", type=int, nargs="*", default=BATCHES)
    ap.add_argument("--batch-pp", type=int, default=1024)
    ap.add_argument("--tg", type=int, default=128)
    ap.add_argument("--timeout", type=float, default=3600)
    ap.add_argument("--note", default="")
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    corpus = CORPUS.read_text()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    out = a.output.open("a")

    def mem():
        if not a.pid:
            return None
        try:
            return round(
                process_tree_memory(a.pid)["physical_footprint_sum_bytes"] / 2**30, 3
            )
        except Exception:  # noqa: BLE001
            return None

    def emit(row):
        out.write(json.dumps(row) + "\n")
        out.flush()
        print(
            json.dumps(
                {
                    k: v
                    for k, v in row.items()
                    if k not in ("t_start", "t_end", "requests")
                }
            ),
            flush=True,
        )

    emit(
        {
            "kind": "meta",
            "url": a.url,
            "model": a.model,
            "tg": a.tg,
            "note": a.note,
            "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "footprint_gib": mem(),
        }
    )
    # Warm-up (not recorded): first-call compile and allocator growth.
    stream(a.url, a.model, make_prompt(tok, corpus, 512), 16, a.timeout)
    for pp in a.lengths:
        r = stream(a.url, a.model, make_prompt(tok, corpus, pp), a.tg, a.timeout)
        emit({"kind": "single", "pp": pp, **r, "footprint_gib": mem()})
        if r.get("error"):
            break
    for bs in a.batches:
        prompts = [make_prompt(tok, corpus, a.batch_pp) for _ in range(bs)]
        t0 = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(bs) as pool:
            rs = list(
                pool.map(lambda p: stream(a.url, a.model, p, a.tg, a.timeout), prompts)
            )
        wall = time.perf_counter() - t0
        ok = [r for r in rs if not r.get("error")]
        total = sum(r["completion_tokens"] or 0 for r in ok)
        emit(
            {
                "kind": "batch",
                "batch_size": bs,
                "pp": a.batch_pp,
                "ok": len(ok),
                "wall_s": round(wall, 2),
                "aggregate_tps": round(total / wall, 1) if wall else None,
                "mean_ttft_s": round(
                    sum(r["ttft_s"] or 0 for r in ok) / max(1, len(ok)), 3
                ),
                "max_ttft_s": max((r["ttft_s"] or 0 for r in ok), default=None),
                "mean_decode_tps": round(
                    sum(r["decode_tps"] or 0 for r in ok) / max(1, len(ok)), 1
                ),
                "footprint_gib": mem(),
                "requests": [
                    {k: v for k, v in r.items() if k not in ("t_start", "t_end")}
                    for r in rs
                ],
            }
        )


if __name__ == "__main__":
    main()
