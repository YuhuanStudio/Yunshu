"""YUNSHU_APC_SEAM_CHECKPOINT A/B on a real VLM-runner checkpoint (run on the Mac).

"Same document, new question": a long document goes in one user message with question 1, and a
moment later the same document comes back with question 2 (TensorFold #1262: 35k tokens on
Qwen3.8-27B, first token 125.9 s -> 9.0 s; Rapid-MLX 421bf658). The flag keeps one extra APC
checkpoint 2048 tokens (64-aligned) before the end of a final user message of >= 8192 tokens.
It is experimental until this settles it. Per document size, one server per arm:

1. Request A = document + question 1 (cold: nothing cached). TTFT_A on vs off is the price of
   the extra checkpoint copy on the first request.
2. Request B = document + question 2. TTFT_B on vs off is the gain; ``cached_tokens`` says where
   it resumed (off: the system turn or an interval cut; on: the seam, ~document end).
3. A fresh server answers B first (true cold B). B-after-A must equal cold B token for token,
   in both arms, and A / B must be identical across arms (an aligned cut is split-invariant).

Arms alternate (off, on, off, on, ...) over ``--rounds`` so drift shows as round-to-round spread.
Peak MLX memory comes from /metrics when the server exposes it (``mlx_peak_memory_bytes``).

Decision rule, printed as the verdict (override with the flags): make the seam checkpoint the
default (delete the flag, keep the cut) when, for every size >= 16k, median TTFT_B off/on >=
``--min-speedup``, the TTFT_A cost on/off - 1 <= ``--max-first-cost``, and every token
comparison above is identical; otherwise delete the cut and the flag.

    ab_seam_checkpoint.py --checkpoint $M --src python --out ab_seam_checkpoint.json

Server: own process, isolated HOME, free port from the shared pool (tool_grammar_replay.Server).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.request
from pathlib import Path

SIZES = (16_384, 35_000, 65_536)
Q1 = "Question one: list the three most important claims in the document above."
Q2 = "Question two: write a one-sentence title for the document above."
ARMS = {"off": "0", "on": "1"}

_PARAGRAPHS = [
    "The harbour authority met on {d} to review the {n}th revision of the dredging plan. ",
    "Engineers reported that sediment samples from berth {n} showed a grain size of {m} microns. ",
    "A dissenting member noted that the cost per cubic metre had risen {m} percent since {d}. ",
    "The committee agreed to defer the decision on pier {n} until the survey of {d} is complete. ",
]


def make_document(tokens: int) -> str:
    """Deterministic, non-repeating prose of about ``tokens`` tokens (~4.2 characters each)."""
    out, i = [], 0
    while sum(map(len, out)) < tokens * 4.2:
        p = _PARAGRAPHS[i % len(_PARAGRAPHS)]
        out.append(
            p.format(n=i, m=(i * 37) % 997, d=f"day {i % 365} of year {2000 + i % 25}")
        )
        i += 1
    return "".join(out)


def stream_chat(
    url: str, model: str, document: str, question: str, max_tokens: int
) -> dict:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": f"{document}\n\n{question}"}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        url + "/v1/chat/completions",
        json.dumps(body).encode(),
        {"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    first, text, usage = None, [], {}
    with urllib.request.urlopen(req, timeout=7200) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line.endswith("[DONE]"):
                continue
            ev = json.loads(line[5:])
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices") or []:
                piece = (ch.get("delta") or {}).get("content")
                if piece:
                    first = first or time.perf_counter()
                    text.append(piece)
    details = usage.get("prompt_tokens_details") or {}
    return {
        "text": "".join(text),
        "ttft_s": (first - t0) if first else None,
        "prompt_tokens": usage.get("prompt_tokens"),
        "cached_tokens": details.get("cached_tokens"),
    }


def peak_memory(url: str) -> float | None:
    try:
        with urllib.request.urlopen(url + "/metrics", timeout=10) as r:
            for line in r.read().decode().splitlines():
                if line.startswith("mlx_peak_memory_bytes"):
                    return float(line.split()[-1])
    except Exception:
        pass
    return None


def run_arm(args, arm: str, rnd: int, size: int, start_server) -> dict:
    doc = make_document(size)
    row: dict = {"arm": arm, "round": rnd, "size": size}
    flag = {"YUNSHU_APC_SEAM_CHECKPOINT": ARMS[arm]}
    server = start_server(
        args.checkpoint, args.src, flag, Path(args.logdir) / f"{arm}-{size}-{rnd}.log"
    )
    try:
        stream_chat(server.url, server.model, "warm up", "Say hi.", 8)
        row["a"] = stream_chat(server.url, server.model, doc, Q1, args.max_tokens)
        row["b"] = stream_chat(server.url, server.model, doc, Q2, args.max_tokens)
        row["peak_memory_bytes"] = peak_memory(server.url)
    finally:
        server.kill()
    server = start_server(
        args.checkpoint,
        args.src,
        flag,
        Path(args.logdir) / f"{arm}-{size}-{rnd}-cold.log",
    )
    try:
        stream_chat(server.url, server.model, "warm up", "Say hi.", 8)
        row["b_cold"] = stream_chat(server.url, server.model, doc, Q2, args.max_tokens)
    finally:
        server.kill()
    return row


def med(rows: list[dict], arm: str, size: int, req: str) -> float:
    vals = [
        r[req]["ttft_s"]
        for r in rows
        if r["arm"] == arm and r["size"] == size and r[req]["ttft_s"]
    ]
    return statistics.median(vals) if vals else float("nan")


def verdict(rows: list[dict], sizes, min_speedup: float, max_first_cost: float) -> dict:
    per_size, identical = {}, True
    for size in sizes:
        speed = med(rows, "off", size, "b") / med(rows, "on", size, "b")
        cost = med(rows, "on", size, "a") / med(rows, "off", size, "a") - 1
        per_size[size] = {"ttft_b_off_over_on": speed, "ttft_a_cost": cost}
    for r in rows:
        identical &= r["b"]["text"] == r["b_cold"]["text"]
    for size in sizes:
        for req in ("a", "b"):
            texts = {r[req]["text"] for r in rows if r["size"] == size}
            identical &= len(texts) == 1
    big = [s for s in sizes if s >= 16_384]
    checks = {
        "speedup_every_size": all(
            per_size[s]["ttft_b_off_over_on"] >= min_speedup for s in big
        ),
        "first_request_cost": all(
            per_size[s]["ttft_a_cost"] <= max_first_cost for s in big
        ),
        "tokens_identical": identical,
    }
    return {
        "per_size": per_size,
        "checks": checks,
        "make_default": all(checks.values()),
    }


def main() -> None:
    sys.path.insert(0, str(Path(__file__).parent))
    from tool_grammar_replay import start_server

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--checkpoint", required=True, help="Qwen3.x VLM-runner checkpoint")
    ap.add_argument(
        "--src", default=None, help="PYTHONPATH for the servers (e.g. python)"
    )
    ap.add_argument("--out", default="ab_seam_checkpoint.json")
    ap.add_argument("--logdir", default="ab-seam-checkpoint-logs")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--sizes", type=int, nargs="+", default=list(SIZES))
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--min-speedup", type=float, default=3.0)
    ap.add_argument("--max-first-cost", type=float, default=0.03)
    args = ap.parse_args()

    rows: list[dict] = []
    for rnd in range(args.rounds):
        for size in args.sizes:
            for arm in ARMS:
                row = run_arm(args, arm, rnd, size, start_server)
                rows.append(row)
                print(
                    arm,
                    size,
                    rnd,
                    {
                        "ttft_a": row["a"]["ttft_s"],
                        "ttft_b": row["b"]["ttft_s"],
                        "cached_b": row["b"]["cached_tokens"],
                    },
                    flush=True,
                )
    report = {
        "args": vars(args),
        "rows": rows,
        "verdict": verdict(rows, args.sizes, args.min_speedup, args.max_first_cost),
    }
    Path(args.out).write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps(report["verdict"], indent=1))
    print("seam checkpoint becomes the default:", report["verdict"]["make_default"])


if __name__ == "__main__":
    main()
