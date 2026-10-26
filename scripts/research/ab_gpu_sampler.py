"""YUNSHU_GPU_SAMPLER A/B on a real text-only checkpoint (run on the Mac).

The text fast path samples temperature > 0 with a float64 NumPy sampler that copies the logits to
the CPU every token (default), or on the GPU with Gumbel-max (YUNSHU_GPU_SAMPLER=1). The flag is
experimental until this settles it. Two questions, same checkpoint, one server per arm:

1. Speed.  Streamed decode tok/s ((completion_tokens - 1) / time after the first chunk) and the
   inter-chunk latency p50/p95 for several sampling configurations. Arms alternate
   (numpy, gpu, numpy, gpu, ...) over ``--rounds`` rounds so slow drift (thermals, other jobs)
   shows up as round-to-round spread instead of being mistaken for an arm effect.
2. Same distribution.  For each prompt, ``--dist-samples`` requests of ``max_tokens=1`` give an
   empirical first-token distribution per arm. The GPU sampler is the same distribution when
   TV(numpy, gpu) is within sampling noise, which is measured as TV(numpy round 1, numpy round 2)
   of the same arm. Seed reproducibility (same seed twice -> same text) is checked per arm.

Decision rule, printed as the verdict (override with the flags): flip the default to GPU when
the median decode tok/s ratio gpu/numpy is >= ``--min-speedup`` in every configuration, TV
(numpy, gpu) <= ``--tv-slack`` x the numpy-vs-numpy noise floor (and at least ``--tv-floor``),
and the seeded rerun is identical in the GPU arm. Then delete the NumPy sampler and the flag
(AGENTS.md: experimental flags are temporary).

    ab_gpu_sampler.py --checkpoint $M --src python --out ab_gpu_sampler.json

Server: own process, isolated HOME, free port from the shared pool (tool_grammar_replay.Server).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path

PROMPTS = [
    "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "Explain how a hash table handles collisions, with an example.",
    "Give me five creative names for a coffee shop and say why each works.",
    "What are the trade-offs between SQL and NoSQL databases?",
    "Describe a walk through a forest in autumn.",
    "Write a haiku sequence about the sea, then explain the imagery.",
]
CONFIGS = {
    "t0.7": {"temperature": 0.7},
    "t0.7_p0.95": {"temperature": 0.7, "top_p": 0.95},
    "t0.6_p0.8_k20": {"temperature": 0.6, "top_p": 0.8, "top_k": 20},
}
ARMS = {"numpy": "0", "gpu": "1"}


def tv_distance(a: Counter, b: Counter) -> float:
    """Total variation distance between two empirical distributions given as counts."""
    na, nb = sum(a.values()), sum(b.values())
    if not na or not nb:
        return float("nan")
    keys = set(a) | set(b)
    return 0.5 * sum(abs(a[k] / na - b[k] / nb) for k in keys)


def stream_chat(url: str, model: str, prompt: str, max_tokens: int, **sampling) -> dict:
    """One streamed chat completion: text, completion_tokens, ttft, decode tok/s, gaps."""
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
        **sampling,
    }
    req = urllib.request.Request(
        url + "/v1/chat/completions",
        json.dumps(body).encode(),
        {"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    stamps: list[float] = []
    text, tokens = [], 0
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line.endswith("[DONE]"):
                continue
            ev = json.loads(line[5:])
            if ev.get("usage"):
                tokens = ev["usage"]["completion_tokens"]
            for ch in ev.get("choices") or []:
                piece = (ch.get("delta") or {}).get("content")
                if piece:
                    text.append(piece)
                    stamps.append(time.perf_counter())
    out = {"text": "".join(text), "completion_tokens": tokens}
    if stamps:
        out["ttft"] = stamps[0] - t0
        out["gaps"] = [b - a for a, b in zip(stamps, stamps[1:], strict=False)]
        span = stamps[-1] - stamps[0]
        out["tok_s"] = (tokens - 1) / span if tokens > 1 and span > 0 else None
    return out


def pct(values: list[float], q: float) -> float:
    s = sorted(values)
    return s[min(len(s) - 1, int(q * len(s)))] if s else float("nan")


def speed_rows(server, max_tokens: int, reps: int) -> dict[str, dict]:
    """Per configuration: decode tok/s samples and inter-chunk gaps over every prompt x rep."""
    rows = {}
    for name, sampling in CONFIGS.items():
        tok_s, gaps = [], []
        for prompt in PROMPTS:
            for rep in range(reps):
                r = stream_chat(
                    server.url, server.model, prompt, max_tokens, seed=rep, **sampling
                )
                if r.get("tok_s"):
                    tok_s.append(r["tok_s"])
                gaps.extend(r.get("gaps", []))
        rows[name] = {
            "tok_s": tok_s,
            "median_tok_s": statistics.median(tok_s) if tok_s else None,
            "gap_p50_ms": pct(gaps, 0.5) * 1e3,
            "gap_p95_ms": pct(gaps, 0.95) * 1e3,
        }
    return rows


def first_token_counts(server, samples: int) -> dict[str, Counter]:
    """Empirical first-token (text) distribution per prompt; unseeded, so every request draws."""
    out = {}
    for prompt in PROMPTS:
        c: Counter = Counter()
        for _ in range(samples):
            c[
                stream_chat(
                    server.url, server.model, prompt, 1, temperature=0.8, top_p=0.95
                )["text"]
            ] += 1
        out[prompt] = c
    return out


def seed_repro(server) -> bool:
    """Same seed twice gives the same text (and a different seed differs)."""
    kw = dict(temperature=0.9, top_p=0.95)
    a = [
        stream_chat(server.url, server.model, p, 48, seed=7, **kw)["text"]
        for p in PROMPTS[:3]
    ]
    b = [
        stream_chat(server.url, server.model, p, 48, seed=7, **kw)["text"]
        for p in PROMPTS[:3]
    ]
    c = [
        stream_chat(server.url, server.model, p, 48, seed=8, **kw)["text"]
        for p in PROMPTS[:3]
    ]
    return a == b and a != c


def verdict(report: dict, min_speedup: float, tv_slack: float, tv_floor: float) -> dict:
    ratios = {}
    for cfg in CONFIGS:
        med = {
            arm: statistics.median(
                [
                    rnd["speed"][cfg]["median_tok_s"]
                    for rnd in report["rounds"]
                    if rnd["arm"] == arm and rnd["speed"][cfg]["median_tok_s"]
                ]
                or [float("nan")]
            )
            for arm in ARMS
        }
        ratios[cfg] = med["gpu"] / med["numpy"]
    noise = report[
        "tv_numpy_vs_numpy"
    ]  # None with a single round: only the floor applies
    cross = report["tv_numpy_vs_gpu"]
    limit = max(tv_floor, tv_slack * (noise or 0.0))
    checks = {
        "speedup_every_config": all(r >= min_speedup for r in ratios.values()),
        "same_distribution": cross <= limit,
        "gpu_seed_reproducible": report["seed_repro"]["gpu"],
        "numpy_seed_reproducible": report["seed_repro"]["numpy"],
    }
    return {
        "ratios_gpu_over_numpy": ratios,
        "tv_numpy_vs_gpu": cross,
        "tv_noise_floor": noise,
        "tv_limit": limit,
        "checks": checks,
        "flip_default_to_gpu": all(checks.values()),
    }


def mean_tv(
    per_prompt_a: dict[str, Counter], per_prompt_b: dict[str, Counter]
) -> float:
    return statistics.mean(
        tv_distance(per_prompt_a[p], per_prompt_b[p]) for p in PROMPTS
    )


def main() -> None:
    sys.path.insert(0, str(Path(__file__).parent))
    from tool_grammar_replay import start_server

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--checkpoint", required=True, help="text-only mlx-lm checkpoint")
    ap.add_argument(
        "--src", default=None, help="PYTHONPATH for the servers (e.g. python)"
    )
    ap.add_argument("--out", default="ab_gpu_sampler.json")
    ap.add_argument("--logdir", default="ab-gpu-sampler-logs")
    ap.add_argument(
        "--rounds", type=int, default=2, help="alternations of the two arms"
    )
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument(
        "--reps", type=int, default=3, help="requests per prompt per config"
    )
    ap.add_argument(
        "--dist-samples", type=int, default=200, help="first-token draws per prompt"
    )
    ap.add_argument("--min-speedup", type=float, default=1.05)
    ap.add_argument("--tv-slack", type=float, default=1.5)
    ap.add_argument("--tv-floor", type=float, default=0.05)
    args = ap.parse_args()

    report: dict = {"args": vars(args), "rounds": [], "seed_repro": {}}
    first: dict[str, list[dict]] = {arm: [] for arm in ARMS}
    for rnd in range(args.rounds):
        for arm, flag in ARMS.items():
            server = start_server(
                args.checkpoint,
                args.src,
                {"YUNSHU_GPU_SAMPLER": flag},
                Path(args.logdir) / f"{arm}-{rnd}.log",
            )
            try:
                stream_chat(
                    server.url, server.model, "Say hi.", 16, temperature=0.7
                )  # warm
                row = {
                    "round": rnd,
                    "arm": arm,
                    "speed": speed_rows(server, args.max_tokens, args.reps),
                }
                first[arm].append(first_token_counts(server, args.dist_samples))
                if rnd == 0:
                    report["seed_repro"][arm] = seed_repro(server)
            finally:
                server.kill()
            report["rounds"].append(row)
            print(
                arm,
                rnd,
                {k: round(v["median_tok_s"] or 0, 1) for k, v in row["speed"].items()},
                flush=True,
            )

    # same sample size on both sides: round r of one arm against round r of the other
    report["tv_numpy_vs_gpu"] = statistics.mean(
        mean_tv(first["numpy"][r], first["gpu"][r]) for r in range(args.rounds)
    )
    report["tv_numpy_vs_numpy"] = (
        mean_tv(first["numpy"][0], first["numpy"][1]) if args.rounds >= 2 else None
    )
    report["verdict"] = verdict(report, args.min_speedup, args.tv_slack, args.tv_floor)
    Path(args.out).write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps(report["verdict"], indent=1))
    print("flip default to GPU sampler:", report["verdict"]["flip_default_to_gpu"])


if __name__ == "__main__":
    main()
