"""S5 vs S6 comparison (CPU only).

    s6_compare.py <s5 bench dir> <s6 bench dir>

Per client: pass rate (Wilson), cache, TTFT, wall; requests that ended finish=length / reached the
completion cap; reasoning-length distribution from the stream captures (tokens before the closing
think tag; a request with no closing tag counts all its tokens); sampling applied (x_yunshu.sampling
and mode); failure taxonomy (stalled turn = last request ended without a tool call, timeout).
"""

from __future__ import annotations

import glob
import json
import os
import statistics
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from s5_report import load, stalled, wilson  # noqa: E402

CAP = 32768


def is_length(q: dict) -> bool:
    return (
        q.get("finish_reason") in ("length", "max_tokens")
        or (q.get("completion_tokens") or 0) >= CAP
    )


def reasoning_lengths(d: str) -> dict[str, list[int]]:
    out: dict[str, list[int]] = defaultdict(list)
    for p in sorted(glob.glob(d + "/*.capture.jsonl")):
        agent = os.path.basename(p).split("-")[0]
        for line in open(p):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            pieces = r.get("pieces") or []
            n = len(r.get("token_ids") or pieces)
            end = next((i for i, s in enumerate(pieces) if "</think>" in s), None)
            out[agent].append(end if end is not None else n)
    return out


def pct(xs: list[int], q: float) -> int:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else 0


def extra(runs: list[dict]) -> dict:
    """Decode tok/s of long generations, overlap with another in-flight request, identical retries."""
    ov = rt = rt_cold = 0
    tps_all, tps_long, tps_solo, tps_conc = [], [], [], []
    for r in runs:
        log = r.get("request_log") or []
        for i, q in enumerate(log):
            a, b = q.get("t") or 0, (q.get("t") or 0) + (q.get("total_s") or 0)
            conc = any(
                j != i
                and (p.get("t") or 0) < b
                and (p.get("t") or 0) + (p.get("total_s") or 0) > a
                for j, p in enumerate(log)
            )
            ov += conc
            t = q.get("decode_tok_s")
            if t and (q.get("completion_tokens") or 0) >= 200:
                tps_all.append(t)
                (tps_conc if conc else tps_solo).append(t)
                if (q.get("completion_tokens") or 0) >= 2000:
                    tps_long.append(t)
            if (
                i
                and q.get("n_messages") == log[i - 1].get("n_messages")
                and q.get("prompt_tokens") == log[i - 1].get("prompt_tokens")
            ):
                rt += 1
                rt_cold += (q.get("cached_tokens") or 0) < 0.5 * (
                    q.get("prompt_tokens") or 1
                )
    med = lambda x: round(statistics.median(x), 1) if x else None  # noqa: E731
    return dict(
        overlap_reqs=ov,
        tps_med=med(tps_all),
        tps_long_med=med(tps_long),
        tps_solo=med(tps_solo),
        tps_conc=med(tps_conc),
        n_conc=len(tps_conc),
        retries=rt,
        retries_cold=rt_cold,
    )


def summarize(d: str) -> dict:
    by = load(sorted(glob.glob(d + "/*.jsonl")))
    by = {a: r for a, r in by.items()}
    rl = reasoning_lengths(d)
    res = {}
    for a, runs in sorted(by.items()):
        n = len(runs)
        k = sum(bool(r.get("passed")) for r in runs)
        lo, hi = wilson(k, n)
        reqs = [q for r in runs for q in (r.get("request_log") or [])]
        pt = sum(r.get("prompt_tokens") or 0 for r in runs)
        ct = sum(r.get("cached_tokens") or 0 for r in runs)
        ttft = [q["ttft_s"] for q in reqs if q.get("ttft_s") is not None]
        samp = Counter()
        for q in reqs:
            x = q.get("x_yunshu") or {}
            s = x.get("sampling") or {}
            samp[
                json.dumps(
                    {k2: s.get(k2) for k2 in sorted(s) if k2 != "source"}
                    | {"mode": s.get("mode") or x.get("mode")},
                    sort_keys=True,
                )[:300]
            ] += 1
        res[a] = dict(
            n=n,
            k=k,
            ci=(round(100 * lo), round(100 * hi)),
            cache=round(100 * ct / pt, 1) if pt else 0,
            ttft=round(statistics.median(ttft), 2) if ttft else None,
            wall=round(statistics.median([r["wall_s"] for r in runs]), 0),
            reqs=len(reqs),
            len_reqs=sum(is_length(q) for q in reqs),
            len_runs=sum(
                any(is_length(q) for q in (r.get("request_log") or [])) for r in runs
            ),
            timeouts=sum(bool(r.get("timed_out")) for r in runs),
            stalled_fail=sum((not r.get("passed")) and stalled(r) for r in runs),
            fail_other=sum(
                (not r.get("passed")) and not stalled(r) and not r.get("timed_out")
                for r in runs
            ),
            reason=dict(
                n=len(rl.get(a, [])),
                p50=pct(rl.get(a, []), 0.5),
                p90=pct(rl.get(a, []), 0.9),
                max=max(rl.get(a, []) or [0]),
                gt8k=sum(x > 8192 for x in rl.get(a, [])),
                gt20k=sum(x > 20000 for x in rl.get(a, [])),
            ),
            extra=extra(runs),
            sampling=samp.most_common(6),
            fails=[
                (
                    r["task"],
                    r["repeat"],
                    r.get("wall_s"),
                    "timeout"
                    if r.get("timed_out")
                    else "stalled"
                    if stalled(r)
                    else "other",
                )
                for r in runs
                if not r.get("passed")
            ],
        )
    return res


if __name__ == "__main__":
    s5, s6 = summarize(sys.argv[1]), summarize(sys.argv[2])
    print(json.dumps({"S5": s5, "S6": s6}, indent=1, default=str))
