"""Summarize session_replay.py results (multi scenario) per configuration.

    tier_report.py runs/2026-10-02-tier4/*.jsonl

Per run: requests, share of prompt tokens served from cache, hits by tier (and storage device),
TTFT mean / p50 / p90 / max, TTFT of the revisit requests, peak server RSS, and the cache's own
counters (WARM ratio, storage tier hits) from the last stats record.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path


def pct(xs: list[float], q: float) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def summarize(path: Path) -> dict:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    reqs = [r for r in rows if r.get("kind") in ("turn", "revisit", "revisit+1")]
    tiers: Counter = Counter()
    devices: Counter = Counter()
    prompt = cached = 0
    ttft, revisit = [], []
    for r in reqs:
        c = r.get("cache") or {}
        tier = c.get("tier") or ("ram" if r.get("cached") else "none")
        tiers[tier] += 1
        if c.get("device"):
            devices[c["device"]] += 1
        prompt += r.get("prompt") or 0
        cached += r.get("cached") or 0
        if r.get("ttft_ms") is not None:
            ttft.append(r["ttft_ms"] / 1000)
            if r["kind"] != "turn":
                revisit.append(r["ttft_ms"] / 1000)
    last = next((r["stats"] for r in reversed(rows) if r.get("stats")), None)
    caches = (last or {}).get("caches") or [{}]
    apc = caches[0].get("apc") or (last or {}).get("apc") or {}
    return {
        "run": path.stem,
        "n": len(reqs),
        "cached_share": round(cached / max(prompt, 1), 3),
        "tiers": dict(tiers),
        "devices": dict(devices),
        "ttft_mean": round(sum(ttft) / max(len(ttft), 1), 2),
        "ttft_p50": round(pct(ttft, 0.5), 2),
        "ttft_p90": round(pct(ttft, 0.9), 2),
        "ttft_max": round(max(ttft, default=0), 2),
        "revisit_mean": round(sum(revisit) / max(len(revisit), 1), 2),
        "rss_max": max((r.get("rss_gib") or 0 for r in rows), default=0),
        "warm": {k: v for k, v in apc.items() if k.startswith("warm_")},
        "storage": apc.get("storage_tiers"),
        "resident_gib": round((apc.get("resident_bytes") or 0) / 2**30, 2),
        "wall_s": max((r.get("t") or 0 for r in rows), default=0),
    }


def main() -> None:
    out = [summarize(Path(p)) for p in sys.argv[1:]]
    cols = [
        "run",
        "n",
        "cached_share",
        "tiers",
        "ttft_mean",
        "ttft_p50",
        "ttft_p90",
        "ttft_max",
        "revisit_mean",
        "rss_max",
    ]
    print(" | ".join(cols))
    for o in out:
        print(" | ".join(str(o[c]) for c in cols))
        if o["devices"]:
            print("    storage devices:", o["devices"])
        if o["warm"]:
            print("    warm:", o["warm"])
        if o["storage"]:
            for t in o["storage"]:
                print(
                    "    tier:",
                    {
                        k: (round(v, 1) if isinstance(v, float) else v)
                        for k, v in t.items()
                        if k
                        in (
                            "name",
                            "used_bytes",
                            "entries",
                            "hits",
                            "read_bps",
                            "demoted_in",
                            "invalidated",
                        )
                    },
                )


if __name__ == "__main__":
    main()
