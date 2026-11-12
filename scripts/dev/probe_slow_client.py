"""Slow-client coupling probe: does one slow SSE consumer slow a normal stream on the same server?

Each rep runs two scenarios, interleaved: "both_normal" (two normal streams) and "one_slow"
(one stream sleeps between tokens, the other is normal). Per stream: tokens, wall time and
tokens/s, measured on the consumer. The normal stream's tok/s in "one_slow" is compared with
its tok/s in "both_normal". Prints one JSON line per scenario and a summary at the end.
"""

import argparse
import asyncio
import json
import statistics
import time


async def read_stream(engine, prompt, max_tokens, sleep_s):
    tokens = 0
    t0 = time.monotonic()
    first = None
    async for out in engine.generate_stream(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.0,
    ):
        if out.new_token_ids:
            tokens += len(out.new_token_ids)
            first = first if first is not None else time.monotonic() - t0
        if sleep_s:
            await asyncio.sleep(sleep_s)
        if out.finished:
            break
    wall = time.monotonic() - t0
    return {
        "tokens": tokens,
        "wall_s": round(wall, 3),
        "first_s": round(first, 3) if first is not None else None,
        "tok_s": round(tokens / wall, 2) if wall else None,
    }


async def scenario(engine, label, specs, max_tokens):
    t0 = time.monotonic()
    results = await asyncio.gather(
        *(
            read_stream(
                engine, f"Write a short story number {i} about a river.", max_tokens, s
            )
            for i, s in enumerate(specs)
        )
    )
    row = {
        "label": label,
        "streams": results,
        "scenario_wall_s": round(time.monotonic() - t0, 3),
    }
    print(json.dumps(row), flush=True)
    return row


async def main_async(args):
    from yunshu_engine.vlm_engine import VLMEngine

    engine = VLMEngine(args.model)
    await engine.start()
    await scenario(engine, "warmup", [0.0], 8)
    rows = []
    for rep in range(args.reps):
        rows.append(
            await scenario(engine, f"both_normal.r{rep}", [0.0, 0.0], args.tokens)
        )
        rows.append(
            await scenario(engine, f"one_slow.r{rep}", [args.sleep, 0.0], args.tokens)
        )
    normal_base = [
        r["streams"][0]["tok_s"] for r in rows if r["label"].startswith("both")
    ]
    normal_slow = [
        r["streams"][1]["tok_s"] for r in rows if r["label"].startswith("one_slow")
    ]
    summary = {
        "reps": args.reps,
        "normal_tok_s_both_normal": normal_base,
        "normal_tok_s_with_slow_peer": normal_slow,
        "median_both_normal": statistics.median(normal_base),
        "median_with_slow_peer": statistics.median(normal_slow),
    }
    summary["ratio_slow_peer_over_both_normal"] = round(
        summary["median_with_slow_peer"] / summary["median_both_normal"], 3
    )
    print("SUMMARY", json.dumps(summary), flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokens", type=int, default=600)
    ap.add_argument("--sleep", type=float, default=0.05)
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()
    raise SystemExit(asyncio.run(main_async(args)))


if __name__ == "__main__":
    main()
