"""Seeded alone/overlap promotion identity and token-arrival timeline (gpuq only)."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import shutil
import threading
import time
import urllib.request
from pathlib import Path

SAMPLING = (
    dict(temperature=0.7, top_p=1.0, top_k=0, seed=17),
    dict(temperature=1.0, top_p=0.95, top_k=20, seed=17),
)


def read_stream(lines, *, clock=time.monotonic, first=None):
    tokens, lps, tops, times = [], [], [], []
    usage = finish = None
    done = False
    for line in lines:
        line = line.strip()
        if not line.startswith(b"data:"):
            continue
        payload = line[5:].strip()
        if payload == b"[DONE]":
            done = True
            break
        event = json.loads(payload)
        if event.get("error"):
            raise RuntimeError(event["error"])
        usage = event.get("usage") or usage
        for choice in event.get("choices") or []:
            for item in (choice.get("logprobs") or {}).get("content") or []:
                tokens.append(item["token"])
                lps.append(item["logprob"])
                tops.append(item.get("top_logprobs") or [])
                times.append(clock())
                if len(times) == 1 and first is not None:
                    first.set()
            finish = choice.get("finish_reason") or finish
    if not (done and usage and finish and times):
        raise RuntimeError("incomplete measured stream")
    if len(tokens) != usage.get("completion_tokens"):
        raise RuntimeError("stream logprob count does not match completion usage")
    return dict(
        tokens=tokens, lps=lps, tops=tops, times=times, usage=usage, finish=finish
    )


def ask(server, prompt, budget, sampling, first=None):
    request = urllib.request.Request(
        server.url + "/v1/chat/completions",
        json.dumps(
            dict(
                model=server.model,
                messages=[{"role": "user", "content": prompt}]
                if isinstance(prompt, str)
                else prompt,
                max_tokens=budget,
                logprobs=True,
                top_logprobs=2,
                stream=True,
                stream_options={"include_usage": True},
                chat_template_kwargs={"enable_thinking": False},
                **sampling,
            )
        ).encode(),
        {"Content-Type": "application/json", "Authorization": "Bearer k"},
    )
    with urllib.request.urlopen(request, timeout=900) as response:
        result = read_stream(response, first=first)
    result["ended"] = time.monotonic()
    return result


def captured_ids(path, start, result):
    deadline = time.monotonic() + 5
    while True:
        rows = (
            [json.loads(line) for line in path.read_text().splitlines()[start:]]
            if path.exists()
            else []
        )
        matches = [
            row["token_ids"]
            for row in rows
            if row["prompt_tokens"] == result["usage"]["prompt_tokens"]
            and len(row["token_ids"]) == len(result["tokens"])
        ]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1 or time.monotonic() >= deadline:
            raise RuntimeError("missing or ambiguous fresh raw-ID capture")
        time.sleep(0.01)


def tail_tps(times, count=64):
    if len(times) <= count or times[-1] <= times[-count - 1]:
        raise RuntimeError("insufficient timed tail")
    return count / (times[-1] - times[-count - 1])


def compare(alone, overlap, blocker, log, *, tokens, min_tail_ratio=0.0):
    for result in (alone, overlap):
        cached = (result["usage"].get("prompt_tokens_details") or {}).get(
            "cached_tokens"
        )
        if cached != 0 or len(result["ids"]) != tokens or result["finish"] != "length":
            raise RuntimeError("invalid budget or cached measurement")
    for field in ("ids", "lps", "tops"):
        if alone[field] != overlap[field]:
            raise RuntimeError(f"alone/overlap {field} differ")
    boundary = blocker["ended"]
    if overlap["times"][0] >= boundary:
        raise RuntimeError("requests did not overlap")
    after = [stamp for stamp in overlap["times"] if stamp > boundary]
    if len(after) < 128:
        raise RuntimeError("survivor has fewer than 128 tokens after blocker ends")
    if "Promoted request onto the speculative lane" not in log:
        raise RuntimeError("missing actual promotion engagement")
    alone_tps, promoted_tps = tail_tps(alone["times"]), tail_tps(after)
    ratio = promoted_tps / alone_tps
    if ratio < min_tail_ratio:
        raise RuntimeError(f"promoted tail speed ratio {ratio:.3f} < {min_tail_ratio}")
    return dict(
        identity=True,
        blocker_end=boundary,
        survivor_first=overlap["times"][0],
        survivor_last=overlap["times"][-1],
        tokens_after_blocker=len(after),
        alone_tail_tps=alone_tps,
        promoted_tail_tps=promoted_tps,
        tail_speed_ratio=ratio,
        # Keep the observed timeline so restoration is reviewable, not inferred
        # solely from a server log or whole-request average.
        survivor_timeline=overlap["times"],
        raw_ids=overlap["ids"],
        logprobs=overlap["lps"],
        top_logprobs=overlap["tops"],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ctx", type=int, default=1024)
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--blocker-tokens", type=int, default=256)
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument("--min-tail-ratio", type=float, default=0.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.tokens < 256 or args.blocker_tokens < 64 or args.reps < 1:
        parser.error("requires survivor>=256, blocker>=64 and reps>=1")
    if not 0 <= args.min_tail_ratio <= 1:
        parser.error("--min-tail-ratio must be in [0, 1]")
    if args.dry_run:
        print(json.dumps(dict(complete=True, sampling=SAMPLING, reps=args.reps)))
        return
    import tfbench as t

    survivor = t.decode_prompt("code", args.ctx, True)
    blocker = [
        {
            "role": "user",
            "content": "Write a long detailed guide to implementing a compiler, including examples.",
        }
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    capture = args.out.with_suffix(".raw-ids.jsonl").resolve()
    if args.out.exists() or capture.exists():
        raise RuntimeError("refuse to reuse existing measurement/capture output")
    apc = Path("/Volumes/Micron/yunshu-apc") / f"sampledspec-promotion-{args.out.stem}"
    server = t.Srv(
        "yunshu",
        {
            "YUNSHU_VLM_DRAFT": t.D,
            "YUNSHU_VLM_APC_DISK": "0",
            "YUNSHU_VLM_APC_MEMORY_GB": "0",
            "YUNSHU_VLM_APC_DISK_DIR": str(apc),
            "YUNSHU_DEBUG_STREAM_CAPTURE": str(capture),
            "YUNSHU_NEXT_TURN_PREFILL": "off",
            "YUNSHU_CONSOLE": "0",
        },
        f"sampledspec-promotion-{args.out.stem}",
    )
    try:
        for rep in range(args.reps):
            for sampling in SAMPLING:
                start = len(capture.read_text().splitlines()) if capture.exists() else 0
                alone = ask(server, survivor, args.tokens, sampling)
                alone["ids"] = captured_ids(capture, start, alone)
                start = len(capture.read_text().splitlines())
                log_start = server.log.stat().st_size
                first = threading.Event()
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    task = pool.submit(
                        ask, server, blocker, args.blocker_tokens, SAMPLING[0], first
                    )
                    if not first.wait(900):
                        raise RuntimeError("blocker never started")
                    overlap = ask(server, survivor, args.tokens, sampling)
                    blocked = task.result()
                overlap["ids"] = captured_ids(capture, start, overlap)
                with server.log.open("rb") as log:
                    log.seek(log_start)
                    proof = log.read().decode(errors="replace")
                result = compare(
                    alone,
                    overlap,
                    blocked,
                    proof,
                    tokens=args.tokens,
                    min_tail_ratio=args.min_tail_ratio,
                )
                result.update(rep=rep, sampling=sampling)
                with args.out.open("a") as output:
                    output.write(json.dumps(result) + "\n")
                print(json.dumps(result), flush=True)
        server.verify_spec_mode()
        with args.out.open("a") as output:
            output.write(json.dumps(dict(complete=True, cells=2 * args.reps)) + "\n")
    finally:
        server.kill()
        shutil.rmtree(apc, ignore_errors=True)


if __name__ == "__main__":
    main()
