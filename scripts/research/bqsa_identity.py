"""Check each concurrent Qwen4 request against its solo token/logprob digest."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import threading
import time
from pathlib import Path


def plans(scenario, tokens):
    if scenario == "stagger":
        return [("A", tokens, 101), ("B", max(16, tokens // 3), 102)]
    n = {"burst2": 2, "burst4": 4}[scenario]
    return [(chr(65 + i), tokens, 101 + i) for i in range(n)]


def compare(solo, concurrent):
    if solo.keys() != concurrent.keys() or any(
        (solo[name]["digest"], solo[name]["finish"])
        != (concurrent[name]["digest"], concurrent[name]["finish"])
        for name in solo
    ):
        raise RuntimeError(
            f"concurrent token/logprob digest differs from solo: {solo!r} vs {concurrent!r}"
        )


def compare_modes(off_rows, mtp_rows):
    if (
        not off_rows
        or off_rows[-1].get("complete") is not True
        or off_rows[-1].get("draft") != "off"
    ):
        raise RuntimeError("missing completed spec-off evidence")

    def cells(rows):
        return {(r["scenario"], r["shape"]): r for r in rows if "scenario" in r}

    off, mtp = cells(off_rows), cells(mtp_rows)
    expected = {
        (scenario, shape)
        for scenario in ("burst2", "burst4", "stagger")
        for shape in ("greedy", "seeded")
    }
    if set(off) != expected or set(mtp) != expected:
        raise RuntimeError("missing spec-mode comparison cells")
    for key in expected:
        compare(off[key]["solo"], mtp[key]["solo"])
        compare(off[key]["concurrent"], mtp[key]["concurrent"])


def stream(url, body, started=None):
    import httpx
    import sampledspec_conc as sampled

    text, logprobs, times = [], [], []
    usage = finish = None
    with httpx.Client(timeout=1800) as client:
        with client.stream("POST", url + "/v1/chat/completions", json=body) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                event = json.loads(line[6:])
                usage = event.get("usage") or usage
                for choice in event.get("choices") or []:
                    delta = choice.get("delta") or {}
                    piece = (
                        delta.get("content")
                        or delta.get("reasoning_content")
                        or delta.get("reasoning")
                    )
                    if piece:
                        if started is not None:
                            started.set()
                        text.append(piece)
                    for item in (choice.get("logprobs") or {}).get("content") or []:
                        logprobs.append([item["token"], item["logprob"]])
                        times.append(time.monotonic())
                    finish = choice.get("finish_reason") or finish
    if not usage or not finish or not logprobs:
        raise RuntimeError("incomplete response or missing logprobs")
    return {
        "digest": sampled.digest("".join(text), logprobs),
        "logprobs": logprobs,
        "first": times[0],
        "last": times[-1],
        "finish": finish,
        "cached": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
        "prompt_tokens": usage.get("prompt_tokens"),
    }


def run_cell(server, scenario, shape, tokens, ctx_tokens, tok, join):
    import conc_probe
    import prompt_len
    import sampledspec_conc as sampled

    jobs = plans(scenario, tokens)

    def request(salt, count):
        prompt = prompt_len.build_text(
            tok,
            conc_probe.filler(ctx_tokens * 6, salt),
            ctx_tokens,
            suffix="\nReview this design and propose concrete improvements.",
        )
        body = sampled.body(server.model, prompt, count, "seeded")
        body.update(logprobs=True, seed=17 + salt)
        if shape == "greedy":
            body.update(temperature=0.0)
        return body

    solo = {
        name: stream(server.url, request(salt, count)) for name, count, salt in jobs
    }
    for row in solo.values():  # served turn-1 length of each solo request
        prompt_len.check_served_length(
            ctx_tokens, {"prompt_tokens": row["prompt_tokens"]}
        )
    started, finished = threading.Event(), threading.Event()

    def run(job):
        name, count, salt = job
        if scenario == "stagger" and name == "B":
            if not started.wait(1800):
                raise RuntimeError("first request did not start")
            time.sleep(join)
            if finished.is_set():
                raise RuntimeError("first request finished before staggered join")
        try:
            return name, stream(
                server.url, request(salt, count), started if name == "A" else None
            )
        finally:
            if name == "A":
                finished.set()
                started.set()

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        concurrent_rows = dict(pool.map(run, jobs))
    compare(solo, concurrent_rows)
    if max(row["first"] for row in concurrent_rows.values()) >= min(
        row["last"] for row in concurrent_rows.values()
    ):
        raise RuntimeError("rows did not decode simultaneously")
    for row in concurrent_rows.values():
        cached, prompt = row["cached"], row["prompt_tokens"]
        if (
            not isinstance(cached, int)
            or not isinstance(prompt, int)
            or cached < prompt - 128
        ):
            raise RuntimeError("concurrent row did not hit its warm prefix")
    return dict(
        scenario=scenario,
        shape=shape,
        solo=solo,
        concurrent=concurrent_rows,
        identity=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--draft", choices=("off", "mtp"), required=True)
    parser.add_argument(
        "--ctx-tokens",
        type=int,
        default=None,
        help="served prompt tokens per request (template included; default: the legacy 16000 characters, converted)",
    )
    parser.add_argument(
        "--ctx-chars",
        type=int,
        default=None,
        help="DEPRECATED alias: characters of filler, converted to served tokens",
    )
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--join", type=float, default=0.5)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--compare-off", type=Path)
    a = parser.parse_args()
    if a.tokens < 48 or a.join < 0:
        parser.error("invalid request shape")
    if a.ctx_tokens is not None and a.ctx_chars is not None:
        parser.error("--ctx-chars is a deprecated alias of --ctx-tokens; give one")
    import prompt_len
    import tfbench

    tok = prompt_len.load_tokenizer(a.model)
    if a.ctx_chars is not None or a.ctx_tokens is None:
        chars = 16000 if a.ctx_chars is None else a.ctx_chars
        ctx_tokens = prompt_len.tokens_from_chars(tok, chars)
        print(
            f"warning: --ctx-chars {chars} is deprecated; using {ctx_tokens} served tokens",
            flush=True,
        )
    else:
        ctx_tokens = a.ctx_tokens
    if ctx_tokens < 16:
        parser.error("invalid request shape")

    tfbench.YUNSHU_SRC = str(a.src.resolve())
    tfbench.OUT = a.out.parent / (a.out.stem + "-servers")
    env = {
        "YUNSHU_VLM_DRAFT": a.draft,
        "YUNSHU_VLM_APC_DISK": "0",
        "YUNSHU_NEXT_TURN_PREFILL": "off",
        "YUNSHU_CONSOLE": "0",
    }
    server = tfbench.Srv("yunshu", env, f"bqsa-id-{a.out.stem}", model=a.model)
    try:
        with a.out.open("x") as out:
            rows = []
            for scenario in ("burst2", "burst4", "stagger"):
                for shape in ("greedy", "seeded"):
                    row = run_cell(
                        server, scenario, shape, a.tokens, ctx_tokens, tok, a.join
                    )
                    rows.append(row)
                    out.write(json.dumps(row) + "\n")
                    out.flush()
            if a.draft == "mtp":
                server.verify_spec_mode()
            if a.compare_off:
                off_rows = [
                    json.loads(line) for line in a.compare_off.read_text().splitlines()
                ]
                compare_modes(off_rows, rows)
            out.write(json.dumps(dict(complete=True, draft=a.draft)) + "\n")
    finally:
        server.kill()


if __name__ == "__main__":
    main()
