"""Decode rate of a running request while another request prefills, vs prefill chunk size.

Hardware background (docs/guides/M5MAX_HARDWARE.md): a second MTLCommandQueue gives decode
no priority and the GPU does not run prefill and decode side by side, so a decode step waits
for the prefill chunk queued ahead of it. In the round driver that chunk is CHUNK tokens
(DECODE_BUDGET while rows decode). This drives the real 27B round driver in-process: one row
decodes, a long prompt arrives, and the row's token rate over the prompt's prefill window and
the prompt's TTFT are recorded per chunk size.

    PYTHONPATH=python:scripts/research/hw python scripts/research/hw/bench_prefill_interleave.py \
        $M --chunks 512 256 128 64 --prompt 8192 --tokens 400
"""

import argparse
import sys
import time
from pathlib import Path

import mlx.core as mx
from _common import Out

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sweep_round_driver import PROMPTS, encode, load  # noqa: E402


def one(model, drafter, stop, a_ids, b_ids, tokens, drafts, chunk=None):
    from yunshu_engine.round_driver.driver import Request, RoundDriver

    d = RoundDriver(
        model, drafter=drafter if drafts else None, stop_tokens=stop, chunk=chunk
    )
    d.add(Request(a_ids, tokens, handle="a"))
    a_times, a_tokens, b_first, b_added = [], [], None, None
    b_tokens = []
    t0 = time.perf_counter()
    while d.busy():
        if b_added is None and len(a_tokens) >= 30:
            d.add(Request(b_ids, 16, handle="b"))
            b_added = time.perf_counter()
        for e in d.step():
            now = time.perf_counter()
            if e.handle == "a" and e.token is not None:
                a_times.append(now)
                a_tokens.append(e.token)
            elif e.handle == "b":
                b_tokens.append(e.token)
                if b_first is None:
                    b_first = now
        if b_first is not None and len(b_tokens) >= 16 and a_times[-1] > b_first + 1.0:
            break
    return d, t0, a_times, a_tokens, b_added, b_first, b_tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--chunks", type=int, nargs="*", default=[512, 256, 128, 64])
    ap.add_argument("--prompt", type=int, default=8192)
    ap.add_argument("--tokens", type=int, default=600)
    ap.add_argument("--no-drafts", action="store_true")
    ap.add_argument("--quantize", action="store_true")
    a = ap.parse_args()
    out = Out("bench_prefill_interleave")
    model, processor, drafter, _ = load(a.model, a.quantize)
    tok = processor.tokenizer
    extra = getattr(tok, "eos_token_ids", None) or []
    stop = {tok.eos_token_id} | set([extra] if isinstance(extra, int) else extra)
    a_ids = encode(tok, PROMPTS[0], 0)
    b_ids = encode(tok, PROMPTS[3], a.prompt)
    out(kind="meta", a_prompt=len(a_ids), b_prompt=len(b_ids), drafts=not a.no_drafts)

    # warm every kernel pipeline once
    one(model, drafter, stop, a_ids, encode(tok, PROMPTS[3], 1024), 60, not a.no_drafts)
    mx.synchronize()
    ref_b = None
    for chunk in a.chunks:
        # prompt alone (same chunk), for the prefill cost of sharing the GPU
        from yunshu_engine.round_driver.driver import Request, RoundDriver

        solo = RoundDriver(model, drafter=None, stop_tokens=stop, chunk=chunk)
        solo.add(Request(b_ids, 16, handle="b"))
        ts = time.perf_counter()
        solo_first = None
        solo_tokens = []
        while solo.busy():
            for e in solo.step():
                solo_tokens.append(e.token)
                solo_first = solo_first or time.perf_counter() - ts
        d, t0, at, atok, b_added, b_first, btok = one(
            model, drafter, stop, a_ids, b_ids, a.tokens, not a.no_drafts, chunk
        )
        before = [t for t in at if t < b_added]
        during = [t for t in at if b_added <= t <= b_first]
        r_before = (
            (len(before) - 1) / (before[-1] - before[0]) if len(before) > 2 else None
        )
        r_during = len(during) / (b_first - b_added)
        gaps = sorted(
            y - x
            for x, y in zip(at[:-1], at[1:], strict=True)
            if x >= b_added and y <= b_first + 0.5
        )
        out(
            kind="interleave",
            chunk=chunk,
            prompt=len(b_ids),
            decode_tok_s_before=round(r_before, 1) if r_before else None,
            decode_tok_s_during_prefill=round(r_during, 2),
            decode_gap_ms_median=round(gaps[len(gaps) // 2] * 1e3, 1) if gaps else None,
            decode_gap_ms_max=round(gaps[-1] * 1e3, 1) if gaps else None,
            prefill_ttft_s=round(b_first - b_added, 2),
            prefill_tok_s=round(len(b_ids) / (b_first - b_added), 0),
            prompt_alone_ttft_s=round(solo_first, 2),
            prompt_alone_tok_s=round(len(b_ids) / solo_first, 0),
            b_tokens_equal_alone=btok == solo_tokens,
            b_tokens_equal_chunk_first=(btok == ref_b) if ref_b else None,
            drafted=d.drafted,
            accepted=d.accepted,
        )
        ref_b = ref_b or btok
    _ = ref_b


if __name__ == "__main__":
    main()
