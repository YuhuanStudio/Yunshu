"""A client ``timeout`` is a deadline on the whole call on the text fast path, streaming or not: a reply
that keeps producing a token inside every gap still ends when the total time is up (pinned for the
unified path; the VLM path restored this in 0799979)."""

from __future__ import annotations

import time

from .tiny_text_engine import make_engine

PROMPT = "the quick brown fox"


class SlowSteady:
    """A token every ~30 ms: never an idle gap longer than the timeout."""

    def __call__(self, tokens, logits):
        time.sleep(0.03)
        return logits


async def test_non_stream_timeout_is_a_total_deadline():
    eng = make_engine()
    await eng._generate_fast(prompt=PROMPT, max_tokens=2, temperature=0.0)  # warm up
    t0 = time.perf_counter()
    out = await eng._generate_fast(
        prompt=PROMPT,
        max_tokens=200,
        temperature=0.0,
        timeout_seconds=0.2,
        logits_processors=[SlowSteady()],
    )
    assert time.perf_counter() - t0 < 0.6
    assert out.finish_reason in ("timeout", "length")  # both are "length" on the wire
    assert out.completion_tokens < 200


async def test_stream_timeout_is_a_total_deadline():
    eng = make_engine()
    await eng._generate_fast(prompt=PROMPT, max_tokens=2, temperature=0.0)  # warm up
    t0 = time.perf_counter()
    last = None
    async for o in eng._stream_generate_fast(
        prompt=PROMPT,
        max_tokens=200,
        temperature=0.0,
        timeout_seconds=0.2,
        logits_processors=[SlowSteady()],
    ):
        last = o
    assert time.perf_counter() - t0 < 0.6
    assert last.finished and last.finish_reason == "timeout"
    assert last.completion_tokens < 200
