"""Text path usage: a stop string that spans tokens leaves held-back stop-prefix tokens whose text
was never delivered; the streamed completion_tokens counts what the consumer took, exactly as the
non-streamed count (which re-encodes the kept text) does."""

from __future__ import annotations

from .tiny_text_engine import make_engine

PROMPT = "the quick brown fox jumps over the lazy dog"


async def test_stop_string_stream_usage_equals_non_stream():
    eng = make_engine()
    base = await eng._generate_fast(
        prompt=PROMPT, max_tokens=12, temperature=1.0, seed=7
    )
    stop = [base.text[4:6]]
    plain = await eng._generate_fast(
        prompt=PROMPT, max_tokens=12, temperature=1.0, seed=7, stop=stop
    )
    last, first_ttft = None, None
    async for o in eng._stream_generate_fast(
        prompt=PROMPT, max_tokens=12, temperature=1.0, seed=7, stop=stop
    ):
        last = o
        if first_ttft is None and o.ttft_ms > 0:
            first_ttft = o.ttft_ms
    assert plain.finish_reason == "stop" and last.finish_reason == "stop"
    assert last.completion_tokens == plain.completion_tokens
    assert first_ttft is not None
