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


def test_delivered_count_comes_from_the_token_stream_not_a_re_encode():
    from yunshu_engine.engine_policy import _delivered_token_count as count

    ends = [2, 4, 6, 8]  # "do" "ne" "ST" "OP"; the stop "STOP" is cut, "done" delivered
    assert count(ends, 4, 4) == 2
    assert count(ends, 3, 4) == 2  # the cut falls inside "ne": it was partly delivered
    assert count(ends, 0, 4) == 0
    assert count(ends, 8, 4) == 4
    assert (
        count([2, 2, 5], 2, 3) == 2
    )  # a zero-width token (partial byte) before the cut
    assert count([2], 2, 3) == 3  # incomplete token log: bill what was generated


async def test_stop_string_usage_does_not_re_tokenize_the_reply():
    eng = make_engine()
    base = await eng._generate_fast(
        prompt=PROMPT, max_tokens=12, temperature=1.0, seed=7
    )
    stop = [base.text[4:6]]
    seen = []
    real = eng._tokenizer.encode

    def spy(text, *a, **kw):
        seen.append(text)
        return real(text, *a, **kw)

    eng._tokenizer.encode = spy
    last = None
    delivered = ""
    async for o in eng._stream_generate_fast(
        prompt=PROMPT, max_tokens=12, temperature=1.0, seed=7, stop=stop
    ):
        last = o
        delivered += o.new_text or ""
    assert last.finish_reason == "stop"
    assert (
        delivered and delivered not in seen
    )  # was: encode(delivered) on the event loop
    assert last.completion_tokens == len(
        delivered
    )  # one token per char in the tiny vocab
