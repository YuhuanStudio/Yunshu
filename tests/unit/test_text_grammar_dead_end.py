"""A grammar dead end fails only its own request with a clean error (stream and non-stream) and
frees the engine: the next request is served and nothing is left counted as active."""

from __future__ import annotations

from yunshu_engine.constraint_eos import ConstrainedDecodingError

from .tiny_text_engine import make_engine

PROMPT = "the quick brown fox"


class DeadEnd:
    """A logits processor whose grammar dies on the 3rd step."""

    def __init__(self):
        self.calls = 0

    def __call__(self, tokens, logits):
        self.calls += 1
        if self.calls >= 3:
            raise ConstrainedDecodingError("constrained decoding dead end")
        return logits


async def test_non_stream_dead_end_is_a_clean_error_and_the_engine_is_free():
    eng = make_engine()
    out = await eng._generate_fast(
        prompt=PROMPT, max_tokens=8, temperature=0.0, logits_processors=[DeadEnd()]
    )
    assert out.finish_reason == "error" and "dead end" in (out.error or "")
    assert eng._active_fast_path_count == 0
    ok = await eng._generate_fast(prompt=PROMPT, max_tokens=4, temperature=0.0)
    assert ok.error is None and ok.completion_tokens == 4


async def test_stream_dead_end_is_a_terminal_error_chunk_and_the_engine_is_free():
    eng = make_engine()
    outs = []
    async for o in eng._stream_generate_fast(
        prompt=PROMPT, max_tokens=8, temperature=0.0, logits_processors=[DeadEnd()]
    ):
        outs.append(o)
    assert outs[-1].finished and outs[-1].finish_reason == "error"
    assert "dead end" in (outs[-1].error or "")
    assert eng._active_fast_path_count == 0
    ok = await eng._generate_fast(prompt=PROMPT, max_tokens=4, temperature=0.0)
    assert ok.error is None and ok.completion_tokens == 4
