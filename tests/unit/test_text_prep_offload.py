"""Chat-template rendering and prompt tokenization of the text fast paths run off the
event loop (a 30k-token prompt used to stall every other connection for tens of ms),
with the request's context variables, and produce the same token ids."""

import asyncio
import threading
import time

from .tiny_text_engine import FakeTokenizer, collect_stream, make_engine

MESSAGES = [
    {"role": "system", "content": "be brief. "},
    {"role": "user", "content": "the quick brown fox jumps"},
]


class _SlowTok(FakeTokenizer):
    def __init__(self):
        self.render_threads: list[str] = []
        self.encode_threads: list[str] = []
        self.tools_seen: list = []

    def apply_chat_template(self, msgs, **kw):
        from yunshu_engine import batched_engine

        self.render_threads.append(threading.current_thread().name)
        self.tools_seen.append(batched_engine._REQUEST_TOOLS.get())
        time.sleep(0.3)  # a long tool-heavy history
        return super().apply_chat_template(msgs, **kw)

    def encode(self, text, add_special_tokens=True, **kw):
        self.encode_threads.append(threading.current_thread().name)
        return super().encode(text, add_special_tokens=add_special_tokens, **kw)


async def _longest_loop_gap(coro) -> float:
    gaps: list[float] = []
    stop = False

    async def ticker():
        last = time.perf_counter()
        while not stop:
            await asyncio.sleep(0.005)
            now = time.perf_counter()
            gaps.append(now - last)
            last = now

    t = asyncio.create_task(ticker())
    await asyncio.sleep(0.02)
    try:
        await coro
    finally:
        stop = True
        await t
    return max(gaps)


def _engine():
    eng = make_engine()
    eng._tokenizer = _SlowTok()
    return eng


async def test_generate_fast_keeps_the_event_loop_free_while_templating():
    from yunshu_engine import batched_engine

    await make_engine()._generate_fast(prompt="warm", max_tokens=1, temperature=0.0)
    eng = _engine()
    batched_engine._REQUEST_TOOLS.set([{"type": "function", "function": {"name": "f"}}])
    out = {}

    async def run():
        out["o"] = await eng._generate_fast(
            prompt=MESSAGES, max_tokens=3, temperature=0.0
        )

    gap = await _longest_loop_gap(run())
    assert out["o"].error is None and out["o"].completion_tokens == 3
    assert gap < 0.15, f"event loop stalled {gap * 1e3:.0f} ms"
    tok = eng._tokenizer
    assert tok.render_threads == ["yunshu-textprep_0"] == tok.encode_threads[:1]
    assert tok.tools_seen == [[{"type": "function", "function": {"name": "f"}}]]
    expected = FakeTokenizer().encode("be brief. the quick brown fox jumps")
    assert out["o"].prompt_tokens == len(expected)


async def test_stream_generate_fast_keeps_the_event_loop_free_while_templating():
    eng = _engine()
    out = {}

    async def run():
        out["r"] = await collect_stream(
            eng, prompt=MESSAGES, max_tokens=3, temperature=0.0
        )

    gap = await _longest_loop_gap(run())
    text, last = out["r"]
    assert last.error is None and last.completion_tokens == 3
    assert gap < 0.15, f"event loop stalled {gap * 1e3:.0f} ms"
    assert eng._tokenizer.render_threads == ["yunshu-textprep_0"]


async def test_templated_and_plain_prompts_generate_the_same_tokens():
    plain = await make_engine()._generate_fast(
        prompt="be brief. the quick brown fox jumps", max_tokens=6, temperature=0.0
    )
    templated = await make_engine()._generate_fast(
        prompt=MESSAGES, max_tokens=6, temperature=0.0
    )
    assert (
        plain.text == templated.text and plain.prompt_tokens == templated.prompt_tokens
    )
