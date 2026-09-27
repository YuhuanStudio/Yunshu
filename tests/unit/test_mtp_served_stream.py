"""The served MTP stream must emit before its decoder has finished."""

import asyncio
import threading

from yunshu_engine.batched_engine import BatchedEngine


class _Detokenizer:
    def reset(self):
        self._segment = ""

    def add_token(self, token):
        self._segment = chr(token)

    @property
    def last_segment(self):
        segment, self._segment = self._segment, ""
        return segment

    def finalize(self):
        pass


class _Tokenizer:
    @property
    def detokenizer(self):
        return _Detokenizer()

    def encode(self, text, add_special_tokens=False):
        return list(text.encode())


class _Backend:
    def __init__(self, resume):
        self.tokenizer = _Tokenizer()
        self.resume = resume
        self.finished = threading.Event()

    def iter_token_ids(self, messages, max_tokens, temperature, prompt):
        try:
            yield ord("A")
            self.resume.wait(timeout=5)
            yield ord("B")
            yield ord("C")
            yield ord("D")
        finally:
            self.finished.set()


def _engine(backend):
    engine = BatchedEngine()
    engine._mlxvlm_mtp = backend
    engine._apply_chat_template = lambda *_args, **_kwargs: "prompt"
    engine._mtp_prompt_tokens = lambda *_args: 6
    return engine


def test_mtp_stream_emits_before_decode_finishes_and_hides_stop():
    async def run():
        resume = threading.Event()
        backend = _Backend(resume)
        stream = _engine(backend).stream_chat(
            [{"role": "user", "content": "hi"}],
            stop=["BC"],
            max_tokens=8,
        )
        first = await asyncio.wait_for(anext(stream), timeout=3)
        assert first.new_text == "A" and not first.finished
        assert not backend.finished.is_set()
        resume.set()
        rest = [item async for item in stream]
        assert rest[-1].text == "A"
        assert rest[-1].finish_reason == "stop"
        assert rest[-1].stopped_by_stop_sequence
        assert rest[-1].completion_tokens == 1
        assert backend.finished.is_set()

    asyncio.run(run())


def test_mtp_stream_close_drains_worker():
    async def run():
        resume = threading.Event()
        backend = _Backend(resume)
        stream = _engine(backend).stream_chat(
            [{"role": "user", "content": "hi"}], max_tokens=8
        )
        assert (await asyncio.wait_for(anext(stream), timeout=3)).new_text == "A"
        resume.set()
        await asyncio.wait_for(stream.aclose(), timeout=3)
        assert backend.finished.is_set()

    asyncio.run(run())


def test_mtp_nonstream_uses_earliest_stop_and_visible_usage():
    async def run():
        backend = _Backend(threading.Event())
        backend.generate = lambda *_args, **_kwargs: {
            "text": "AxxB",
            "completion_tokens": 4,
        }
        result = await _engine(backend).chat(
            [{"role": "user", "content": "hi"}],
            stop=["B", "xx"],
            max_tokens=4,
        )
        assert result.text == "A"
        assert result.completion_tokens == 1
        assert result.finish_reason == "stop"
        assert result.stopped_by_stop_sequence

    asyncio.run(run())
