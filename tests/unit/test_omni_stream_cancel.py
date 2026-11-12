"""Omni speech: setting the cancel event stops the Thinker/Talker loop before its next step."""

import asyncio
import threading

from yunshu_engine.omni_engine import OmniEngine


def _engine(steps: int, cancel_after: int, cancel: threading.Event) -> tuple:
    eng = OmniEngine.__new__(OmniEngine)
    eng._shared = False
    eng._prev_text_ids = []
    ran = []

    def make_gen(conv, spk, tmax):
        for i in range(steps):
            ran.append(i)
            if i + 1 == cancel_after:
                cancel.set()
            yield ("text", [i])

    eng._make_gen = make_gen
    eng._decode_fragment = lambda payload: "x"
    return eng, ran


def test_cancel_stops_inline_generation():
    cancel = threading.Event()
    eng, ran = _engine(steps=50, cancel_after=3, cancel=cancel)

    async def run():
        return [x async for x in eng._iter_materialized([], "s", 10, cancel)]

    out = asyncio.run(run())
    assert len(ran) == 3 and len(out) == 2
