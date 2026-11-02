"""A realtime response's engine calls get child cancel events: the LLM stream ending (orphan adoption
sets the caller's event when the consumer stops reading) must not cancel the TTS that follows, while
cancelling the response reaches every engine call (0.1.6 release sweep: realtime voice, 0 bytes of
speech; the TTS stopped 1 ms after it started)."""

import asyncio
import inspect

from yunshu_gateway.routers import realtime as rt


def test_child_set_does_not_cancel_the_response():
    async def main():
        resp = rt._ResponseCancel()
        llm = resp.child()
        llm.set()  # the LLM stream's own teardown
        assert not resp.is_set()
        tts = resp.child()
        assert not tts.is_set()
        resp.set()  # barge-in / response.cancel
        assert tts.is_set() and llm.is_set()
        await asyncio.wait_for(tts.wait(), 1)  # engines that await the event wake up
        assert resp.child().is_set()  # a call started after the cancel starts cancelled

    asyncio.run(main())


def test_engine_calls_take_a_child_of_the_response_cancel():
    src = inspect.getsource(rt)
    assert "cancel_event=self._cancel_event," not in src
    assert src.count("cancel_event=self._engine_cancel()") >= 2
    assert src.count("self._cancel_event = _ResponseCancel()") == 2
