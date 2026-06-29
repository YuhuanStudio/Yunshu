"""the in-flight tracking mixin + decorators that let the media engines
(Image/Video/TTS/ASR/STS/OCR) report has_active_requests() so the ModelManager can
safely LRU-evict them when idle. The decrement MUST be leak-proof on every exit
path (success, exception, generator early-break) — a leaked counter would make a
model permanently un-evictable."""

from __future__ import annotations

import asyncio

import pytest

from yunshu_engine.active_tracking import (
    ActiveRequestMixin,
    tracks_active,
    tracks_active_gen,
)


class _Eng(ActiveRequestMixin):
    @tracks_active
    async def do(self, fail=False):
        assert self.has_active_requests() is True  # active DURING the call
        await asyncio.sleep(0)
        if fail:
            raise ValueError("boom")
        return "ok"

    @tracks_active_gen
    async def stream(self, n, fail_at=None):
        for i in range(n):
            assert self.has_active_requests() is True
            if fail_at is not None and i == fail_at:
                raise RuntimeError("mid-stream")
            yield i


def test_idle_by_default():
    assert _Eng().has_active_requests() is False


def test_coroutine_success_decrements():
    e = _Eng()
    assert asyncio.run(e.do()) == "ok"
    assert e.has_active_requests() is False


def test_coroutine_exception_decrements():
    e = _Eng()
    with pytest.raises(ValueError):
        asyncio.run(e.do(fail=True))
    assert e.has_active_requests() is False  # finally ran despite the raise


def test_generator_full_consume_decrements():
    e = _Eng()

    async def _run():
        out = [x async for x in e.stream(3)]
        return out

    assert asyncio.run(_run()) == [0, 1, 2]
    assert e.has_active_requests() is False


def test_generator_exception_decrements():
    e = _Eng()

    async def _run():
        async for _ in e.stream(5, fail_at=2):
            pass

    with pytest.raises(RuntimeError):
        asyncio.run(_run())
    assert e.has_active_requests() is False


def test_generator_early_break_decrements():
    e = _Eng()

    async def _run():
        gen = e.stream(100)
        got = []
        async for x in gen:
            got.append(x)
            if x == 1:
                break
        await gen.aclose()  # explicit close → finally runs deterministically
        return got

    assert asyncio.run(_run()) == [0, 1]
    assert e.has_active_requests() is False


def test_concurrent_requests_counted():
    e = _Eng()

    async def _run():
        async def _slow():
            await e.do()

        # two overlapping calls → count should reach 2 then return to 0
        t1 = asyncio.create_task(e.do())
        t2 = asyncio.create_task(e.do())
        await asyncio.gather(t1, t2)

    asyncio.run(_run())
    assert e.has_active_requests() is False
