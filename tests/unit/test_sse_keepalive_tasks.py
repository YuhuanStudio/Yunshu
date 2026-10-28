"""with_sse_keepalive: one reader task per stream (not per chunk), same output, lazy pulls."""

import asyncio

import pytest

from yunshu_gateway.streaming import with_sse_keepalive


async def _collect(agen):
    return [x async for x in agen]


def _count_tasks(loop):
    created = []
    inner = loop.get_task_factory()

    def factory(lp, coro, **kw):
        created.append(coro)
        task = (
            asyncio.Task(coro, loop=lp, **kw)
            if inner is None
            else inner(lp, coro, **kw)
        )
        return task

    loop.set_task_factory(factory)
    return created


def test_no_task_per_chunk_and_same_output():
    async def gen():
        for i in range(500):
            yield f"c{i}"
            if i % 7 == 0:
                await asyncio.sleep(0)

    async def run():
        created = _count_tasks(asyncio.get_running_loop())
        out = await _collect(with_sse_keepalive(gen()))
        return out, len(created)

    out, tasks = asyncio.run(run())
    assert out == [": keep-alive\n\n"] + [f"c{i}" for i in range(500)]
    assert tasks <= 2, f"{tasks} tasks created for 500 chunks"


def test_generator_is_pulled_lazily():
    pulled = []

    async def gen():
        for i in range(3):
            pulled.append(i)
            yield f"c{i}"

    async def run():
        ait = with_sse_keepalive(gen()).__aiter__()
        await ait.__anext__()  # initial keep-alive
        assert pulled == []
        assert await ait.__anext__() == "c0"
        await asyncio.sleep(0.05)
        assert pulled == [0]  # not read ahead while the consumer holds c0
        await ait.aclose()

    asyncio.run(run())


def test_error_propagates_after_earlier_chunks():
    async def gen():
        yield "a"
        raise ValueError("boom")

    async def run():
        got = []
        with pytest.raises(ValueError, match="boom"):
            async for x in with_sse_keepalive(gen()):
                got.append(x)
        return got

    assert asyncio.run(run()) == [": keep-alive\n\n", "a"]


def test_keepalive_comments_while_waiting_and_close_cancels_producer():
    state = {}

    async def gen():
        try:
            await asyncio.sleep(10)
            yield "never"
        finally:
            state["closed"] = True

    async def run():
        ait = with_sse_keepalive(gen(), interval=0.02).__aiter__()
        assert await ait.__anext__() == ": keep-alive\n\n"
        assert await ait.__anext__() == ": keep-alive\n\n"  # emitted during the wait
        await ait.aclose()

    asyncio.run(run())
    assert state["closed"]
