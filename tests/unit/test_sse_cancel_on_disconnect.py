"""a client disconnect mid-stream reaches with_sse_keepalive as GeneratorExit/
CancelledError thrown INTO the generator (both BaseException, bypassing the is_disconnected
poll AND the routers' `except` branches that set cancel_event). The hub's finally must set
cancel_event on ANY exit so the engine's decode loop (running on the serialized MLX executor,
where task.cancel() is a no-op) actually stops — else generation runs to max_tokens, wasting
GPU and head-of-line-blocking every subsequent request."""
from __future__ import annotations

import asyncio
import contextlib
import threading

from yunshu_gateway.streaming import with_sse_keepalive


def test_cancel_event_set_on_consumer_aclose():
    """Consumer stops early and aclose()s the wrapper (the disconnect path)."""
    cancel = threading.Event()

    async def _slow():
        for i in range(1000):
            await asyncio.sleep(0.001)
            yield f"data: {i}\n\n"

    async def _run():
        wrapped = with_sse_keepalive(_slow(), cancel_event=cancel)
        got = 0
        async for _ in wrapped:
            got += 1
            if got == 2:
                break
        await wrapped.aclose()  # simulate starlette tearing down the stream on disconnect

    asyncio.run(_run())
    assert cancel.is_set(), "cancel_event must be set on stream teardown so the engine stops"


def test_cancel_event_set_on_normal_completion():
    """Harmless on normal completion (the engine loop already ended)."""
    cancel = threading.Event()

    async def _short():
        yield "data: a\n\n"
        yield "data: b\n\n"

    async def _run():
        out = [c async for c in with_sse_keepalive(_short(), cancel_event=cancel)]
        return out

    out = asyncio.run(_run())
    # with_sse_keepalive may emit leading ": keep-alive" comments — ignore them.
    data = [c for c in out if not c.startswith(":")]
    assert data == ["data: a\n\n", "data: b\n\n"]
    assert cancel.is_set()  # set in finally; harmless (loop already done)


def test_cancel_event_set_on_exception():
    cancel = threading.Event()

    async def _boom():
        yield "data: a\n\n"
        raise RuntimeError("mid-stream")

    async def _run():
        async for _ in with_sse_keepalive(_boom(), cancel_event=cancel):
            pass

    with contextlib.suppress(RuntimeError):
        asyncio.run(_run())
    assert cancel.is_set()


def test_no_cancel_event_does_not_crash():
    async def _short():
        yield "x"

    async def _run():
        return [c async for c in with_sse_keepalive(_short(), cancel_event=None)]

    data = [c for c in asyncio.run(_run()) if not c.startswith(":")]
    assert data == ["x"]
