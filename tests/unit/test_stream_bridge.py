"""B12: executor->loop stream bridge must not lose items or terminals."""

import asyncio
import threading

import pytest

from yunshu_engine.stream_bridge import StreamBridge, make_stream_queue

_END = object()


def _term(x):
    return x is _END or isinstance(x, BaseException)


@pytest.mark.asyncio
async def test_burst_into_tiny_queue_no_loss_no_callback_errors():
    loop = asyncio.get_running_loop()
    errors = []
    loop.set_exception_handler(lambda l, c: errors.append(c))
    q = make_stream_queue(2)
    br = StreamBridge(loop, q, _term, wait_s=2.0)
    got = []

    def producer():
        for i in range(50):
            assert br.put(i)
        br.put(_END)

    t = threading.Thread(target=producer)
    t.start()
    while True:
        item = await q.get()
        if item is _END:
            break
        got.append(item)
        await asyncio.sleep(0)
    t.join()
    assert got == list(range(50))
    await asyncio.sleep(0.01)
    assert errors == []


@pytest.mark.asyncio
async def test_stalled_consumer_gets_error_then_end_never_lost():
    loop = asyncio.get_running_loop()
    errors = []
    loop.set_exception_handler(lambda l, c: errors.append(c))
    q = make_stream_queue(2)
    cancelled = threading.Event()
    br = StreamBridge(loop, q, _term, on_overflow=cancelled.set, wait_s=0.01)
    res = []

    def producer():
        for i in range(10):
            res.append(br.put(i))
        br.put(_END)

    t = threading.Thread(target=producer)
    t.start()
    await asyncio.to_thread(t.join)
    await asyncio.sleep(0.05)
    assert cancelled.is_set()
    assert res.count(True) == 2 and res[-1] is False
    items = []
    while not q.empty():
        items.append(q.get_nowait())
    assert isinstance(items[2], Exception)  # terminal arrived despite full queue
    assert items[-1] is _END
    assert errors == []


@pytest.mark.asyncio
async def test_loop_closed_is_quiet():
    loop = asyncio.get_running_loop()
    q = make_stream_queue(2)
    br = StreamBridge(loop, q, _term)

    class Dead:
        def call_soon_threadsafe(self, fn):
            raise RuntimeError("Event loop is closed")

    br._loop = Dead()
    assert br.put(1) is False
    br.put(_END)


@pytest.mark.asyncio
async def test_legacy_check_then_enqueue_loses_items():
    """The retired pattern (full() check, then schedule put_nowait) drops items and
    leaves unhandled QueueFull callbacks; documents why the bridge exists."""
    loop = asyncio.get_running_loop()
    errors = []
    loop.set_exception_handler(lambda l, c: errors.append(c))
    q = asyncio.Queue(maxsize=2)

    def legacy_put(item):
        if not q.full():
            loop.call_soon_threadsafe(q.put_nowait, item)

    # all 5 pass the full() check before any callback runs
    t = threading.Thread(target=lambda: [legacy_put(i) for i in range(5)])
    t.start()
    t.join()
    await asyncio.sleep(0.05)
    assert q.qsize() == 2 and len(errors) == 3
