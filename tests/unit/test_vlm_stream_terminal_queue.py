"""The stream's terminal items (end sentinel, error) survive a full bounded queue."""

import asyncio
import threading

from yunshu_engine.request import RequestOutput
from yunshu_engine.vlm_engine import _ThreadSafeQueue


def test_terminal_items_are_not_dropped_when_the_queue_is_full():
    async def main():
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue(maxsize=2)
        safe = _ThreadSafeQueue(q, loop)
        q.put_nowait("a")
        q.put_nowait("b")  # a slow SSE client: the queue is full
        err = RequestOutput(
            request_id="r", finish_reason="error", finished=True, error="x"
        )
        t = threading.Thread(
            target=lambda: (safe.put_nowait(err), safe.put_nowait(None))
        )
        t.start()
        t.join()
        got = []
        for _ in range(4):
            got.append(await asyncio.wait_for(q.get(), 2))
        return got, err

    got, err = asyncio.run(main())
    assert got == ["a", "b", err, None]  # nothing lost, order kept


def test_close_cancels_the_drain_task_of_a_gone_consumer():
    async def main():
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue(maxsize=1)
        safe = _ThreadSafeQueue(q, loop)
        q.put_nowait("a")  # full, and nobody reads it any more
        t = threading.Thread(target=lambda: safe.put_nowait(None))
        t.start()
        t.join()
        await asyncio.sleep(0.05)
        drain = safe._drain
        assert drain is not None and not drain.done()  # parked, waiting for room
        safe.close()
        await asyncio.sleep(0)
        assert drain.cancelled()
        safe._put_terminal(None)  # a late terminal item from the runner thread
        assert not safe._backlog and q.qsize() == 1

    asyncio.run(main())
