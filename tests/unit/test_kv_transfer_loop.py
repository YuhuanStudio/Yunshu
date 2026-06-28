"""KVTransferClient connection pool must not reuse a StreamWriter across
event loops. send_blocks_sync runs every transfer under a fresh throwaway asyncio.run
loop, so a writer cached by a prior call is bound to a dead loop — reusing it raised
'got Future attached to a different loop' and every disagg transfer after the first
silently failed (decode fell back to re-prefill). Reuse only within the same loop."""
from __future__ import annotations

import asyncio
from unittest.mock import patch

from yunshu_engine.kv_transfer import KVTransferClient


class _FakeWriter:
    def __init__(self):
        self._closing = False

    def is_closing(self):
        return self._closing

    def close(self):
        self._closing = True

    async def wait_closed(self):
        return None


def _make_open_counter(counter):
    async def _open(host, port):
        counter["n"] += 1
        return (object(), _FakeWriter())  # (reader, writer)
    return _open


def test_no_cross_loop_writer_reuse():
    client = KVTransferClient()
    counter = {"n": 0}

    with patch("asyncio.open_connection", _make_open_counter(counter)):
        # Each asyncio.run is a fresh, throwaway loop (the send_blocks_sync pattern).
        asyncio.run(client._get_connection())
        assert counter["n"] == 1
        # Second call under a DIFFERENT loop must NOT reuse the dead-loop writer.
        asyncio.run(client._get_connection())
        assert counter["n"] == 2, "cross-loop reuse → would raise on write/drain"


def test_same_loop_reuses_connection():
    client = KVTransferClient()
    counter = {"n": 0}

    async def _twice():
        await client._get_connection()
        await client._get_connection()  # same loop → reuse, no new open

    with patch("asyncio.open_connection", _make_open_counter(counter)):
        asyncio.run(_twice())

    assert counter["n"] == 1  # opened once, reused within the loop
