"""WebRTC data-channel adapter: a slow session drops audio frames, never the call or a control event."""

import asyncio
import json
from types import SimpleNamespace

from yunshu_gateway.realtime_webrtc import ChannelSocket

AUDIO = json.dumps({"type": "input_audio_buffer.append", "audio": "AAAA"})
CANCEL = json.dumps({"type": "response.cancel"})


def _socket():
    return ChannelSocket(SimpleNamespace(bufferedAmount=0, send=lambda d: None), None)


def test_full_queue_drops_audio_and_keeps_the_call():
    async def run():
        s = _socket()
        for _ in range(256):
            s.feed(AUDIO)
        for _ in range(10):
            s.feed(AUDIO)  # no room: these frames are dropped and counted
        assert not s.closed and s.dropped_audio == 10 and s.incoming.qsize() == 256
        s.feed(CANCEL)  # a control event evicts the oldest audio frame instead
        assert not s.closed and s.dropped_audio == 11 and s.incoming.qsize() == 256
        got = [await s.receive_text() for _ in range(256)]
        assert CANCEL in got and got[-1] == CANCEL

    asyncio.run(run())


def test_rtp_audio_offer_never_blocks_or_closes():
    async def run():
        s = _socket()
        for _ in range(300):
            s.offer(AUDIO, audio=True)
        assert not s.closed and s.dropped_audio == 44

    asyncio.run(run())


def test_control_flood_still_disconnects():
    async def run():
        s = _socket()
        for _ in range(257):
            s.feed(CANCEL)
        assert s.closed

    asyncio.run(run())
