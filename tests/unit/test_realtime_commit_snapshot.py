"""A commit transcribes the audio present when it happens, not when the lane runs it.

The ASR runs in an ordered lane; the server_vad idle-silence trim (and audio appended
meanwhile) used to change the buffer before the lane read it: a 1 s utterance reached the
auto-commit as its last 400 ms of silence, and commit + clear became commit_empty.
"""

from __future__ import annotations

import asyncio
import base64
import struct
from unittest.mock import AsyncMock, MagicMock

from yunshu_gateway.routers import realtime as rt


def _pcm(ms, amp):
    n = int(24000 * ms / 1000)
    return struct.pack(f"<{n}h", *([amp] * n))


def _append(b):
    return {"type": "input_audio_buffer.append", "audio": base64.b64encode(b).decode()}


def _session():
    ws = MagicMock()
    ws.send_json = AsyncMock()
    s = rt.RealtimeSession(ws)
    s.session.turn_detection = {
        "type": "server_vad",
        "threshold": 0.5,
        "silence_duration_ms": 500,
        "prefix_padding_ms": 300,
    }
    seen = []

    async def asr(event, **kw):
        seen.append(event.get("_snapshot"))
        return False

    s._handle_input_audio_buffer_commit = asr
    return s, seen


def test_vad_auto_commit_keeps_the_utterance():
    async def run():
        s, seen = _session()
        for _ in range(20):  # 1 s of speech
            await s._handle_input_audio_buffer_append(_append(_pcm(50, 16000)))
        for _ in range(12):  # 600 ms of silence -> speech_stopped at 500 ms
            await s._handle_input_audio_buffer_append(_append(_pcm(50, 0)))
        await asyncio.sleep(0.05)
        return seen

    seen = asyncio.run(run())
    assert len(seen) == 1
    audio, item_id = seen[0]
    # the whole utterance (1 s) plus the silence window up to speech_stopped (500 ms)
    assert len(audio) // 48 >= 1000
    assert item_id


def test_manual_commit_then_clear_still_transcribes_the_committed_audio():
    async def run():
        ws = MagicMock()
        ws.send_json = AsyncMock()
        s = rt.RealtimeSession(ws)
        s.session.turn_detection = None
        got = []

        async def asr(event, **kw):
            await asyncio.sleep(0.01)  # the ASR is slower than the next events
            snap = event.get("_snapshot")
            got.append(len(snap[0]) if snap else None)
            return False

        s._handle_input_audio_buffer_commit = asr
        await s._handle_input_audio_buffer_append(_append(_pcm(200, 16000)))
        s._lane_run(
            asr(
                {
                    "type": "input_audio_buffer.commit",
                    "_snapshot": s._take_commit_snapshot(),
                }
            )
        )
        await s._handle_input_audio_buffer_append(_append(_pcm(100, 16000)))
        await asyncio.sleep(0.05)
        return got, len(s._audio_buffer)

    got, left = asyncio.run(run())
    assert got == [200 * 48]  # exactly the audio present at commit time
    assert left == 100 * 48  # later audio belongs to the next turn
