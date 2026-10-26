"""server_vad: silence_duration_ms is one CONTIGUOUS silent window.

The silence counter was reset only when a new utterance started, so speech that resumed
after a short pause kept the earlier pause's bytes: 300 ms pause + speech + 300 ms pause
crossed the 500 ms threshold and fired speech_stopped mid-utterance.
"""

from __future__ import annotations

import asyncio
import struct

from yunshu_gateway.routers.realtime import (
    RealtimeEvent,
    RealtimeSession,
    SessionConfig,
)

SPEECH = struct.pack("<2400h", *([10000] * 2400))  # 100 ms, loud
SILENCE = b"\x00" * 4800  # 100 ms


def _session():
    s = RealtimeSession.__new__(RealtimeSession)
    s.session = SessionConfig()  # pcm16 (48 B/ms), server_vad, 500ms
    s._audio_buffer = bytearray()
    s._vad_speaking = False
    s._vad_silence_bytes = 0
    s._vad_speech_start_offset = 0
    s._active_response = None
    s._events = []

    async def _send(ev):
        s._events.append(ev)

    async def _noop():
        return None

    s.send_event = _send
    s._auto_commit_and_respond = _noop
    return s


async def _feed(s, chunk):
    s._audio_buffer.extend(chunk)
    await s._run_vad(chunk)


def _stopped(s):
    return [
        e
        for e in s._events
        if e.get("type") == RealtimeEvent.INPUT_AUDIO_BUFFER_SPEECH_STOPPED
    ]


async def _run(s, plan):
    for chunk, n in plan:
        for _ in range(n):
            await _feed(s, chunk)


def test_short_pauses_inside_one_utterance_do_not_add_up():
    s = _session()
    # speech, 300 ms pause, speech, 300 ms pause: never 500 ms of contiguous silence
    asyncio.run(_run(s, [(SPEECH, 1), (SILENCE, 3), (SPEECH, 1), (SILENCE, 3)]))
    assert _stopped(s) == []


def test_contiguous_window_after_resumed_speech_still_ends_the_turn():
    s = _session()
    asyncio.run(_run(s, [(SPEECH, 1), (SILENCE, 3), (SPEECH, 1), (SILENCE, 5)]))
    stopped = _stopped(s)
    assert len(stopped) == 1
    # speech ended after 100 + 300 + 100 ms
    assert stopped[0]["audio_end_ms"] == 500
