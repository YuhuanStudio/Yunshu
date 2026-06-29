"""(LOW-MED): realtime server_vad speech_stopped reported audio_end_ms ~one
silence-window too LATE.

audio_end_ms must mark the END of speech (= the start of the trailing silence) so clients
can trim the committed audio. The code reported `_pre_offset` — the CURRENT buffer position
(start of the final silent chunk), which is ~silence_duration_ms after speech actually
ended — so clients over-trimmed by the whole silence window. Fixed by backing up by the
accumulated silence: audio_end_ms = (_pre_offset + len(chunk) - _vad_silence_bytes).
"""

from __future__ import annotations

import asyncio
import struct

from yunshu_gateway.routers.realtime import (
    RealtimeEvent,
    RealtimeSession,
    SessionConfig,
)


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

    s.send_event = _send

    async def _noop():
        return None

    s._auto_commit_and_respond = _noop
    return s


async def _feed(s, chunk):
    # mirror the caller: extend the buffer, then run VAD on the chunk
    s._audio_buffer.extend(chunk)
    await s._run_vad(chunk)


def test_audio_end_ms_marks_true_speech_end_not_silence_window_end():
    s = _session()
    # 100ms speech chunk (2400 samples * 2B = 4800B at 48 B/ms), loud → RMS well above gate
    speech = struct.pack("<2400h", *([10000] * 2400))
    silence = b"\x00" * 4800  # 100ms of digital silence

    asyncio.run(_run(s, speech, silence))

    stopped = [
        e
        for e in s._events
        if e.get("type") == RealtimeEvent.INPUT_AUDIO_BUFFER_SPEECH_STOPPED
    ]
    assert len(stopped) == 1
    # speech occupied bytes [0, 4800) → true speech end = offset 4800 = 100 ms.
    # The OLD code reported _pre_offset of the final silent chunk = 24000B = 500 ms.
    assert stopped[0]["audio_end_ms"] == 100


async def _run(s, speech, silence):
    await _feed(s, speech)  # speech_started
    # 5 × 100ms silence = 500ms → crosses the default silence_duration_ms threshold
    for _ in range(5):
        await _feed(s, silence)


def test_speech_started_still_reported():
    s = _session()
    speech = struct.pack("<2400h", *([10000] * 2400))
    asyncio.run(_feed(s, speech))
    started = [
        e
        for e in s._events
        if e.get("type") == RealtimeEvent.INPUT_AUDIO_BUFFER_SPEECH_STARTED
    ]
    assert len(started) == 1
    assert started[0]["audio_start_ms"] == 0
