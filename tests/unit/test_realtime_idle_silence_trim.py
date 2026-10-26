"""server_vad: idle silence on an always-open mic must not fill the buffer cap.

Pre-speech audio was only dropped at commit, so an idle mic reached the 10 MiB cap after
~3.5 min and got input_audio_buffer_overflow (whose text said to enable server VAD, already
on). While nobody is speaking only ~prefix_padding_ms of pre-roll is kept.
"""

from __future__ import annotations

import asyncio
import base64
import struct

from yunshu_gateway.routers.realtime import RealtimeSession, SessionConfig

SPEECH = struct.pack("<2400h", *([10000] * 2400))  # 100 ms, loud
SILENCE = b"\x00" * 4800  # 100 ms


def _session(events):
    s = RealtimeSession.__new__(RealtimeSession)
    s.session = SessionConfig()  # pcm16 (48 B/ms), server_vad, prefix_padding_ms=300
    s._audio_buffer = bytearray()
    s._vad_speaking = False
    s._vad_silence_bytes = 0
    s._vad_speech_start_offset = 0
    s._active_response = None

    async def _send(ev):
        events.append(ev)

    async def _noop():
        return None

    s.send_event = _send
    s._auto_commit_and_respond = _noop
    return s


def _append(s, chunk):
    audio = base64.b64encode(chunk).decode()
    asyncio.run(s._handle_input_audio_buffer_append({"audio": audio}))


def test_idle_silence_is_bounded_and_never_overflows(monkeypatch):
    monkeypatch.setenv("YUNSHU_REALTIME_MAX_INPUT_AUDIO_BYTES", "200000")
    events = []
    s = _session(events)
    for _ in range(200):  # 20 s idle = 960 kB, way over the 200 kB cap
        _append(s, SILENCE)
    assert not [e for e in events if e.get("type") == "error"]
    assert len(s._audio_buffer) <= 2 * 300 * 48 + len(SILENCE)  # ~prefix_padding_ms


def test_speech_after_idle_keeps_preroll(monkeypatch):
    monkeypatch.setenv("YUNSHU_REALTIME_MAX_INPUT_AUDIO_BYTES", "200000")
    events = []
    s = _session(events)
    for _ in range(100):
        _append(s, SILENCE)
    _append(s, SPEECH)
    assert s._vad_speaking
    # the speech chunk starts after >= 300 ms of pre-roll
    assert s._vad_speech_start_offset >= 300 * 48
    assert s._audio_buffer.endswith(SPEECH)


def test_overflow_message_does_not_tell_vad_users_to_enable_vad(monkeypatch):
    monkeypatch.setenv("YUNSHU_REALTIME_MAX_INPUT_AUDIO_BYTES", "100000")
    events = []
    s = _session(events)
    for _ in range(30):  # continuous speech: cannot be trimmed, hits the cap
        _append(s, SPEECH)
    errs = [e for e in events if e.get("type") == "error"]
    assert errs and "enable server VAD" not in errs[0]["error"]["message"]
