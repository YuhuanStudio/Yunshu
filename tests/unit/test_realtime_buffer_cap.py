"""the realtime input_audio_buffer must be bounded. It is only drained by
commit/clear/auto-commit, so a client that streams append forever without committing
(legal when turn_detection is null) — or speaks continuously so server-VAD never hits
the silence window — grew it without limit (a memory/DoS vector reachable by any
can_infer key). Over the cap → emit an overflow error + clear the buffer."""
from __future__ import annotations

import asyncio
import base64
import types

from yunshu_gateway.routers.realtime import RealtimeSession


def _make_session(events):
    s = RealtimeSession.__new__(RealtimeSession)
    s._audio_buffer = bytearray()
    s._vad_speaking = False
    s._vad_silence_start = None
    s.session = types.SimpleNamespace(input_audio_format="pcm16", turn_detection=None)

    async def _cap(ev):
        events.append(ev)

    s.send_event = _cap
    return s


def _append(session, nbytes):
    audio = base64.b64encode(b"\x00" * nbytes).decode()
    asyncio.run(session._handle_input_audio_buffer_append({"audio": audio}))


def test_buffer_overflow_emits_error_and_clears(monkeypatch):
    monkeypatch.setenv("YUNSHU_REALTIME_MAX_INPUT_AUDIO_BYTES", "100")
    events = []
    s = _make_session(events)
    _append(s, 50)
    assert len(s._audio_buffer) == 50 and events == []  # under cap, accumulates
    _append(s, 80)  # now 130 > 100 → overflow
    assert any(e.get("type") == "error"
               and e.get("error", {}).get("code") == "input_audio_buffer_overflow"
               for e in events)
    assert len(s._audio_buffer) == 0  # cleared
    assert s._vad_speaking is False and s._vad_silence_start is None


def test_under_cap_no_error(monkeypatch):
    monkeypatch.setenv("YUNSHU_REALTIME_MAX_INPUT_AUDIO_BYTES", "10000")
    events = []
    s = _make_session(events)
    for _ in range(5):
        _append(s, 100)
    assert len(s._audio_buffer) == 500
    assert events == []


def test_default_cap_is_generous(monkeypatch):
    monkeypatch.delenv("YUNSHU_REALTIME_MAX_INPUT_AUDIO_BYTES", raising=False)
    from yunshu_gateway.routers.realtime import _max_input_audio_bytes
    # ~10MB default ≈ 3.5 min @ 24kHz/16-bit — a real utterance fits.
    assert _max_input_audio_bytes() >= 8 * 1024 * 1024
