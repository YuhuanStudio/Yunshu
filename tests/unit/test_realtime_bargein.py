"""Barge-in debounce: a one-window noise blip must NOT cancel an active response;
only SUSTAINED speech (>= barge_in_min_ms) interrupts. Uses the energy VAD path
(loud PCM) so no model is needed."""

from __future__ import annotations

import struct
from unittest.mock import AsyncMock, MagicMock

import pytest

import yunshu_gateway.routers.realtime as rt


def _loud_pcm24k(ms: int) -> bytes:
    n = int(24000 * ms / 1000)  # 24 kHz pcm16 input
    return struct.pack(f"<{n}h", *([16000] * n))  # amp ~0.5 → well above the gate


def _session_with_active_response():
    ws = MagicMock()
    ws.send_json = AsyncMock()
    s = rt.RealtimeSession(ws)
    s.session.turn_detection = {
        "type": "server_vad",
        "threshold": 0.5,
        "silence_duration_ms": 500,
        "barge_in_min_ms": 120,
    }
    # an active, not-yet-done response to be (maybe) barged-in
    resp = MagicMock()
    resp.done.return_value = False
    s._active_response = resp
    s._handle_response_cancel = AsyncMock()
    return s


@pytest.mark.asyncio
async def test_blip_does_not_barge_in():
    s = _session_with_active_response()
    # a single short blip (50 ms < 120 ms) of speech
    chunk = _loud_pcm24k(50)
    s._audio_buffer.extend(chunk)
    await s._run_vad(chunk)
    assert s._vad_speaking is True  # speech_started fired
    s._handle_response_cancel.assert_not_called()  # but no barge-in


@pytest.mark.asyncio
async def test_sustained_speech_barges_in_once():
    s = _session_with_active_response()
    # feed 50ms chunks until past 120ms: 50, 100, 150 → barge-in at the third
    for _ in range(3):
        chunk = _loud_pcm24k(50)
        s._audio_buffer.extend(chunk)
        await s._run_vad(chunk)
    assert s._handle_response_cancel.await_count == 1
    # further speech does not re-cancel within the same utterance
    chunk = _loud_pcm24k(50)
    s._audio_buffer.extend(chunk)
    await s._run_vad(chunk)
    assert s._handle_response_cancel.await_count == 1
