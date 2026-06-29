"""server_vad silence-window timing must be measured in AUDIO time, not
wall-clock.

The old `_run_vad` used `time.monotonic()` to time the post-speech silence window, so
turn-end detection depended on network DELIVERY rather than audio content: a
faster-than-realtime bulk/catch-up upload fired speech_stopped very late or never, and
jittery/sparse delivery fired it early, truncating the user mid-pause. OpenAI server_vad
measures the silence window in audio time. This test feeds silent chunks and asserts
speech_stopped fires once `silence_duration_ms` worth of silent BYTES has accumulated,
regardless of how fast (in wall-clock terms) those chunks arrive.
"""

from __future__ import annotations

import inspect

from yunshu_gateway.routers import realtime


def test_run_vad_silence_window_is_audio_bytes_not_wallclock():
    src = inspect.getsource(realtime.RealtimeSession._run_vad)
    # the silence window accumulates silent audio bytes ...
    assert "self._vad_silence_bytes += len(audio_chunk)" in src
    # ... and the threshold is bytes/_bytes_per_ms >= silence_duration_ms (audio time)
    assert "self._vad_silence_bytes / _bytes_per_ms >= silence_duration_ms" in src
    # the old wall-clock basis is gone from the silence path (the timing attribute and
    # the live monotonic() read; the explanatory comment may still name the old basis)
    assert "_vad_silence_start" not in src
    assert "now = time.monotonic()" not in src


def test_vad_silence_start_attr_fully_replaced():
    # no stale wall-clock attribute survives anywhere in the module
    full = inspect.getsource(realtime)
    assert "_vad_silence_start" not in full
    # the new audio-byte counter is the one in use
    assert "_vad_silence_bytes" in full
