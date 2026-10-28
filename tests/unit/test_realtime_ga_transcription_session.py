"""GA session.created / session.updated of a transcription-only session report
type "transcription" with the input side only, not a full "realtime" session."""

from __future__ import annotations

import pytest

from yunshu_gateway import realtime_ga as ga

from .test_realtime_gaps import _session


def test_to_ga_transcription_session_shape():
    out = ga.to_ga(
        {
            "type": "session.updated",
            "session": {
                "id": "sess_1",
                "model": "m",
                "voice": "alloy",
                "input_audio_format": "g711_ulaw",
                "turn_detection": {"type": "server_vad"},
                "input_audio_transcription": {"model": "whisper-1"},
            },
        },
        "transcription",
    )[0]["session"]
    assert out["type"] == "transcription" and out["id"] == "sess_1"
    assert out["audio"]["input"]["transcription"] == {"model": "whisper-1"}
    assert out["audio"]["input"]["format"] == {"type": "audio/pcmu"}
    assert "output" not in out["audio"] and "model" not in out and "tools" not in out


@pytest.mark.asyncio
async def test_transcription_only_ga_session_updated_is_a_transcription_session():
    s, sent = _session("ga")
    await s._handle_event(
        {"type": "session.update", "session": {"type": "transcription"}}
    )
    updated = next(e for e in sent if e["type"] == "session.updated")
    assert updated["session"]["type"] == "transcription"
    assert "output" not in updated["session"]["audio"]
