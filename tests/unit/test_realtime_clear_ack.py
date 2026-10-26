"""input_audio_buffer.clear is answered with input_audio_buffer.cleared (OpenAI Realtime),
so a client waiting for the ack does not hang."""

from __future__ import annotations

import pytest

from .test_realtime_audio import _make_session


@pytest.mark.asyncio
async def test_clear_is_acknowledged():
    s = _make_session()
    s._audio_buffer.extend(b"\x01\x00" * 100)
    await s._handle_event({"type": "input_audio_buffer.clear"})
    assert [e["type"] for e in s.ws.sent] == ["input_audio_buffer.cleared"]
    assert len(s._audio_buffer) == 0
