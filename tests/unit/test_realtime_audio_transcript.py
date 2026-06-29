"""an audio response must stream its transcript.

OpenAI's Realtime API always emits response.audio_transcript.delta/.done alongside the
audio so the client can render what's being spoken. The token loop only emitted
response.text.delta gated behind `"text" in modalities`; for an audio-only modality
response the client received audio but NO transcript at all (text delta gated off,
audio_transcript never emitted anywhere). Stream the transcript whenever audio is in the
negotiated modalities.
"""

from __future__ import annotations

import inspect

from yunshu_gateway.routers import realtime


def test_generate_response_emits_audio_transcript_for_audio_modality():
    src = inspect.getsource(realtime.RealtimeSession._generate_response)
    # transcript deltas are emitted under an audio-modality gate, in BOTH stream branches
    assert src.count("RESPONSE_AUDIO_TRANSCRIPT_DELTA") >= 2
    # and the transcript stream is finalized
    assert "RESPONSE_AUDIO_TRANSCRIPT_DONE" in src
    # the emission is gated on audio being in the negotiated modalities
    i = src.index("RESPONSE_AUDIO_TRANSCRIPT_DELTA")
    window = src[max(0, i - 200) : i]
    assert '"audio" in modalities' in window
