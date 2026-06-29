"""(MED, sibling): a per-response output_audio_format override was ignored.

OpenAI Realtime lets response.create carry an output_audio_format that overrides the session
default for that one response. _encode_output_audio read self.session.output_audio_format LIVE,
so a session negotiated as pcm16 + a response.create with output_audio_format="g711_ulaw"
silently encoded the audio as pcm16 (24 kHz) instead of 8 kHz μ-law — the exact wrong-
format/rate failure class, on the per-response sibling of the field.

Fix: snapshot the per-response format at response-create time (_snap_out_fmt, validated against
SUPPORTED_AUDIO_FORMATS) and thread it through _synthesize_audio_response → _encode_output_audio,
which now takes an explicit fmt that wins over the session default.
"""

from __future__ import annotations

import inspect

from yunshu_gateway.routers.realtime import RealtimeSession


class _WS:
    async def send_json(self, *a, **k):
        pass

    async def accept(self, *a, **k):
        pass


def _session():
    return RealtimeSession(_WS())


def test_encode_honors_per_response_fmt_over_session_default():
    s = _session()
    assert s.session.output_audio_format == "pcm16"  # session default
    pcm = b"\x00\x00" * 480  # 480 samples @ 24 kHz = 20 ms

    # per-response override → g711 μ-law: 8 kHz, 1 byte/sample (480 → 160 after /3 decimation)
    enc, csz = s._encode_output_audio(pcm, 24000, fmt="g711_ulaw")
    assert csz == 160
    assert len(enc) == 160

    # no override → session pcm16 passthrough at 24 kHz (2 bytes/sample)
    enc2, csz2 = s._encode_output_audio(pcm, 24000)
    assert csz2 == s._AUDIO_CHUNK_BYTES
    assert len(enc2) == len(pcm)


def test_encode_fmt_none_falls_back_to_session():
    s = _session()
    s.session.output_audio_format = "g711_alaw"
    pcm = b"\x00\x00" * 480
    enc, csz = s._encode_output_audio(pcm, 24000, fmt=None)  # None → session g711_alaw
    assert csz == 160
    assert len(enc) == 160


def test_generate_response_snapshots_and_threads_output_audio_format():
    src = inspect.getsource(RealtimeSession._generate_response)
    assert "_snap_out_fmt" in src
    assert 'config.get("output_audio_format")' in src
    assert "SUPPORTED_AUDIO_FORMATS" in src  # invalid per-response value falls back
    assert "out_fmt=_snap_out_fmt" in src  # threaded into the synth call


def test_synth_signature_accepts_out_fmt():
    sig = inspect.signature(RealtimeSession._synthesize_audio_response)
    assert "out_fmt" in sig.parameters
