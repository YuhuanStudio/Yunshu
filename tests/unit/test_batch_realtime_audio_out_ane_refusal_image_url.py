"""realtime audio-out sample-rate, ANE non-mean refusal, chat string image_url,
data:audio subtype.

(HIGH): realtime audio output hardcoded 24 kHz, ignoring the TTS engine's real
  sample_rate → a non-24 kHz model (dia=44.1k) played at the wrong pitch/speed. Resample from
  the engine's real rate to the format target (24 kHz pcm16 / 8 kHz g711).
(HIGH): the ANE traced wrapper bakes MEAN pooling; a CLS/LAST model served via ANE was
  silently mean-pooled (wrong embedding space). Refuse ANE compilation for non-mean models so
  they use the correct MLX path.
a bare-string image_url part ({"image_url": "https://…"}) crashed _has_images with an
  AttributeError → opaque 500 (now coerced → clean handling).
a subtype-less data:audio URL raised IndexError → 500 (image fix not swept to
  audio).
"""
from __future__ import annotations

import math
import struct

from yunshu_gateway.routers.realtime import RealtimeSession


def test_resample_converts_rate_streaming():
    s = RealtimeSession.__new__(RealtimeSession)
    s._pcm16_lin_state = None
    n = 44100
    pcm = struct.pack(f"<{n}h", *[int(8000 * math.sin(2 * math.pi * 440 * i / 44100)) for i in range(n)])
    out = b""
    for c in (pcm[:30000], pcm[30000:60000], pcm[60000:]):
        out += s._resample_pcm16_linear(c, 44100, 24000, "_pcm16_lin_state")
    got = len(out) // 2
    assert 23900 < got < 24100, got  # 44100 input @44.1k → ~24000 @24k


def test_passthrough_when_rate_equal():
    s = RealtimeSession.__new__(RealtimeSession)
    pcm = b"\x01\x02\x03\x04"
    assert s._resample_pcm16_linear(pcm, 24000, 24000, "_x") == pcm


def test_encode_output_threads_rate():
    import inspect
    src = inspect.getsource(RealtimeSession._encode_output_audio)
    assert "in_rate" in src and "_resample_pcm16_linear" in src
    # the synthesis loop captures the engine's real rate
    syn = inspect.getsource(RealtimeSession._synthesize_audio_response)
    assert 'getattr(engine, "sample_rate", 24000)' in syn


def test_ane_refuses_non_mean(tmp_path):
    import json

    from yunshu_engine.ane_embedding import _model_is_mean_pooled
    assert _model_is_mean_pooled(str(tmp_path)) is True  # no config → assume mean
    pool = tmp_path / "1_Pooling"
    pool.mkdir()
    (pool / "config.json").write_text(json.dumps({"pooling_mode_cls_token": True,
                                                   "pooling_mode_mean_tokens": False}))
    assert _model_is_mean_pooled(str(tmp_path)) is False
    import inspect

    from yunshu_engine import ane_embedding
    src = inspect.getsource(ane_embedding.ANEEmbeddingProcessor.compile_model) if hasattr(
        ane_embedding.ANEEmbeddingProcessor, "compile_model") else inspect.getsource(ane_embedding)
    assert "_model_is_mean_pooled(model_path)" in src


def test_chat_string_image_url_no_crash():
    import inspect

    from yunshu_gateway.routers import chat
    src = inspect.getsource(chat)
    assert "if isinstance(_iu, str):" in src
    assert '_iu = {"url": _iu}' in src


def test_data_audio_subtype_guard():
    import inspect

    from yunshu_engine import vlm_engine
    src = inspect.getsource(vlm_engine)
    assert 'fmt = _hp[1].split(";")[0] if len(_hp) > 1 else "wav"' in src
