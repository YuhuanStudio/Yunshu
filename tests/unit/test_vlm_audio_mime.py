"""VLM/omni base64 audio MIME → file-extension normalization. The decode path
(mlx_audio → miniaudio.get_file_info) dispatches on the file SUFFIX, so a data:audio/mpeg
URL (fmt="mpeg") that fell through _save_base64_audio's allowlist to the ".wav" fallback
wrote MP3 bytes into a .wav file → miniaudio DecodeError → the audio was LOST for the most
common compressed format. _save_base64_audio now maps the MIME subtype (mpeg→mp3,
x-wav→wav, x-flac→flac, …) to the canonical extension before the allowlist."""
from __future__ import annotations

import asyncio
import base64
import os

from yunshu_engine.vlm_engine import VLMEngine


def _eng():
    e = VLMEngine.__new__(VLMEngine)
    e._register_temp_file = lambda p: None  # bypass the ContextVar temp registry
    return e


def _ext_for(fmt):
    e = _eng()
    data = base64.b64encode(b"\x00\x01\x02\x03some-audio-bytes").decode()
    path = asyncio.run(e._save_base64_audio(data, fmt))
    try:
        return os.path.splitext(path)[1]
    finally:
        with __import__("contextlib").suppress(OSError):
            os.unlink(path)


def test_mpeg_maps_to_mp3():
    # the headline bug: data:audio/mpeg → "mpeg" → must become .mp3, NOT .wav
    assert _ext_for("mpeg") == ".mp3"
    assert _ext_for("mp3") == ".mp3"


def test_wav_variants():
    assert _ext_for("wav") == ".wav"
    assert _ext_for("x-wav") == ".wav"
    assert _ext_for("vnd.wave") == ".wav"


def test_flac_and_ogg_variants():
    assert _ext_for("flac") == ".flac"
    assert _ext_for("x-flac") == ".flac"
    assert _ext_for("ogg") == ".ogg"
    assert _ext_for("vorbis") == ".ogg"


def test_unknown_falls_back_to_wav():
    assert _ext_for("totally-unknown") == ".wav"
    assert _ext_for("") == ".wav"
