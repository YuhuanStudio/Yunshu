"""audio DSP robustness (the core STFT/overlap-add/pitch/WAV math was verified
clean; these are edge/numeric guards).

- sts_engine._load_audio crashed on a truncated / odd-length WAV PCM data chunk
  (np.frombuffer "buffer size must be a multiple of element size", and the channel
  reshape) → enhance/separate/transform 500'd on a short upload. audio_engine.py:821
  already guarded this; sts was the un-propagated sibling. Now truncates to the
  element/frame boundary.
- NaN/Inf audio samples passed np.clip unchanged → NaN.astype(int16)=0 with a
  RuntimeWarning (silent click + log noise). Now nan_to_num before clip in all THREE
  encoders .
- the VAD frame conversion used *32768 (full-scale +1.0 → 32768 wraps to -32768); now
  *32767 like every other conversion.
"""
from __future__ import annotations

import inspect
import struct
import warnings

import numpy as np

from yunshu_engine import audio_engine
from yunshu_engine.sts_engine import STSEngine


def _wav(data: bytes, sr=16000, ch=1, bps=16) -> bytes:
    br = sr * ch * bps // 8
    ba = ch * bps // 8
    return (b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVEfmt "
            + struct.pack("<IHHIIHH", 16, 1, ch, sr, br, ba, bps)
            + b"data" + struct.pack("<I", len(data)) + data)


def test_audio_to_wav_bytes_guards_nan_inf():
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any RuntimeWarning → failure
        out = audio_engine._audio_to_wav_bytes(
            np.array([0.5, np.nan, np.inf, -np.inf, -0.5], dtype=np.float32), 16000)
    assert len(out) > 44  # valid WAV (header + data)


def test_sts_load_audio_odd_length_no_crash():
    e = STSEngine.__new__(STSEngine)
    # 3-byte data chunk is odd for int16 — used to raise ValueError from frombuffer
    samples, sr = e._load_audio(_wav(b"\x01\x02\x03"))
    assert sr == 16000
    assert len(samples) == 1  # truncated to the 2-byte boundary


def test_sts_load_audio_misaligned_stereo_no_crash():
    e = STSEngine.__new__(STSEngine)
    # stereo (2ch) 16-bit: 6 bytes = 3 int16 = 1.5 frames → trailing partial frame dropped
    samples, sr = e._load_audio(_wav(b"\x01\x02\x03\x04\x05\x06", ch=2))
    assert len(samples) == 1  # 1 complete stereo frame after trimming


def test_sts_encode_wav_guards_nan():
    e = STSEngine.__new__(STSEngine)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = e._encode_wav([0.5, float("nan"), float("inf"), -0.5], 16000)
    assert len(out) > 44


def test_vad_uses_32767_not_32768():
    src = inspect.getsource(audio_engine)
    # the overflow-prone *32768 int16 conversion is gone
    assert "* 32768.0).astype(np.int16)" not in src


def test_streaming_tts_encoder_guards_nan():
    # the streaming TTS inline encoder (synthesize_stream) is the third
    # encode site; it clips result.audio → must nan_to_num FIRST (it's deep inside a
    # model-driven generator loop, so source-guard the ordering).
    src = inspect.getsource(audio_engine.TTSEngine.synthesize_stream)
    code = "\n".join(ln.split("#", 1)[0] for ln in src.splitlines())
    assert "np.array(result.audio).flatten()" in code
    nan_pos = code.find("nan_to_num")
    clip_pos = code.find("np.clip(audio")
    assert nan_pos != -1, "streaming encoder must guard NaN/Inf"
    assert nan_pos < clip_pos, "nan_to_num must precede np.clip (NaN passes through clip)"
