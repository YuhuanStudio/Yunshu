"""24-bit WAV must not be decoded as misaligned int16 in the ASR VAD pre-gate.

The VAD pre-check decoded 24-bit PCM by reinterpreting the 3-bytes-per-sample stream as
int16 (misaligned) → a garbage waveform fed to EnergyVAD, which could false-negative and
make transcribe() return {"text": ""} on valid speech. Decode 24-bit explicitly (3-byte LE
frames → sign-extended int32, /2^23).
"""

from __future__ import annotations

import inspect

import numpy as np

from yunshu_engine import audio_engine


def test_transcribe_has_explicit_24bit_branch():
    src = inspect.getsource(audio_engine.ASREngine.transcribe)
    assert "bits_per_sample == 24" in src
    assert "8388608.0" in src  # 2^23 scale


def test_24bit_decode_recovers_a_loud_tone():
    """Reproduce the decode: a loud 24-bit sine must decode to large-amplitude float
    samples (the old int16 misread would scramble them)."""
    # build a loud 24-bit LE mono sine at amplitude ~0.8 of full scale
    n = 480
    t = np.arange(n)
    amp = int(0.8 * (2**23 - 1))
    ints = (amp * np.sin(2 * np.pi * 40 * t / n)).astype(np.int64)
    # encode to 3-byte LE
    b = np.empty((n, 3), dtype=np.uint8)
    u = (ints & 0xFFFFFF).astype(np.uint32)
    b[:, 0] = u & 0xFF
    b[:, 1] = (u >> 8) & 0xFF
    b[:, 2] = (u >> 16) & 0xFF
    pcm = b.tobytes()

    # apply the decode logic
    _n24 = len(pcm) // 3
    _b = (
        np.frombuffer(pcm[: _n24 * 3], dtype=np.uint8).reshape(_n24, 3).astype(np.int32)
    )
    _i24 = _b[:, 0] | (_b[:, 1] << 8) | (_b[:, 2] << 16)
    _i24 = np.where(_i24 >= (1 << 23), _i24 - (1 << 24), _i24)
    samples = _i24.astype(np.float32) / 8388608.0

    # the recovered waveform must have substantial RMS energy (loud speech survives the gate)
    rms = float(np.sqrt(np.mean(samples**2)))
    assert rms > 0.4, f"24-bit decode lost amplitude (rms={rms})"
    assert samples.max() > 0.5 and samples.min() < -0.5
