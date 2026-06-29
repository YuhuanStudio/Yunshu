"""(HIGH): STS enhance/separate/transform truncated the trailing audio to silence.

All three STFT/iSTFT frame loops iterated `range(0, max(1, len-fft+1), hop)`, emitting only
frames that fully fit inside the signal. The `+1` recovers the tail ONLY when (len-fft)
is an exact multiple of hop; in the general case the last ~hop samples got no frame →
window_sum stayed at its 1e-8 floor → that tail reconstructed as pure SILENCE (a clipped end
on every separate/transform and every long-enough enhance). Fixed by iterating to len(arr)
(a final zero-padded frame anchors the tail; the end=min(start+fft,len) write already clamps).
"""

from __future__ import annotations

import inspect

import numpy as np

from yunshu_engine import sts_engine
from yunshu_engine.sts_engine import STSEngine


def _sine(n, sr=16000, f=440.0, amp=0.5):
    t = np.arange(n) / sr
    return (amp * np.sin(2 * np.pi * f * t)).astype(np.float32).tolist()


def _tail_has_energy(out, tail=256):
    arr = np.asarray(out, dtype=np.float32)
    return (
        float(np.max(np.abs(arr[-tail:]))) > 1e-3
    )  # clearly not the 1e-8-floor silence


def test_separate_tail_not_silenced():
    e = STSEngine.__new__(STSEngine)
    out = e._energy_mask_separation(_sine(5000), 16000, None)
    assert len(out) == 5000
    assert _tail_has_energy(out), "separate dropped the trailing samples to silence"


def test_transform_formant_tail_not_silenced():
    e = STSEngine.__new__(STSEngine)
    out = e._formant_shift(_sine(5000), 16000, 1.2)
    assert len(out) == 5000
    assert _tail_has_energy(out), (
        "transform/formant dropped the trailing samples to silence"
    )


def test_enhance_tail_not_silenced():
    e = STSEngine.__new__(STSEngine)
    # long enough that len(frames) > noise_frames so the iSTFT actually runs
    out = e._spectral_gating_enhance(_sine(20000), 16000, -40.0)
    assert len(out) == 20000
    assert _tail_has_energy(out), "enhance dropped the trailing samples to silence"


def test_all_three_loops_iterate_to_len():
    src = inspect.getsource(sts_engine)
    code = "\n".join(ln.split("#", 1)[0] for ln in src.splitlines())
    # the truncating bound is gone from all three STFT loops
    assert "max(1, len(arr) - fft_size + 1)" not in code
    # all three now iterate to len(arr)
    assert code.count("range(0, len(arr), hop_size)") >= 3
