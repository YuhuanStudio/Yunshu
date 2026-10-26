"""The numpy realtime audio conversions are byte-identical to the former pure-Python ones.

Resampling, G.711 encode/decode and the encode-LUT build ran as per-sample Python loops on
the event loop (seconds for a 30-60 s non-streaming WAV); they are vectorized now. The
reference implementations below are the previous code, verbatim in behaviour.
"""

from __future__ import annotations

import bisect
import random
import struct
import time

import pytest

from yunshu_gateway.routers.realtime import RealtimeSession


def _ref_lut(decode_table):
    codes = sorted(range(256), key=lambda c: decode_table[c])
    vals = [decode_table[c] for c in codes]
    lut = bytearray(65536)
    for pcm in range(-32768, 32768):
        j = bisect.bisect_left(vals, pcm)
        if j <= 0:
            best = codes[0]
        elif j >= 256:
            best = codes[255]
        else:
            best = (
                codes[j]
                if abs(vals[j] - pcm) < abs(vals[j - 1] - pcm)
                else codes[j - 1]
            )
        lut[pcm & 0xFFFF] = best
    return bytes(lut)


def _ref_24k_to_8k(pcm16):
    n = len(pcm16) // 2
    if n < 3:
        return b"", pcm16[: n * 2]
    s = struct.unpack(f"<{n}h", pcm16[: n * 2])
    consumed = (n // 3) * 3
    out = [(s[i] + s[i + 1] + s[i + 2]) // 3 for i in range(0, consumed, 3)]
    return struct.pack(f"<{len(out)}h", *out), pcm16[consumed * 2 : n * 2]


def _ref_linear(state, pcm16, from_rate, to_rate):
    n = len(pcm16) // 2
    if n == 0:
        return b"", state
    samples = list(struct.unpack(f"<{n}h", pcm16[: n * 2]))
    last, pos = state if state is not None else (samples[0], 0.0)
    buf = [last] + samples
    step = from_rate / to_rate
    out = []
    i = pos
    last_idx = len(buf) - 1
    while i < last_idx:
        i0 = int(i)
        frac = i - i0
        out.append(int(buf[i0] * (1.0 - frac) + buf[i0 + 1] * frac))
        i += step
    return (struct.pack(f"<{len(out)}h", *out) if out else b""), (
        buf[-1],
        i - last_idx,
    )


def _pcm(rng, n, amp=32767):
    return struct.pack(f"<{n}h", *[rng.randint(-amp - 1, amp) for _ in range(n)])


def test_encode_luts_match_the_bisect_build():
    for table in (
        RealtimeSession._get_ulaw_table(),
        RealtimeSession._get_alaw_table(),
    ):
        assert RealtimeSession._build_encode_lut(table) == _ref_lut(table)


def test_g711_decode_matches_table_lookup():
    rng = random.Random(1)
    data = bytes(rng.randrange(256) for _ in range(5000))
    s = RealtimeSession.__new__(RealtimeSession)
    for fn, table in (
        (s._decode_g711_ulaw, RealtimeSession._get_ulaw_table()),
        (s._decode_g711_alaw, RealtimeSession._get_alaw_table()),
    ):
        assert fn(data) == struct.pack(f"<{len(data)}h", *[table[b] for b in data])
    assert s._decode_g711_ulaw(b"") == b""


def test_24k_to_8k_matches_including_leftover_and_extremes():
    rng = random.Random(2)
    for n in (0, 1, 2, 3, 4, 5, 7, 960, 4801):
        pcm = _pcm(rng, n)
        assert RealtimeSession._resample_24k_to_8k(pcm) == _ref_24k_to_8k(pcm)
    ext = struct.pack("<6h", -32768, -32768, -32767, 32767, 32767, -1)
    assert RealtimeSession._resample_24k_to_8k(ext) == _ref_24k_to_8k(ext)


@pytest.mark.parametrize("rates", [(44100, 24000), (44100, 8000), (16000, 24000)])
def test_linear_resample_streaming_matches_across_chunks(rates):
    rng = random.Random(3)
    s = RealtimeSession.__new__(RealtimeSession)
    ref_state = None
    for n in (1, 2, 3, 7, 441, 1000, 4410, 13):
        pcm = _pcm(rng, n, amp=20000)
        want, ref_state = _ref_linear(ref_state, pcm, *rates)
        assert s._resample_pcm16_linear(pcm, *rates, "_st") == want
        assert s._st == ref_state


def test_encode_output_audio_g711_matches_reference_end_to_end():
    rng = random.Random(4)
    for fmt, lut_getter in (
        ("g711_ulaw", RealtimeSession._get_ulaw_encode_lut),
        ("g711_alaw", RealtimeSession._get_alaw_encode_lut),
    ):
        s = RealtimeSession.__new__(RealtimeSession)
        pcm = _pcm(rng, 4803)
        lut = lut_getter()
        got, size = s._encode_output_audio(pcm, 24000, fmt)
        pcm8k, _ = _ref_24k_to_8k(pcm)
        samples = struct.unpack(f"<{len(pcm8k) // 2}h", pcm8k)
        assert got == bytes(lut[x & 0xFFFF] for x in samples) and size == 160
    assert RealtimeSession.__new__(RealtimeSession)._encode_output_audio(
        b"", 24000, "g711_ulaw"
    ) == (b"", 160)


def test_a_minute_of_audio_converts_in_well_under_a_second():
    pcm = b"\x10\x20" * (24000 * 60)
    s = RealtimeSession.__new__(RealtimeSession)
    t0 = time.process_time()
    s._encode_output_audio(pcm, 24000, "g711_ulaw")
    s._encode_output_audio(b"\x10\x20" * (44100 * 60), 44100, "g711_alaw")
    assert time.process_time() - t0 < 0.3
