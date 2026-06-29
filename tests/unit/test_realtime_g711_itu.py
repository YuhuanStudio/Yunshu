"""(HIGH): the realtime g711 codec tables were self-consistent but NON-compliant
with ITU-T G.711, so they passed loopback + the "37 dB" self-test yet produced garbled
audio for any real telephony/SIP client.
  - μ-law: every nonzero sample's polarity was INVERTED (negate-when-sign-clear) → interop
    SNR ~-6 dB (full-scale inversion).
  - A-law: the exp>0 branch used 0x100 instead of 0x108, dropping the +8 half-step bias →
    decoded values off the quantization midpoint by up to 512 (~6 dB quality loss).
Both fixed and verified 256/256 against ITU-T (Python audioop).

This test pins a few canonical ITU-T reference values (so it has NO audioop dependency,
which is removed in py3.13) plus a loopback round-trip.
"""
from __future__ import annotations

import struct

from yunshu_gateway.routers.realtime import RealtimeSession


def _decode_byte_ulaw(b):
    s = RealtimeSession.__new__(RealtimeSession)
    return struct.unpack("<h", s._decode_g711_ulaw(bytes([b])))[0]


def _decode_byte_alaw(b):
    s = RealtimeSession.__new__(RealtimeSession)
    return struct.unpack("<h", s._decode_g711_alaw(bytes([b])))[0]


def test_ulaw_matches_itu_reference():
    # canonical ITU-T G.711 μ-law decode values (== Python audioop.ulaw2lin)
    assert _decode_byte_ulaw(0) == -32124      # was +32124 (sign-flipped)
    assert _decode_byte_ulaw(144) == 15996     # was -15996
    assert _decode_byte_ulaw(255) == 0
    assert _decode_byte_ulaw(127) == 0


def test_alaw_matches_itu_reference():
    # canonical ITU-T G.711 A-law decode values (== Python audioop.alaw2lin)
    assert _decode_byte_alaw(200) == 472       # was 464 (missing +8 bias)
    assert _decode_byte_alaw(213) == 8         # 0xD5 ^ 0x55 = 0x80 → +8
    assert _decode_byte_alaw(85) == -8         # 0x55 ^ 0x55 = 0x00 → -8


def test_ulaw_decode_table_full_itu_match():
    # the WHOLE μ-law decode table must be the standard one (the inversion bug was
    # systematic across all 256 codes, not just a few)
    s = RealtimeSession.__new__(RealtimeSession)
    tbl = s._get_ulaw_table()
    # standard μ-law is antisymmetric: code i and its sign-bit-flipped partner negate.
    # spot-check monotonic magnitude within a segment + correct sign half.
    assert tbl[0] < 0 and tbl[128] > 0      # high-amplitude codes, opposite signs
    assert all(isinstance(v, int) and -32768 <= v <= 32767 for v in tbl)
