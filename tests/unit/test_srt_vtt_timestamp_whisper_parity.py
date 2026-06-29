"""(LOW): SRT/VTT timestamps drifted vs the whisper-standard writer.

_seconds_to_srt/vtt_timestamp computed each field independently with truncation
(int((seconds-int(seconds))*1000)), drifting by up to ~1ms and rendering e.g. 5.999999
as "05,999" instead of "06,000" and 12.555 as ",554" instead of ",555". The whisper
writer rounds ONCE to total milliseconds then derives all fields; now we match it.
"""
from __future__ import annotations

from yunshu_gateway.routers.audio import (
    _seconds_to_srt_timestamp,
    _seconds_to_vtt_timestamp,
)


def test_rounds_at_the_millisecond_not_truncates():
    assert _seconds_to_srt_timestamp(12.555) == "00:00:12,555"   # was ,554
    assert _seconds_to_srt_timestamp(5.999999) == "00:00:06,000"  # was 05,999
    assert _seconds_to_srt_timestamp(1.2999999) == "00:00:01,300"  # was ,299


def test_field_carry_across_minute_and_hour():
    assert _seconds_to_srt_timestamp(59.9999) == "00:01:00,000"
    assert _seconds_to_srt_timestamp(3599.9999) == "01:00:00,000"
    assert _seconds_to_srt_timestamp(3661.5) == "01:01:01,500"


def test_vtt_uses_dot_separator():
    assert _seconds_to_vtt_timestamp(5.999999) == "00:00:06.000"
    assert _seconds_to_vtt_timestamp(0.0) == "00:00:00.000"


def test_none_and_negative_tolerated():
    assert _seconds_to_srt_timestamp(None) == "00:00:00,000"   # tolerance preserved
    assert _seconds_to_srt_timestamp(-1.0) == "00:00:00,000"   # clamped at 0
    assert _seconds_to_vtt_timestamp("bad") == "00:00:00.000"
