"""the SRT/VTT timestamp formatters used in /v1/audio/transcriptions must
tolerate a None/str segment start (dict.get returns None when the key exists with a
null value) instead of crashing the whole srt/vtt response with a TypeError → 500.
The audio transcode + ASR segmentation surface was otherwise verified clean."""
from __future__ import annotations

from yunshu_gateway.routers.audio import (
    _seconds_to_srt_timestamp,
    _seconds_to_vtt_timestamp,
)


def test_float_unchanged():
    assert _seconds_to_srt_timestamp(3661.5) == "01:01:01,500"
    assert _seconds_to_vtt_timestamp(3661.5) == "01:01:01.500"
    assert _seconds_to_srt_timestamp(0.0) == "00:00:00,000"


def test_none_does_not_crash():
    assert _seconds_to_srt_timestamp(None) == "00:00:00,000"
    assert _seconds_to_vtt_timestamp(None) == "00:00:00.000"


def test_numeric_string_coerced():
    assert _seconds_to_srt_timestamp("1.5") == "00:00:01,500"
    assert _seconds_to_vtt_timestamp("1.5") == "00:00:01.500"


def test_garbage_string_falls_back_to_zero():
    assert _seconds_to_srt_timestamp("not-a-number") == "00:00:00,000"
    assert _seconds_to_vtt_timestamp("") == "00:00:00.000"
