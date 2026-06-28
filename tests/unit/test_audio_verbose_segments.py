"""transcription verbose_json segments were emitted as whatever the underlying
mlx-audio model produced (typically text/start/end[/words]) — missing the OpenAI segment
fields id/seek/tokens/temperature/avg_logprob/compression_ratio/no_speech_prob — so strict
OpenAI-SDK consumers that type-validate segments or read those fields got a partial object.
_normalize_verbose_segments pads the full schema while preserving real values.
"""
from __future__ import annotations

from yunshu_gateway.routers.audio import _normalize_verbose_segments

_OPENAI_FIELDS = {"id", "seek", "start", "end", "text", "tokens", "temperature",
                  "avg_logprob", "compression_ratio", "no_speech_prob"}


def test_pads_full_openai_schema():
    out = _normalize_verbose_segments([{"text": "hi", "start": 1.0, "end": 2.0}])
    assert set(out[0]) >= _OPENAI_FIELDS
    # real values preserved
    assert out[0]["text"] == "hi" and out[0]["start"] == 1.0 and out[0]["end"] == 2.0
    # id is the index, defaults neutral
    assert out[0]["id"] == 0 and out[0]["no_speech_prob"] == 0.0 and out[0]["tokens"] == []


def test_empty_and_none_safe():
    assert _normalize_verbose_segments([]) == []
    assert _normalize_verbose_segments(None) == []


def test_index_increments_and_preserves_extra_keys():
    out = _normalize_verbose_segments([
        {"text": "a", "words": [{"word": "a"}]},
        {"text": "b", "avg_logprob": -0.3},
    ])
    assert out[0]["id"] == 0 and out[1]["id"] == 1
    assert out[0]["words"] == [{"word": "a"}]          # extra real key kept
    assert out[1]["avg_logprob"] == -0.3                # real value not overwritten by default
