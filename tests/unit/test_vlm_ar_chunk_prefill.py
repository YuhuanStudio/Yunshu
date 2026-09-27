"""The opt-in VLM AR prefill keeps the final token and checks cancellation."""

import threading

import mlx.core as mx

from yunshu_engine.vlm_engine import VLMEngine


class _CacheEntry:
    state = mx.array([0])


def _engine():
    engine = object.__new__(VLMEngine)
    engine._config = {"text_config": {"model_type": "qwen3_5_text"}}
    return engine


def test_chunk_prefill_keeps_last_token_for_logits(monkeypatch):
    monkeypatch.setenv("YUNSHU_VLM_AR_PREFILL_CHUNK_TOKENS", "16")
    calls = []

    def lm(ids, *, cache, skip_logits=False):
        calls.append((ids.tolist()[0], skip_logits))
        return object()

    result = _engine()._prefill_vlm_ar_text(
        lm, mx.arange(35), [_CacheEntry()], threading.Event()
    )
    assert result is not None
    assert calls == [
        (list(range(16)), True),
        (list(range(16, 32)), True),
        ([32, 33], True),
        ([34], False),
    ]


def test_chunk_prefill_stops_between_evaluated_chunks(monkeypatch):
    monkeypatch.setenv("YUNSHU_VLM_AR_PREFILL_CHUNK_TOKENS", "16")
    cancel = threading.Event()
    calls = []

    def lm(ids, *, cache, skip_logits=False):
        calls.append(ids.tolist()[0])
        cancel.set()
        return object()

    result = _engine()._prefill_vlm_ar_text(lm, mx.arange(35), [_CacheEntry()], cancel)
    assert result is None
    assert calls == [list(range(16))]
