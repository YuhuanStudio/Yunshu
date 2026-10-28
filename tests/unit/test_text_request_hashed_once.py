"""The text fast path hashes the rendered conversation once per request, not once for the prompt-cache
lookup and again for the store (port of the VLM 5808886 principle: the whole prompt is serialized per hash)."""

from __future__ import annotations

from unittest.mock import patch

from yunshu_engine import prompt_cache as pc

from .tiny_text_engine import make_engine

PROMPT = "the quick brown fox jumps over the lazy dog " * 4


async def test_prompt_cache_key_is_computed_once_per_request():
    eng = make_engine()
    eng._prompt_cache = pc.PromptCacheManager()
    calls = []
    real = pc.compute_messages_hash

    def counting(messages, **params):
        calls.append(1)
        return real(messages, **params)

    with patch.object(pc, "compute_messages_hash", counting):
        out = await eng._generate_fast(prompt=PROMPT, max_tokens=3, temperature=0.0)
    assert out.error is None
    assert len(calls) == 1
