"""(HIGH): the KV prefix cache and the exact-match prompt cache (both on the
DEFAULT served fast path) are keyed on token ids (+ model name) but NOT on the active LoRA
adapter. The cached KV is computed with whatever adapter was applied when stored, so a
later request with the SAME prompt prefix but a DIFFERENT adapter (or base) would get a
prefix/exact hit and decode on the WRONG adapter's KV — silently serving a model the
caller didn't ask for (the 'LoRA-fail must not silently serve base' class, reopened
through the cache). Until the caches are adapter-keyed, both fast paths bypass cache
whenever a LoRA adapter is active."""

from __future__ import annotations

import inspect

from yunshu_engine import batched_engine


def test_generate_fast_bypasses_cache_for_lora():
    src = inspect.getsource(batched_engine.BatchedEngine._generate_fast)
    assert "if lora_adapter is not None:" in src
    assert "_bypass_cache = True" in src
    # the bypass gates BOTH caches: prompt cache (not _bypass_cache) + prefix cache
    assert "None if _bypass_cache else self._kv_prefix_cache" in src


def test_stream_generate_fast_bypasses_cache_for_lora():
    src = inspect.getsource(batched_engine.BatchedEngine._stream_generate_fast)
    # the streaming bypass expression includes the lora_adapter condition
    assert "lora_adapter is not None" in src
    assert "_stream_bypass_cache" in src
    assert "None if _stream_bypass_cache else self._kv_prefix_cache" in src
