"""The VLM text prompt cache is bounded by total size, not only by entry count."""

from yunshu_engine.vlm_engine import _VLMTextPromptCache


def test_token_cache_is_bounded_by_total_tokens():
    cache = _VLMTextPromptCache(max_entries=256, max_total_tokens=100)
    for i in range(10):
        cache.put_token_ids(f"k{i}", list(range(30)))
    stats = cache.stats
    assert stats["cached_tokens"] <= 100
    assert (
        stats["tokenization_entries"] == 3
    )  # 3 x 30 tokens fit, the oldest went first
    assert cache.get_token_ids("k0") is None
    assert cache.get_token_ids("k9") == list(range(30))


def test_a_prompt_larger_than_the_budget_is_not_kept():
    cache = _VLMTextPromptCache(max_total_tokens=10)
    cache.put_token_ids("big", list(range(11)))
    assert cache.get_token_ids("big") is None
    assert cache.stats["cached_tokens"] == 0


def test_overwrite_keeps_the_token_count_exact():
    cache = _VLMTextPromptCache(max_total_tokens=100)
    cache.put_token_ids("a", [1, 2, 3])
    cache.put_token_ids("a", [1, 2])
    assert cache.stats["cached_tokens"] == 2
    cache.clear()
    assert cache.stats["cached_tokens"] == 0


def test_template_cache_is_bounded_by_total_chars():
    cache = _VLMTextPromptCache(max_entries=256, max_total_chars=100)
    for i in range(10):
        cache.put_template_text(f"k{i}", "x" * 40)
    assert cache.stats["template_chars"] <= 100
    assert cache.get_template_text("k0") is None
    assert cache.get_template_text("k9") == "x" * 40


def test_tokens_are_stored_compactly():
    cache = _VLMTextPromptCache()
    cache.put_token_ids("a", list(range(30000)))
    entry = cache._cache["a"]
    assert entry.itemsize == 4 and len(entry) == 30000
