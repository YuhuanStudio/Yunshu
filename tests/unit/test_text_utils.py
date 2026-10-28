"""Chunked prefill utility tests."""

from yunshu_engine.text_utils import (
    estimate_prefill_memory,
)


class TestEstimatePrefillMemory:
    def test_basic_estimate(self):
        mem = estimate_prefill_memory(1024)
        assert mem > 0

    def test_scales_with_tokens(self):
        small = estimate_prefill_memory(512)
        large = estimate_prefill_memory(2048)
        assert large > small

    def test_head_dim_256_uses_more(self):
        """head_dim > 128 triggers O(n^2) attention."""
        small_dim = estimate_prefill_memory(1024, head_dim=128)
        large_dim = estimate_prefill_memory(1024, head_dim=256)
        assert large_dim > small_dim
