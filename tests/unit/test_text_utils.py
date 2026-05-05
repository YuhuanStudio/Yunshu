"""Chunked prefill utility tests."""

from yunshu_engine.text_utils import (
    clean_special_tokens,
    chunk_prompt_tokens,
    estimate_prefill_memory,
    should_chunk_prefill,
)


class TestCleanSpecialTokens:
    def test_clean_im_end(self):
        assert clean_special_tokens("hello<|im_end|>world") == "helloworld"

    def test_clean_eos(self):
        assert clean_special_tokens("text<|endoftext|>") == "text"

    def test_clean_empty(self):
        assert clean_special_tokens("") == ""

    def test_clean_no_special(self):
        assert clean_special_tokens("hello world") == "hello world"

    def test_clean_multiple(self):
        text = "a<|im_start|>b<|im_end|>c</s>d"
        result = clean_special_tokens(text)
        assert "<|" not in result
        assert "</s>" not in result


class TestChunkPromptTokens:
    def test_empty(self):
        assert chunk_prompt_tokens([]) == []

    def test_small_prompt(self):
        tokens = list(range(10))
        chunks = chunk_prompt_tokens(tokens, chunk_size=4)
        assert len(chunks) == 3
        assert len(chunks[0]) == 4
        assert len(chunks[1]) == 4
        assert len(chunks[2]) == 2

    def test_exact_fit(self):
        tokens = list(range(8))
        chunks = chunk_prompt_tokens(tokens, chunk_size=4)
        assert len(chunks) == 2
        assert all(len(c) == 4 for c in chunks)

    def test_single_token(self):
        chunks = chunk_prompt_tokens([1], chunk_size=2048)
        assert len(chunks) == 1
        assert chunks[0] == [1]

    def test_large_prompt(self):
        tokens = list(range(10000))
        chunks = chunk_prompt_tokens(tokens, chunk_size=2048)
        assert len(chunks) == 5
        total = sum(len(c) for c in chunks)
        assert total == 10000


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


class TestShouldChunkPrefill:
    def test_small_prompt_no_chunk(self):
        # 128 tokens with plenty of memory — no chunking needed
        assert not should_chunk_prefill(128, available_memory=10 * 1024**3)

    def test_large_prompt_with_little_memory(self):
        # 32000 tokens with 1MB memory — definitely needs chunking
        assert should_chunk_prefill(32000, available_memory=1024)

    def test_custom_model_config(self):
        config = {"num_layers": 64, "num_kv_heads": 16, "head_dim": 128}
        result = should_chunk_prefill(2048, available_memory=1024**2, model_config=config)
        assert result  # 64 layers with 16 heads needs much more memory
