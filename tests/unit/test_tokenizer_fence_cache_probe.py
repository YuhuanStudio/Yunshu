"""NFC/BPE prefix reuse only crosses a certified non-normalized added token."""

import pytest
from mlx_lm.tokenizer_utils import _byte_decoder
from scripts.research.tokenizer_fence_cache import FenceCache
from tokenizers import AddedToken, normalizers, processors
from transformers.models.qwen3_5.tokenization_qwen3_5 import Qwen3_5Tokenizer


@pytest.fixture
def tokenizer():
    vocab = dict(_byte_decoder())
    tokenizer = Qwen3_5Tokenizer(
        vocab=vocab,
        merges=[],
        bos_token=None,
        eos_token="<|im_end|>",
        pad_token="<|endoftext|>",
    )
    tokenizer.backend_tokenizer.post_processor = processors.ByteLevel(
        trim_offsets=False
    )
    return tokenizer


@pytest.mark.parametrize(
    "prefix", ["hello", "e\u0301", "你好🏳️‍🌈", "\r\n  ", "literal <|im_end|> token"]
)
@pytest.mark.parametrize(
    "suffix",
    ["\u0301a", "\u0338a", "\n<|im_start|>assistant\n你好", " \n test", "<|im_end|>"],
)
def test_fence_cache_keeps_all_ids_across_nfc_and_special_boundaries(
    tokenizer, prefix, suffix
):
    cache = FenceCache()
    text = prefix + "<|im_end|>"
    assert cache.encode(tokenizer, text + "\nassistant") == tokenizer.encode(
        text + "\nassistant"
    )
    assert cache.encode(tokenizer, text + suffix) == tokenizer.encode(text + suffix)
    assert cache.hits == 1 and cache.reused_tokens > 0


def test_no_arbitrary_bpe_boundary_and_bounded_storage(tokenizer):
    cache = FenceCache(max_bytes=256, max_entries=1)
    cache.encode(tokenizer, "hello")
    cache.encode(tokenizer, "hello world")
    assert not cache.entries and cache.hits == 0
    for text in ("a<|im_end|>", "b<|im_end|>", "x" * 100 + "<|im_end|>"):
        assert cache.encode(tokenizer, text) == tokenizer.encode(text)
        assert len(cache.entries) <= 1 and cache.bytes <= 256


def test_unsupported_normalizer_and_inplace_added_token_policy_invalidate(tokenizer):
    cache = FenceCache()
    text = "hello <|im_end|>"
    cache.encode(tokenizer, text)
    tokenizer.backend_tokenizer.normalizer = normalizers.NFKC()
    assert cache.encode(tokenizer, text + "tail") == tokenizer.encode(text + "tail")
    assert not cache.entries
    tokenizer.backend_tokenizer.normalizer = normalizers.NFC()
    cache.encode(tokenizer, text)
    tokenizer.backend_tokenizer.add_special_tokens(
        [AddedToken("<|im_end|>", normalized=True, special=True)]
    )
    assert cache.encode(tokenizer, text + "tail") == tokenizer.encode(text + "tail")
    assert not cache.entries


def test_encoder_override_falls_back_without_reusing_stale_ids(tokenizer):
    cache = FenceCache()
    text = "hello <|im_end|>"
    cache.encode(tokenizer, text)
    tokenizer.encode = lambda text, **kwargs: [9]
    assert cache.encode(tokenizer, text + "tail") == [9]
    assert not cache.entries


def test_longer_added_token_cannot_swallow_the_fence_and_suffix(tokenizer):
    tokenizer.backend_tokenizer.add_special_tokens(
        [AddedToken("<|im_end|>foo", normalized=False, special=True)]
    )
    cache = FenceCache()
    first = "hello <|im_end|>bar"
    second = "hello <|im_end|>foo"
    assert cache.encode(tokenizer, first) == tokenizer.encode(first)
    assert cache.encode(tokenizer, second) == tokenizer.encode(second)
    assert not cache.entries and cache.hits == 0


def test_legacy_encoder_without_special_keyword_keeps_original_fallback():
    class Tokenizer:
        def encode(self, text):
            return [len(text)]

    assert FenceCache().encode(Tokenizer(), "hello") == [5]
