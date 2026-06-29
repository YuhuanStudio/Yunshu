"""cache_tokenizer_vocab — memoizes the HF tokenizer's get_vocab so per-request
streaming-detokenizer construction doesn't rebuild the ~150k-entry vocab (~98ms/req,
the bulk of fast-path TTFT). ."""

from yunshu_engine.text_utils import cache_tokenizer_vocab


class _FakeHF:
    def __init__(self):
        self.calls = 0
        self._vocab = {"a": 0, "b": 1, "c": 2}

    def get_vocab(self, with_added_tokens=True):
        self.calls += 1
        return dict(self._vocab)


class _FakeWrapper:
    """Mimics mlx-lm TokenizerWrapper holding the HF tokenizer as _tokenizer."""

    def __init__(self, hf):
        self._tokenizer = hf


def test_caches_get_vocab_after_first_call():
    hf = _FakeHF()
    cache_tokenizer_vocab(_FakeWrapper(hf))
    # one priming call inside the cache setup
    assert hf.calls == 1
    # subsequent default-arg calls return the cache without re-hitting the base
    v1 = hf.get_vocab()
    v2 = hf.get_vocab(with_added_tokens=True)
    assert hf.calls == 1  # no extra base calls
    assert v1 == {"a": 0, "b": 1, "c": 2} == v2


def test_idempotent_no_double_wrap():
    hf = _FakeHF()
    w = _FakeWrapper(hf)
    cache_tokenizer_vocab(w)
    cache_tokenizer_vocab(w)  # second call must be a no-op (not re-prime)
    assert hf.calls == 1


def test_non_default_args_still_delegate():
    hf = _FakeHF()
    cache_tokenizer_vocab(_FakeWrapper(hf))
    before = hf.calls
    # with_added_tokens=False must NOT be served from the (added-tokens) cache
    hf.get_vocab(with_added_tokens=False)
    assert hf.calls == before + 1


def test_direct_tokenizer_without_wrapper():
    hf = _FakeHF()
    cache_tokenizer_vocab(hf)  # no _tokenizer attr → patch hf itself
    hf.get_vocab()
    assert hf.calls == 1
