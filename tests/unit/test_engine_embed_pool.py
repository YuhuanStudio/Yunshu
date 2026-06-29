"""Tests for BatchedEngine embed() and pool() methods."""

import pytest

from yunshu_engine.batched_engine import BatchedEngine


class _FakeConfig:
    """Minimal model config with hidden_size."""

    hidden_size = 64
    d_model = None
    n_embd = None
    embed_dim = None


class _FakeModel:
    """Mock model that returns predictable hidden states."""

    config = _FakeConfig()

    def __call__(self, input_ids):
        # Return shape [batch=1, seq_len, hidden_size=64]
        import mlx.core as mx

        seq_len = input_ids.shape[1]
        # Return ones so mean pooling gives all-ones
        return mx.ones((1, seq_len, 64))


class _FakeTokenizer:
    """Mock tokenizer."""

    def encode(self, text, **kwargs):
        if not text.strip():
            return []
        return list(range(len(text)))


def _make_loaded_engine():
    """Create a BatchedEngine with fake model/tokenizer loaded."""
    engine = BatchedEngine.__new__(BatchedEngine)
    engine._model = _FakeModel()
    engine._tokenizer = _FakeTokenizer()
    engine._loaded = True
    return engine


class TestBatchedEngineEmbed:
    """Tests for BatchedEngine.embed()."""

    def test_embed_returns_normalized_vectors(self):
        engine = _make_loaded_engine()
        result = engine.embed(["hello"])
        assert len(result) == 1
        vec = result[0]
        assert len(vec) == 64
        # L2 norm should be ~1.0 (normalized)
        norm = sum(v * v for v in vec) ** 0.5
        assert abs(norm - 1.0) < 1e-5

    def test_embed_batch(self):
        engine = _make_loaded_engine()
        result = engine.embed(["hello", "world", "test"])
        assert len(result) == 3
        for vec in result:
            assert len(vec) == 64

    def test_embed_empty_text_gives_zero_vector(self):
        engine = _make_loaded_engine()
        result = engine.embed([""])
        assert len(result) == 1
        assert all(v == 0.0 for v in result[0])

    def test_embed_without_normalize(self):
        engine = _make_loaded_engine()
        result = engine.embed(["hello"], normalize=False)
        assert len(result) == 1
        vec = result[0]
        # Without normalization, all-ones mean-pooled should be all-ones
        assert all(v == 1.0 for v in vec)

    def test_embed_not_loaded_raises(self):
        engine = BatchedEngine.__new__(BatchedEngine)
        engine._model = None
        engine._tokenizer = None
        engine._loaded = False
        with pytest.raises(RuntimeError, match="not loaded"):
            engine.embed(["test"])


class TestBatchedEnginePool:
    """Tests for BatchedEngine.pool()."""

    def test_pool_mean(self):
        engine = _make_loaded_engine()
        result = engine.pool(["hello world"], pooling_type="MEAN")
        assert len(result) == 1
        assert len(result[0]) == 64
        # Mean of ones should be ones
        assert all(v == 1.0 for v in result[0])

    def test_pool_cls(self):
        engine = _make_loaded_engine()
        result = engine.pool(["hi"], pooling_type="CLS")
        assert len(result) == 1
        assert len(result[0]) == 64

    def test_pool_last(self):
        engine = _make_loaded_engine()
        result = engine.pool(["hi"], pooling_type="LAST")
        assert len(result) == 1
        assert len(result[0]) == 64

    def test_pool_empty_text_gives_zero_vector(self):
        engine = _make_loaded_engine()
        result = engine.pool([""], pooling_type="MEAN")
        assert len(result) == 1
        assert all(v == 0.0 for v in result[0])

    def test_pool_not_loaded_raises(self):
        engine = BatchedEngine.__new__(BatchedEngine)
        engine._model = None
        engine._tokenizer = None
        engine._loaded = False
        with pytest.raises(RuntimeError, match="not loaded"):
            engine.pool(["test"])


class TestBatchedEngineHiddenSize:
    """Tests for _get_hidden_size()."""

    def test_hidden_size_from_config(self):
        engine = _make_loaded_engine()
        assert engine._get_hidden_size() == 64

    def test_hidden_size_no_config(self):
        engine = BatchedEngine.__new__(BatchedEngine)
        engine._model = object()  # no config
        assert engine._get_hidden_size() == 768  # default fallback


class TestExtractHiddenStates:
    """Tests for _extract_hidden_states()."""

    def test_extract_from_plain_array(self):
        import mlx.core as mx

        engine = _make_loaded_engine()
        arr = mx.ones((1, 5, 64))
        result = engine._extract_hidden_states(arr)
        assert result.shape == (1, 5, 64)

    def test_extract_from_tuple(self):
        import mlx.core as mx

        engine = _make_loaded_engine()
        arr = mx.ones((1, 5, 64))
        result = engine._extract_hidden_states((arr, mx.zeros(10)))
        assert result.shape == (1, 5, 64)

    def test_extract_from_named_tuple(self):
        import mlx.core as mx

        engine = _make_loaded_engine()
        arr = mx.ones((1, 5, 64))
        result = engine._extract_hidden_states((arr,))
        assert result.shape == (1, 5, 64)
