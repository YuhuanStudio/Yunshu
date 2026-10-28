"""Lane eligibility rejects foreign cache layouts without reading dense KV fields."""

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("mlx.core")

from yunshu_engine import tree_verify  # noqa: E402
from yunshu_engine.kernels import ragged_kv  # noqa: E402


@pytest.mark.parametrize("rows", [1, 2, 4])
def test_qsa_cache_is_not_a_dense_lane(rows):
    # Flash-Next keeps dense K/V inside kv_cache and also owns index keys.
    # Unwrapping only the inner cache would silently discard QSA semantics.
    class BatchQSAKVCache:
        def __init__(self):
            self.kv_cache = SimpleNamespace(keys=np.zeros((rows, 2, 8, 4)))
            self.index_keys = None

    cache = BatchQSAKVCache()
    lm = SimpleNamespace(model=SimpleNamespace(fa_idx=0))
    assert ragged_kv._lane_length(cache) is None
    assert not tree_verify.lane_ready(lm, [cache])


def test_lane_probe_does_not_read_unknown_cache_keys():
    class ForeignCache:
        @property
        def keys(self):
            raise AssertionError("unknown cache layout must be rejected first")

    assert ragged_kv._lane_length(ForeignCache()) is None


def test_lane_probe_preserves_stock_dense_cache(monkeypatch):
    class DenseCache:
        keys = np.zeros((1, 2, 8, 4))
        offset = 7

    monkeypatch.setattr(ragged_kv, "_dense_kv_classes", lambda: (DenseCache,))
    assert ragged_kv._lane_length(DenseCache()) == 7
    assert ragged_kv._lane_length(None) is None
    cache = DenseCache()
    cache.keys = None
    assert ragged_kv._lane_length(cache) is None
    cache.keys = np.zeros((2, 2, 8, 4))
    assert ragged_kv._lane_length(cache) is None
