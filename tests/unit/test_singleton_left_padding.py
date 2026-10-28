"""extract_rows reads left_padding once per array, not once per layer per step."""

import mlx.core as mx
from mlx_vlm.models.cache import BatchKVCache

from yunshu_engine.kernels import singleton_cache as sc


class Pad:
    """An array stand-in that counts device reads."""

    size = 1

    def __init__(self, value):
        self.value, self.reads = value, 0

    def item(self):
        self.reads += 1
        return self.value


def _cache(pad):
    kv = BatchKVCache([0])
    kv.update_and_fetch(mx.ones((1, 1, 32, 4)), mx.ones((1, 1, 32, 4)))
    kv.left_padding = pad
    return kv


def test_zero_padding_is_read_once_across_steps():
    pad = Pad(0)
    kv = _cache(pad)
    for _ in range(5):
        assert sc.extract_rows([kv]) is not None
    assert pad.reads == 1


def test_replaced_padding_is_read_again_and_nonzero_falls_back():
    kv = _cache(Pad(0))
    assert sc.extract_rows([kv]) is not None
    new = Pad(3)
    kv.left_padding = new
    assert sc.extract_rows([kv]) is None
    assert sc.extract_rows([kv]) is None
    assert new.reads == 1
