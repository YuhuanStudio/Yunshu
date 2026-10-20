"""The draft lattice's two-stage top-k picks the same candidate set as one argpartition."""

import mlx.core as mx
import pytest

from yunshu_engine.dflash_tree import top_k_ids


@pytest.mark.parametrize("n", [100, 65536, 65536 + 37, 70000])
def test_same_set_as_argpartition(n):
    mx.random.seed(n)
    logits = mx.random.normal((15, n)).astype(mx.bfloat16)
    want = mx.argpartition(logits, -16, axis=-1)[..., -16:]
    got = top_k_ids(logits, 16)
    assert got.shape == (15, 16)
    assert int(got.max()) < n  # padding never selected
    assert [set(r) for r in got.tolist()] == [set(r) for r in want.tolist()]
