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


@pytest.mark.parametrize("nodes", [1, 7, 15])
def test_compiled_search_tree_equals_the_eager_one(nodes):
    from yunshu_engine import dflash_tree as dt

    d, k, r = 15, 16, 32
    mx.random.seed(nodes)
    lat = dt.GpuLattice(
        mx.random.randint(0, 65536, (d, k)).astype(mx.uint32),
        mx.random.normal((d, k)) * 3,
        mx.random.normal((d, r)) * 0.1,
        mx.random.normal((d, k, r)) * 0.1,
        mx.random.normal((d, k, r)) * 0.1,
        mx.random.normal((r,)) * 0.1,
    )
    want = dt._search_tree(lat, nodes)
    got = dt.search_tree(lat, nodes)
    assert [a.tolist() for a in got] == [b.tolist() for b in want]
