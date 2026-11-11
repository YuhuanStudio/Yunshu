"""Prepared GDN replay has the same bits as host-path replay on production tiny kernels."""

import mlx.core as mx
import pytest

from yunshu_engine import tree_verify as tv

from .test_round_driver_dflash import restore_probe_lane_mode, target  # noqa: F401
from .test_round_driver_one_forward import _joined


@pytest.mark.parametrize("path", [[0], [0, 1], [0, 2, 5], [0, 1, 3, 6, 7]])
@pytest.mark.usefixtures("restore_probe_lane_mode")
def test_prepared_gdn_commit_matches_host_commit_bits(target, path):  # noqa: F811
    drivers = [_joined(target, plain_verify=True) for _ in range(2)]
    shape = tv.TreeShape([-1, 0, 0, 1, 1, 2, 3, 6])
    tokens = mx.array(
        [[d.rows[0].pending for d in drivers[:1]] + [11, 12, 13, 14, 15, 16, 17]],
        dtype=mx.int32,
    )
    caches = [d.rows[0].cache for d in drivers]
    results = [tv.tree_forward(target, tokens, shape, c) for c in caches]
    prepared = tv.prepare_tree_commit(
        results[1],
        mx.array(path + [0] * (shape.width - len(path)), dtype=mx.int32),
        mx.array([len(path)], dtype=mx.int32),
    )
    assert prepared
    mx.eval(*[a for pair in prepared.values() for a in pair])
    tv.tree_commit(target, caches[0], results[0], path)
    tv.tree_commit(target, caches[1], results[1], path, prepared=prepared)
    for a, b in zip(caches[0], caches[1], strict=True):
        if hasattr(a, "keys"):
            assert a.offset == b.offset
            for name in ("keys", "values"):
                assert mx.array_equal(
                    getattr(a, name)[..., : a.offset, :],
                    getattr(b, name)[..., : b.offset, :],
                )
        else:
            assert mx.array_equal(a[0], b[0])
            assert mx.array_equal(a[1], b[1])
