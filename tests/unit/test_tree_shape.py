"""Window structure tables: host-built (TreeShape) vs GPU-built (DynamicShape)."""

from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from yunshu_engine import tree_verify as tv  # noqa: E402

SHAPES = [
    [-1],
    [-1, 0],
    [-1, 0, 1, 2, 3, 4, 5, 6],
    [-1, 0, 0, 1, 1, 2, 3, 3],
    [-1, 0, 0, 0, 1, 2, 3, 4],
    [-1, 0, 0, 1, 2, 3, 3, 5],
]


@pytest.mark.parametrize("parents", SHAPES)
def test_dynamic_tables_match_static(parents):
    static = tv.TreeShape(parents)
    dyn = tv.DynamicShape(mx.array(parents, dtype=mx.int32), static.max_depth + 2)
    assert dyn.depth_array().tolist() == static.depths
    # rows' ancestor paths agree up to each row's own depth
    sp, dp = np.array(static.path_table()), np.array(dyn.path_table())
    for row, depth in enumerate(static.depths):
        assert dp[row][: depth + 1].tolist() == sp[row][: depth + 1].tolist()
    assert dyn.conv_index().tolist() == static.conv_index().tolist()


def test_tree_shape_rejects_bad_parents():
    with pytest.raises(ValueError):
        tv.TreeShape([0, 0])
    with pytest.raises(ValueError):
        tv.TreeShape([-1, 1])
