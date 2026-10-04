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


def test_shared_plan_serves_both_groups_in_one_launch():
    from yunshu_engine import tree_verify as tv

    ck = tv.CK
    lengths, nc, items, starts, total = tv.shared_plan_lists(2 * ck, 2)
    # each group's length keeps its real tokens' causal limit (length - (7 - t))
    assert lengths == [2 * ck + 7, 2 * ck + 15]
    assert nc == 2  # shared chunks only; the merge never reads the partial third
    # chunk-major, group-minor: both groups of a chunk are adjacent items
    assert items == [0, 2, 1, 3]
    # slot starts per (group, token): contiguous, node index = group * 8 + token
    assert len(starts) == 16 and starts == [2 * i for i in range(16)]
    assert total == 32
    # one group: nothing from the second
    lengths1, _, items1, starts1, _ = tv.shared_plan_lists(2 * ck, 1)
    assert lengths1 == [2 * ck + 7] and len(starts1) == 8 and items1 == [0, 1]


def test_only_the_fast_shape_takes_the_fused_glue():
    from yunshu_engine.dflash_plan import FastShape

    assert FastShape.fast_glue is True
    assert not getattr(tv_shape_classes(), "fast_glue", False)


def tv_shape_classes():
    from yunshu_engine import tree_verify as tv

    return tv.DynamicShape
