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


def test_split_merge_reads_each_group_from_its_own_partials():
    from yunshu_engine import tree_verify as tv

    old, new = tv._MERGE2, tv._MERGE_SPLIT
    for name in ("PMA", "PLA", "POA"):
        assert new.count(name + "1[") == 1
        assert old.count(name + "[") == new.count(name + "[") == 1
    assert "node >= (uint)TG" in new
    # the accumulate order is the original's: nothing else changed
    stripped = (
        new.replace("const bool g1 = node >= (uint)TG;\n", "")
        .replace("(g1 ? PMA1[row] : PMA[row])", "PMA[row]")
        .replace("(g1 ? PLA1[row] : PLA[row])", "PLA[row]")
        .replace(
            "(g1 ? POA1[row * D + lane * DPL + i] : POA[row * D + lane * DPL + i])",
            "POA[row * D + lane * DPL + i]",
        )
    )
    assert stripped.split() == old.split()


def test_only_the_fast_shape_takes_the_fused_glue():
    from yunshu_engine.dflash_plan import FastShape

    assert FastShape.fast_glue is True
    assert not getattr(tv_shape_classes(), "fast_glue", False)


def tv_shape_classes():
    from yunshu_engine import tree_verify as tv

    return tv.DynamicShape
