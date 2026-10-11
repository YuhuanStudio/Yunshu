from types import SimpleNamespace as NS

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
from yunshu_engine import ple_scale  # noqa: E402


class Table:
    def __init__(self):
        self.calls = 0

    def __call__(self, indices):
        self.calls += 1
        return mx.ones((*indices.shape, 4), dtype=mx.bfloat16) * 3

    lookup = __call__


def _lm(table):
    layer = NS(ple=NS(ple_embedding=NS(ngram_embedding=table)))
    return NS(model=NS(layers=[NS(), layer]))


def test_collect_scales_keys():
    w = {
        "language_model.model.layers.1.ple.ple_embedding.ngram_embedding.weight_scale": mx.array(
            [0.5], dtype=mx.bfloat16
        ),
        "language_model.model.layers.1.mlp.gate.weight": mx.zeros((2, 2)),
    }
    assert ple_scale.collect_table_scales(w) == {1: 0.5}
    with pytest.raises(ValueError):
        ple_scale.collect_table_scales(
            {"layers.2.ple.ple_embedding.ngram_embedding.weight_scale": mx.zeros((2,))}
        )


def test_scale_applied_once_in_float32_and_identity_for_one():
    table = Table()
    lm = _lm(table)
    assert ple_scale.apply_table_scales(lm, {1: 1.0}) == 0
    assert np.array_equal(
        np.array(table(mx.zeros((2,), dtype=mx.int32)).astype(mx.float32)),
        np.full((2, 4), 3.0),
    )
    assert ple_scale.apply_table_scales(lm, {1: 0.25}) == 1
    out = table(mx.zeros((2,), dtype=mx.int32))
    assert out.dtype == mx.bfloat16
    assert np.array_equal(np.array(out.astype(mx.float32)), np.full((2, 4), 0.75))
    assert table.lookup(mx.zeros((1,), dtype=mx.int32)).shape == (1, 4)
    # re-applying changes the factor, never stacks
    ple_scale.apply_table_scales(lm, {1: 2.0})
    assert (
        np.array(table(mx.zeros((1,), dtype=mx.int32)).astype(mx.float32))[0, 0] == 6.0
    )
