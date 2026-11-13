"""The GDN chain window's TreeShape is shared across rows and steps (not rebuilt per row per layer),
and the shape handed to the layer is a chain of the same width."""

from types import SimpleNamespace

import mlx.core as mx
import pytest

from yunshu_engine import tree_verify as tv
from yunshu_engine.round_driver import batch


def _stub_gdn(monkeypatch):
    shapes = []

    def fake_layer(verifier, g, x, cache, shape, *, projected, project_output):
        shapes.append(shape)
        return x, None

    monkeypatch.setattr(tv, "_gdn_layer", fake_layer)
    stub = SimpleNamespace(conv=[[None, None]], state=[[None, None]])
    verifier = SimpleNamespace(
        _linears=lambda ws, xn: tuple(xn for _ in ws), _linear=lambda lin, x: x
    )
    g = SimpleNamespace(
        in_proj_qkv=None, in_proj_z=None, in_proj_b=None, in_proj_a=None, out_proj=None
    )
    return stub, verifier, g, shapes


def test_gdn_shares_one_chain_shape_across_rows_and_steps(monkeypatch):
    stub, verifier, g, shapes = _stub_gdn(monkeypatch)
    xn = mx.zeros((1, 6, 4))
    recs = [[None], [None]]
    for _ in range(3):
        batch.DecodeBatch._gdn(stub, verifier, g, xn, 0, [3, 3], recs)
    assert len({id(s) for s in shapes}) == 1


@pytest.mark.parametrize("width", [1, 2, 5, 8])
def test_gdn_chain_shape_parents_are_a_chain(monkeypatch, width):
    stub, verifier, g, shapes = _stub_gdn(monkeypatch)
    xn = mx.zeros((1, width, 4))
    batch.DecodeBatch._gdn(stub, verifier, g, xn, 0, [width], [[None]])
    shape = shapes[0]
    assert shape.parents == tuple(range(-1, width - 1))
    assert shape.is_chain and shape.width == width and shape.max_depth == width - 1
