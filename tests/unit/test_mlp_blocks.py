"""MLP blocks (M4): the structure choice, and that each block's methods make the calls the layers made before M4.

CPU only: the MoE per-token call equality on the tiny sparse block runs on Metal (test_mlp_moe_metal.py)."""

from types import SimpleNamespace

import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")
pytest.importorskip("mlx_vlm")

from yunshu_engine.blocks import mlp as blocks  # noqa: E402
from yunshu_engine.round_driver import forward as fwd  # noqa: E402


@pytest.fixture(autouse=True)
def cpu():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


class Dense(nn.Module):
    def __init__(self, d=8, h=16):
        super().__init__()
        self.gate_proj = nn.Linear(d, h, bias=False)
        self.up_proj = nn.Linear(d, h, bias=False)
        self.down_proj = nn.Linear(h, d, bias=False)

    def __call__(self, x):
        from mlx_vlm.models.qwen3_5.language import swiglu

        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class Routed(nn.Module):
    """A MoE-shaped block: what matters is that it is called once per segment."""

    def __init__(self):
        super().__init__()
        self.switch_mlp = nn.Linear(4, 4, bias=False)
        self.calls = []

    def __call__(self, x):
        self.calls.append(tuple(x.shape))
        return self.switch_mlp(x) * 2.0


def test_block_for_picks_by_structure_not_by_name():
    assert isinstance(blocks.block_for(Dense()), blocks.DenseMLP)
    assert isinstance(blocks.block_for(Routed()), blocks.RoutedMoE)
    # a module with a router and experts is routed whatever it is called
    assert isinstance(
        blocks.block_for(SimpleNamespace(gate=1, switch_mlp=2, shared_expert=3)),
        blocks.RoutedMoE,
    )
    assert blocks.routed(SimpleNamespace(switch_mlp=1)) and not blocks.routed(Dense())


def test_dense_packed_is_the_swiglu_of_the_projections():
    m = Dense()
    x = mx.arange(1 * 9 * 8, dtype=mx.float32).reshape(1, 9, 8) / 40
    spans = [(0, 3), (3, 4), (4, 9)]
    out = blocks.DenseMLP(m).packed(x, spans, [False, True, False])
    # the per-segment projections of the old forward._mlp dense branch, same bits as the module on each segment
    ref = mx.concatenate([m(x[:, s:e]) for s, e in spans], axis=1)
    assert mx.allclose(out, ref, atol=1e-5).item()


def test_routed_packed_runs_each_segment_on_its_own_rows():
    m = Routed()
    x = mx.arange(1 * 9 * 4, dtype=mx.float32).reshape(1, 9, 4) / 10
    spans = [(0, 3), (3, 4), (4, 9)]
    out = blocks.RoutedMoE(m).packed(x, spans, [False, True, False])
    assert m.calls == [(1, 3, 4), (1, 1, 4), (1, 5, 4)]
    assert mx.array_equal(out, fwd._mlp(m, x, spans, [False, True, False]))


def test_routed_single_span_is_the_flat_call():
    m = Routed()
    x = mx.arange(1 * 6 * 4, dtype=mx.float32).reshape(1, 6, 4) / 10
    out = blocks.RoutedMoE(m).packed(x, [(0, 6)], [False])
    assert mx.array_equal(out, m(x))


class Verifier:
    def __init__(self):
        self.calls = []

    def _feed_forward(self, mlp, x):
        self.calls.append((mlp, tuple(x.shape)))
        return x * 3


@pytest.mark.parametrize("mlp", [Dense(), Routed()], ids=["dense", "routed"])
def test_verified_is_the_verifier_feed_forward(mlp):
    v = Verifier()
    x = mx.ones((1, 2, 8 if isinstance(mlp, Dense) else 4))
    out = blocks.block_for(mlp).verified(v, x)
    assert v.calls == [(mlp, tuple(x.shape))]
    assert mx.array_equal(out, x * 3)


@pytest.mark.parametrize("mlp", [Dense(), Routed()], ids=["dense", "routed"])
def test_fused_attempt_calls_the_moe_entry_for_every_layer(monkeypatch, mlp):
    # a dense layer has no fused MoE, but its plan check still runs (and records) as it always did
    from yunshu_engine.kernels import tf_hc_forward

    seen = []
    monkeypatch.setattr(tf_hc_forward, "_moe", lambda m, x: seen.append(m) or None)
    x = mx.ones((2, 8))
    assert blocks.block_for(mlp).fused(x) is None
    assert seen == [mlp]
