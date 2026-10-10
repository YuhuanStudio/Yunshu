"""Stacked drafter projections equal the separate ones column for column."""

import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

from yunshu_engine.dflash_tree import _stack_linears, _stacked_mlp  # noqa: E402


def _linear(n, k, bits, seed):
    mx.random.seed(seed)
    lin = nn.Linear(k, n, bias=False)
    mx.eval(lin.parameters())
    return (
        nn.QuantizedLinear.from_linear(lin, group_size=64, bits=bits) if bits else lin
    )


@pytest.mark.parametrize("bits", [8, 4, 0])
def test_stacked_projection_equals_the_two_separate_ones(bits):
    a, b = _linear(96, 128, bits, 1), _linear(96, 128, bits, 2)
    stacked = _stack_linears(a, b)
    x = mx.random.normal((1, 5, 128)).astype(mx.bfloat16)
    got = stacked(x)
    want = mx.concatenate([a(x), b(x)], axis=-1)
    assert mx.allclose(
        got.astype(mx.float32), want.astype(mx.float32), atol=2e-2
    ).item()


def test_mismatched_bits_are_not_stacked():
    assert _stack_linears(_linear(64, 128, 8, 1), _linear(64, 128, 4, 2)) is None


def test_stacked_mlp_matches_the_gated_mlp():
    from mlx_vlm.speculative.drafters.qwen3_dflash.dflash import Qwen3MLP

    mx.random.seed(5)
    mlp = Qwen3MLP(128, 192)
    mx.eval(mlp.parameters())
    x = mx.random.normal((1, 3, 128))
    want = mlp(x)
    stacked = _stacked_mlp(mlp, _stack_linears(mlp.gate_proj, mlp.up_proj))
    assert mx.allclose(stacked(x), want, atol=1e-4).item()
