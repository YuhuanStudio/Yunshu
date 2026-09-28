"""TensorFold integer-code tensor-unit projections (5/6/8-bit) vs mx.quantized_matmul."""

import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

from yunshu_engine.kernels.omlx import is_nax_available  # noqa: E402

pytestmark = pytest.mark.skipif(
    not is_nax_available(), reason="needs an M5-class GPU with tensor units"
)


def _linear(K, N, bits, seed=0):
    mx.random.seed(seed)
    lin = nn.QuantizedLinear(K, N, bias=False, group_size=64, bits=bits)
    q, s, b = mx.quantize(mx.random.normal((N, K)) * 0.05, group_size=64, bits=bits)
    lin.weight, lin.scales, lin.biases = q, s.astype(mx.bfloat16), b.astype(mx.bfloat16)
    mx.eval(lin.parameters())
    return lin


def _reference(lin, x):
    return mx.quantized_matmul(
        x.astype(mx.float32),
        lin.weight,
        lin.scales.astype(mx.float32),
        lin.biases.astype(mx.float32),
        transpose=True,
        group_size=64,
        bits=lin.bits,
    )


@pytest.mark.parametrize("bits", [5, 6, 8])
@pytest.mark.parametrize("rows", [1, 2, 4, 8, 16, 64])
def test_int_code_matches_quantized_matmul(bits, rows):
    from yunshu_engine.kernels.int_code_linear import IntCodeLinear

    lin = _linear(4096, 2048, bits)
    ic = IntCodeLinear(lin)
    mx.random.seed(1)
    x = (mx.random.normal((rows, 4096)) * 0.5).astype(mx.bfloat16)
    ref = _reference(lin, x)
    y = ic(x).astype(mx.float32)
    rel = (mx.max(mx.abs(y - ref)) / mx.max(mx.abs(ref))).item()
    # bf16 output rounding; the same bound the packed 4/5-bit tests use.
    assert rel < 0.02, rel


@pytest.mark.parametrize("bits", [5, 8])
def test_int_code_rows_are_row_invariant(bits, monkeypatch):
    """Rows 1..16 share one 16-row op: a row's bits never depend on the row count
    (the speculative lane relies on it for 1 decode row and 2..8 verify rows)."""
    from yunshu_engine.kernels import batch_invariant
    from yunshu_engine.kernels.int_code_linear import IntCodeLinear

    monkeypatch.setitem(batch_invariant._STATE, "active", True)
    ic = IntCodeLinear(_linear(2048, 1024, bits, seed=3))
    mx.random.seed(4)
    x = (mx.random.normal((16, 2048)) * 0.5).astype(mx.bfloat16)
    full = ic(x)
    for rows in range(1, 17):
        assert mx.array_equal(ic(x[:rows]), full[:rows]).item(), rows
    # A single row equals a 2-row call's first row without padding.
    assert mx.array_equal(ic(x[:1]), ic(x[:2])[:1]).item()


def test_int_code_single_row_outside_lane_uses_stock(monkeypatch):
    from yunshu_engine.kernels import batch_invariant
    from yunshu_engine.kernels.int_code_linear import IntCodeLinear

    monkeypatch.setitem(batch_invariant._STATE, "active", False)
    lin = _linear(2048, 1024, 5, seed=11)
    ic = IntCodeLinear(lin)
    x = (mx.random.normal((1, 2048)) * 0.5).astype(mx.bfloat16)
    assert mx.array_equal(ic(x), lin(x)).item()


def test_int_code_large_prompts_use_stock_matmul():
    from yunshu_engine.kernels.int_code_linear import IntCodeLinear
    from yunshu_engine.kernels.tensorfold import lane_qmm

    lin = _linear(1024, 512, 5, seed=5)
    ic = IntCodeLinear(lin)
    x = (mx.random.normal((lane_qmm.MAX_ROWS + 1, 1024)) * 0.5).astype(mx.bfloat16)
    assert mx.array_equal(ic(x), lin(x)).item()


def test_int_code_quantized_rows():
    from yunshu_engine.kernels.int_code_linear import IntCodeLinear

    lin = _linear(2048, 1024, 5, seed=6)
    ic = IntCodeLinear(lin)
    ids = mx.array([0, 7, 511, 1023])
    w, s, b = ic.quantized_rows(ids)
    assert mx.array_equal(w, lin.weight[ids]).item()
    assert mx.array_equal(s, lin.scales[ids]).item()
    assert mx.array_equal(b, lin.biases[ids]).item()


def test_pack_layer_serves_5bit_with_int_codes():
    from types import SimpleNamespace

    from yunshu_engine.kernels.int_code_linear import IntCodeLinear
    from yunshu_engine.kernels.omlx import qwen35_packed_linear as pl

    mlp = SimpleNamespace(
        gate_proj=_linear(2048, 1024, 4, seed=7),
        up_proj=_linear(2048, 1024, 4, seed=8),
        down_proj=_linear(1024, 2048, 5, seed=9),
    )
    x = (mx.random.normal((3, 1024)) * 0.5).astype(mx.bfloat16)
    before = mlp.down_proj(x)
    assert pl._pack_layer(SimpleNamespace(mlp=mlp)) == 3
    assert isinstance(mlp.gate_proj, pl.PackedLinear)
    assert isinstance(mlp.down_proj, IntCodeLinear) and mlp.down_proj.bits == 5
    err = mx.max(mx.abs(mlp.down_proj(x) - before)) / mx.max(mx.abs(before))
    assert err.item() < 0.02


def test_int_mode_is_default():
    from yunshu_engine.kernels.omlx import qwen35_packed_linear as pl

    assert pl.int_mode() == "int"
    # 5-bit layers are not repacked by the 4-bit packed kernels.
    assert not pl.eligible(_linear(2048, 1024, 5))
