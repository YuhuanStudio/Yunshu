"""5-bit (and unchanged 4-bit) NAX packed projections vs mx.quantized_matmul."""

import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

from yunshu_engine.kernels.omlx import is_nax_available  # noqa: E402

pytestmark = pytest.mark.skipif(
    not is_nax_available(), reason="needs an M5-class GPU with NAX kernels"
)


@pytest.fixture(autouse=True)
def _enable_5bit(monkeypatch):
    # 5-bit packing is opt-in (YUNSHU_PACKED_5BIT=1); these tests exercise it.
    from yunshu_engine.kernels.omlx import qwen35_packed_linear as pl

    monkeypatch.setattr(pl, "PACKED_BITS", (4, 5))


def _linear(K, N, bits, seed=0):
    mx.random.seed(seed)
    lin = nn.QuantizedLinear(K, N, bias=False, group_size=64, bits=bits)
    w = mx.random.normal((N, K)) * 0.05
    q, s, b = mx.quantize(w, group_size=64, bits=bits)
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


def _packed(lin):
    from yunshu_engine.kernels.omlx import qwen35_packed_linear as pl

    assert pl.eligible(lin)
    (packed,) = pl._pack([lin])
    return packed


@pytest.mark.parametrize("bits", [4, 5])
@pytest.mark.parametrize("K,N", [(2048, 1024), (4096, 2048)])
@pytest.mark.parametrize("rows", [1, 2, 4, 8, 16, 64])
def test_packed_matches_quantized_matmul(bits, K, N, rows):
    lin = _linear(K, N, bits)
    packed = _packed(lin)
    mx.random.seed(1)
    x = (mx.random.normal((rows, K)) * 0.5).astype(mx.bfloat16)
    ref = _reference(lin, x)
    out = packed(x).astype(mx.float32)
    err = mx.abs(out - ref)
    scale = mx.maximum(mx.abs(ref), 1.0)
    # bf16 output rounding plus fp32 accumulation order.
    assert float(mx.max(err / scale)) < 2e-2


@pytest.mark.parametrize("bits", [4, 5])
def test_packed_rows_2_to_8_are_row_invariant(bits):
    lin = _linear(2048, 1024, bits, seed=3)
    packed = _packed(lin)
    mx.random.seed(4)
    x = (mx.random.normal((8, 2048)) * 0.5).astype(mx.bfloat16)
    full = packed(x)
    for rows in range(2, 9):
        part = packed(x[:rows])
        assert mx.array_equal(part, full[:rows]).item(), rows


def test_pack_layer_splits_runs_by_bit_width():
    from types import SimpleNamespace

    from yunshu_engine.kernels.omlx import qwen35_packed_linear as pl

    mlp = SimpleNamespace(
        gate_proj=_linear(2048, 1024, 4, seed=5),
        up_proj=_linear(2048, 1024, 5, seed=6),
        down_proj=_linear(1024, 2048, 5, seed=7),
    )
    layer = SimpleNamespace(mlp=mlp)
    assert pl._pack_layer(layer) == 3
    assert isinstance(mlp.gate_proj, pl.PackedLinear) and mlp.gate_proj.bits == 4
    assert isinstance(mlp.up_proj, pl.PackedLinear) and mlp.up_proj.bits == 5
    # Different widths never share a store (one code layout per store).
    assert mlp.gate_proj._store is not mlp.up_proj._store
    x = (mx.random.normal((3, 2048)) * 0.5).astype(mx.bfloat16)
    assert pl.project([mlp.gate_proj, mlp.up_proj], x) is None


def test_quantized_rows_roundtrip_5bit():
    lin = _linear(2048, 1024, 5, seed=8)
    packed = _packed(lin)
    ids = mx.array([0, 5, 130, 1023])
    w, s, b = packed.quantized_rows(ids)
    assert mx.array_equal(w, lin.weight[ids]).item()
    assert mx.array_equal(s, lin.scales[ids]).item()
    assert mx.array_equal(b, lin.biases[ids]).item()


def test_5bit_packing_is_opt_in_by_default(monkeypatch):
    from yunshu_engine.kernels.omlx import qwen35_packed_linear as pl

    monkeypatch.delenv("YUNSHU_PACKED_5BIT", raising=False)
    assert pl._packed_bits_from_env() == (4,)
    monkeypatch.setenv("YUNSHU_PACKED_5BIT", "1")
    assert pl._packed_bits_from_env() == (4, 5)
    monkeypatch.setattr(pl, "PACKED_BITS", (4,))
    assert not pl.eligible(_linear(2048, 1024, 5))
    assert pl.eligible(_linear(2048, 1024, 4))
