"""Row-invariant lane projections (round driver): a row's bits do not depend
on how many rows share the call, including calls cut into 128-row pieces."""

import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

from yunshu_engine.kernels import lane_linear  # noqa: E402
from yunshu_engine.kernels.ragged_attention import tile_ready  # noqa: E402

pytestmark = pytest.mark.skipif(
    not tile_ready(), reason="lane matmul needs M5-class tensor ops"
)


@pytest.mark.parametrize("bits,n", [(4, 3072), (5, 3072), (8, 1024), (4, 48), (5, 16)])
def test_rows_invariant_to_row_count(bits, n):
    mx.random.seed(1)
    lin = nn.Linear(512, n, bias=False)
    lin.set_dtype(mx.bfloat16)
    q = nn.QuantizedLinear.from_linear(lin, group_size=64, bits=bits)
    assert lane_linear.eligible(q)
    lane = lane_linear.LaneLinear.from_quantized(q)
    x = mx.random.normal((300, 512)).astype(mx.bfloat16)
    full = lane(x)
    for m in (1, 2, 5, 16, 33, 128, 129, 300):
        assert mx.array_equal(lane(x[:m]), full[:m]).item(), m
    for i in (0, 77, 250):
        assert mx.array_equal(lane(x[i : i + 1]), full[i : i + 1]).item(), i
    # close to MLX's own quantized matmul (different summation order only)
    ref = q(x).astype(mx.float32)
    assert float(mx.abs(full.astype(mx.float32) - ref).max()) < 0.05


def test_convert_swaps_eligible_layers():
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(256, 128, bias=False)
            self.b = nn.Linear(256, 20, bias=False)  # N % 4 == 0
            self.c = nn.Linear(256, 30, bias=False)  # N % 4 != 0: stays

    m = Tiny()
    m.set_dtype(mx.bfloat16)
    nn.quantize(m, group_size=64, bits=4)
    out = lane_linear.convert(m)
    assert out["converted"] == 2
    assert isinstance(m.a, lane_linear.LaneLinear)
    assert isinstance(m.b, lane_linear.LaneLinear)
    assert isinstance(m.c, nn.QuantizedLinear)
    assert out["skipped"] == ["c"]


@pytest.mark.parametrize("n", [16, 48, 128])
def test_prefill_of_narrow_projections_is_row_invariant(n):
    """Stock quantized matmul picks its kernel by row count, so a narrow
    projection's prefill goes through the lane kernel: a token's output does
    not depend on the span it prefilled in."""
    mx.random.seed(2)
    lin = nn.Linear(512, n, bias=False)
    lin.set_dtype(mx.bfloat16)
    q = nn.QuantizedLinear.from_linear(lin, group_size=64, bits=4)
    lane = lane_linear.LaneLinear.from_quantized(q)
    x = mx.random.normal((1, 2048, 512)).astype(mx.bfloat16)
    full = lane.prefill([x])[0]
    for m in (37, 100, 512, 1024):
        assert mx.array_equal(lane.prefill([x[:, :m]])[0], full[:, :m]).item(), m
