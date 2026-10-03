"""Row-invariant lane projections (round driver): a row's bits do not depend
on how many rows share the call, including calls cut into 128-row pieces."""

import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

from yunshu_engine.kernels import lane_linear  # noqa: E402
from yunshu_engine.kernels.tensorfold.lane_qmm import ready  # noqa: E402

pytestmark = pytest.mark.skipif(
    not ready(),
    reason="lane tensor-op arithmetic self-test failed on this GPU / Metal compiler",
)


@pytest.mark.parametrize(
    "bits,n",
    [(4, 3072), (5, 3072), (8, 1024), (4, 48), (5, 16), (2, 48), (3, 16), (6, 48)],
)
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


@pytest.mark.parametrize("n", [3072, 48])
def test_quantized_rows_are_mlx_layout_rows(n):
    from yunshu_engine.draft_vocab import DraftVocab

    mx.random.seed(2)
    lin = nn.Linear(512, n, bias=False)
    lin.set_dtype(mx.bfloat16)
    q = nn.QuantizedLinear.from_linear(lin, group_size=64, bits=4)
    lane = lane_linear.LaneLinear.from_quantized(q)
    ids = mx.array([0, 5, 31, 32, 33, n - 1], dtype=mx.uint32)
    for got, want in zip(
        lane.quantized_rows(ids),
        (q.weight[ids], q.scales[ids], q.biases[ids]),
        strict=True,
    ):
        assert mx.array_equal(got, want).item()
    # the draft vocabulary over a lane head drafts what it drafts over the original
    x = mx.random.normal((3, 512)).astype(mx.bfloat16)
    a, b = DraftVocab(q, n // 2), DraftVocab(lane, n // 2)
    for v in (a, b):
        v.set_context([n // 2 + 1, n - 2] * 10)
    assert mx.array_equal(a.argmax(x), b.argmax(x)).item()


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


def test_prefill_chunks_run_stock_matmul_only_when_enabled():
    mx.random.seed(4)
    lin = nn.Linear(512, 3072, bias=False)
    lin.set_dtype(mx.bfloat16)
    q = nn.QuantizedLinear.from_linear(lin, group_size=64, bits=4)
    lane = lane_linear.LaneLinear.from_quantized(q)
    x = mx.random.normal((300, 512)).astype(mx.bfloat16)
    lane_out = lane(x)
    assert lane_linear.STOCK_ROWS == 0
    assert lane_linear.prefill_kernel_id() == "lane-qmm"
    lane_linear.set_stock_rows(128)
    try:
        assert lane_linear.prefill_kernel_id() == "stock-qmm-gt128"
        stock_out = lane(x)
        # above the threshold: MLX's own matmul on the same rows
        assert mx.array_equal(stock_out, q(x)).item()
        assert not mx.array_equal(stock_out, lane_out).item()
        # at or below it (decode / verify / short tails): still the lane kernel
        assert mx.array_equal(lane(x[:128]), lane_out[:128]).item()
        assert mx.array_equal(lane(x[:5]), lane_out[:5]).item()
    finally:
        lane_linear.set_stock_rows(0)
    assert mx.array_equal(lane(x), lane_out).item()


def test_wide_prefill_does_not_expand_other_lane_callers_guard():
    from yunshu_engine.kernels.tensorfold import lane_qmm

    assert lane_qmm.MAX_ROWS == 128
    assert lane_linear.PIECE == 512
    x = mx.zeros((129, 512), mx.bfloat16)
    weight = mx.zeros((32, 64), mx.uint32)
    sbt = mx.zeros((8, 32, 2), mx.bfloat16)
    with pytest.raises(ValueError, match="at most 128 rows"):
        lane_qmm.lane_matmul(x, weight, sbt)
    with pytest.raises(ValueError, match="row_limit"):
        lane_qmm.lane_matmul(x, weight, sbt, row_limit=1024)
    actual = lane_qmm.lane_matmul(x, weight, sbt, row_limit=512)
    assert actual.shape == (129, 32)
