"""Row 339: the tile kernel multiplied every loaded V row by its probability, masked keys
included, so a NaN past a row's length reached the output (0 * NaN). The V tile now ends at
the chunk's last key (zeros past it)."""

import pytest

mx = pytest.importorskip("mlx.core")

if not mx.metal.is_available():  # pragma: no cover - CI without Apple GPU
    pytest.skip("needs an Apple GPU", allow_module_level=True)

from yunshu_engine.kernels.ragged_attention import ragged_decode_attention, tile_ready  # noqa: E402

H, HKV, D = 24, 4, 256


@pytest.mark.parametrize("T", [1, 8])
def test_tile_output_ignores_nan_past_row_length(T):
    if not tile_ready():
        pytest.skip("tensor-op tile kernel unavailable on this GPU")
    lengths = [1030, 37, 512, 513, 1024]
    mx.random.seed(40 + T)
    cap = -(-max(lengths) // 256) * 256 + 256
    B = len(lengths)
    k = mx.random.normal((B, HKV, cap, D)).astype(mx.bfloat16)
    v = mx.random.normal((B, HKV, cap, D)).astype(mx.bfloat16)
    q = mx.random.normal((B, H, T, D)).astype(mx.bfloat16)
    pos = mx.arange(cap)[None, None, :, None]
    v = mx.where(
        pos >= mx.array(lengths)[:, None, None, None],
        mx.array(float("nan")).astype(mx.bfloat16),
        v,
    )
    out = ragged_decode_attention(q, k, v, mx.array(lengths), D**-0.5, impl="tile")
    assert not mx.isnan(out.astype(mx.float32)).any().item()
