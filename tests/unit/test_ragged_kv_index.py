"""Scatter index of the ragged KV write: int64 once the buffer's rows pass int32."""

import pytest

mx = pytest.importorskip("mlx.core")

if not mx.metal.is_available():  # pragma: no cover - CI without Apple GPU
    pytest.skip("needs an Apple GPU", allow_module_level=True)

from yunshu_engine.kernels.ragged_kv import token_row_index  # noqa: E402


def test_small_buffer_keeps_int32_index():
    idx = token_row_index((0, 1), (5, 9), 4, 64, 2)
    assert idx.dtype == mx.int32
    assert idx.tolist()[1][2] == [(1 * 4 + 2) * 64 + 9 + t for t in (0, 1)]


def test_index_past_int32_is_int64_and_exact():
    H, cap, slot = 8, 1 << 20, 300  # (300 * 8 + 7) * 2**20 rows: past 2**31
    idx = token_row_index((slot,), (1000,), H, cap, 1)
    assert idx.dtype == mx.int64
    assert idx.tolist()[0][7][0] == (slot * H + 7) * cap + 1000
