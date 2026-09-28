"""Row-exact verify (oMLX d403e460): verify rows equal one-row serial decode.

Small GPU checks (tiny shapes, seconds): each row of the row-exact projection
equals ``quantized_matmul`` of that row alone (4- and 5-bit), and the row-exact
causal attention equals a one-row SDPA call per row across MLX's 1024-key
plan switch. Also records whether the non-row-exact verify attention (the one
the batch-invariant mode uses) matches serial decode there.
"""

import mlx.core as mx
import mlx.nn as nn
import pytest

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="needs Metal")


def _qlinear(k, n, bits, seed):
    mx.random.seed(seed)
    lin = nn.Linear(k, n, bias=False)
    lin.weight = (mx.random.normal((n, k)) * 0.02).astype(mx.bfloat16)
    return nn.QuantizedLinear.from_linear(lin, group_size=64, bits=bits)


@pytest.mark.parametrize("bits", [4, 5])
@pytest.mark.parametrize("rows", [2, 5, 8])
def test_row_exact_projection_matches_one_row_decode(bits, rows):
    from yunshu_engine.kernels.omlx import row_exact_qmv

    lin = _qlinear(1024, 512, bits, seed=bits * 10 + rows)
    x = (mx.random.normal((1, rows, 1024)) * 0.5).astype(mx.bfloat16)
    out = row_exact_qmv.quantized_linear(lin, x)
    for r in range(rows):
        ref = lin(x[:, r : r + 1, :])
        assert mx.array_equal(out[:, r : r + 1, :], ref).item(), f"row {r}"


def _one_row_reference(q, k, v, scale, r):
    q_len, kv_len = q.shape[-2], k.shape[-2]
    end = kv_len - (q_len - 1 - r)
    return mx.fast.scaled_dot_product_attention(
        q[..., r : r + 1, :], k[..., :end, :], v[..., :end, :], scale=scale
    )


def _attention_inputs(kv_len, q_len=7, hq=6, hk=1, d=256, seed=0):
    mx.random.seed(seed)
    q = mx.random.normal((1, hq, q_len, d)).astype(mx.bfloat16)
    k = mx.random.normal((1, hk, kv_len, d)).astype(mx.bfloat16)
    v = mx.random.normal((1, hk, kv_len, d)).astype(mx.bfloat16)
    return q, k, v, d**-0.5


@pytest.mark.parametrize("kv_len", [600, 1027, 1030])
def test_row_exact_attention_matches_one_row_decode(kv_len):
    from yunshu_engine.kernels.omlx import qwen35_verify_sdpa_split as s

    q, k, v, scale = _attention_inputs(kv_len)
    limit = s._eligible(q, k, None)  # the vector-kernel row budget, as in use
    assert limit > 0
    out = s._row_exact_causal_sdpa(q, k, v, scale, limit=limit)
    for r in range(q.shape[-2]):
        ref = _one_row_reference(q, k, v, scale, r)
        assert mx.array_equal(out[..., r : r + 1, :], ref).item(), f"row {r}"


def test_non_row_exact_verify_attention_vs_serial_decode():
    """Evidence for the report: the verify attention the batch-invariant mode
    uses (gqa / chunked kernels) is not guaranteed bit-equal to serial decode.
    This test does not fail on a mismatch; it records the count."""
    from yunshu_engine.kernels.omlx import qwen35_verify_sdpa_split as s

    mismatches = {}
    for kv_len in (600, 1030, 4200):
        q, k, v, scale = _attention_inputs(kv_len, seed=1)
        paths = {
            "chunked": s._chunked_causal_sdpa(q, k, v, scale, s._eligible(q, k, None))
        }
        if s._gqa_ready():
            paths["gqa"] = s._gqa_causal_sdpa(q, k, v, scale)
        for name, out in paths.items():
            bad = sum(
                not mx.array_equal(
                    out[..., r : r + 1, :], _one_row_reference(q, k, v, scale, r)
                ).item()
                for r in range(q.shape[-2])
            )
            mismatches[(name, kv_len)] = bad
    print("non-row-exact verify rows != serial decode:", mismatches)
