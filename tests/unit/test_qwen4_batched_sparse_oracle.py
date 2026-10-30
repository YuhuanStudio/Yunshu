"""Check the proposed padded address arithmetic against upstream singleton QSA."""

import mlx.core as mx
import pytest
from mlx_vlm.models.qwen4_exp import qsa_kernel as upstream

from yunshu_engine.kernels.qwen4_qsa.attention import padded_sparse_attention


@pytest.mark.parametrize("dtype,dim", [(mx.float16, 64), (mx.bfloat16, 256)])
@pytest.mark.parametrize("batch,length", [(1, 2051), (2, 8195), (3, 8195), (4, 32771)])
def test_padded_sparse_matches_singletons(batch, length, dtype, dim):
    mx.random.seed(11)
    heads, kv_heads, topk, block = 4, 2, 512, 4
    pads = [i * 7 for i in range(batch)]
    q = mx.random.normal((batch, heads, 1, dim)).astype(dtype)
    k = mx.random.normal((batch, kv_heads, length, dim)).astype(dtype)
    v = mx.random.normal((batch, kv_heads, length, dim)).astype(dtype)
    indices = mx.broadcast_to(
        mx.arange(topk - 1, -1, -1, dtype=mx.int32), (batch, 1, topk)
    )
    indices = mx.where(indices % 97 == 0, -1, indices)
    ends = mx.full((batch, 1), length, dtype=mx.int32)
    scale = dim**-0.5
    got = padded_sparse_attention(
        q,
        k,
        v,
        indices,
        ends,
        mx.array(pads, dtype=mx.int32),
        scale=scale,
        block_size=block,
    )
    expected = mx.concatenate(
        [
            upstream.qsa_sparse_attention(
                q[i : i + 1],
                k[i : i + 1, :, pad:],
                v[i : i + 1, :, pad:],
                indices[i : i + 1],
                mx.array([[length - pad]], dtype=mx.int32),
                scale=scale,
                block_size=block,
                allow_sparse_decode=True,
            )
            for i, pad in enumerate(pads)
        ]
    )
    assert mx.array_equal(got, expected).item()
