"""Padded batched variant of the upstream QSA sparse decode kernel."""

from functools import lru_cache

import mlx.core as mx

from .source import SOURCE


@lru_cache(maxsize=1)
def _kernel():
    return mx.fast.metal_kernel(
        name="yunshu_qwen4_padded_qsa",
        input_names=[
            "queries",
            "keys",
            "values",
            "block_indices",
            "query_ends",
            "scale",
            "k_size",
            "left_padding",
        ],
        output_names=["out"],
        header="#include <metal_simdgroup>\nusing namespace metal;\n",
        source=SOURCE,
    )


def padded_sparse_attention(
    queries, keys, values, block_indices, query_ends, left_padding, *, scale, block_size
):
    batch, heads, q_len, dim = queries.shape
    kv_heads = keys.shape[1]
    key_length = keys.shape[2]
    topk = block_indices.shape[-1]
    return _kernel()(
        inputs=[
            mx.contiguous(queries),
            mx.contiguous(keys),
            mx.contiguous(values),
            mx.contiguous(mx.sort(block_indices.astype(mx.int32), axis=-1)),
            mx.contiguous(query_ends.astype(mx.int32)),
            mx.array([scale], dtype=mx.float32),
            mx.array([key_length], dtype=mx.int32),
            mx.contiguous(left_padding.astype(mx.int32)),
        ],
        template=[
            ("T", queries.dtype),
            ("D_SIZE", dim),
            ("Q_LEN", q_len),
            ("NUM_Q_HEADS", heads),
            ("NUM_KV_HEADS", kv_heads),
            ("GQA_FACTOR", heads // kv_heads),
            ("BLOCK_SIZE", block_size),
            ("TOPK_BLOCKS", topk),
            ("SELECTED_LENGTH", topk * block_size),
        ],
        grid=(1024, batch * heads * q_len, 1),
        threadgroup=(1024, 1, 1),
        output_shapes=[queries.shape],
        output_dtypes=[queries.dtype],
    )[0]
