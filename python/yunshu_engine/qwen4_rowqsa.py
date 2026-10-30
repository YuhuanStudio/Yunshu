# Patches upstream (MIT): mlx-vlm Qwen4ExpBatchInvariantForward._qsa_attention
# uses singleton-equivalent sparse attention for batched decode (see vendor.json).
"""Row-exact QSA attention for batched Qwen4 decode.

Upstream's singleton decode selects the indexed sparse attention once the prefix outgrows the indexer
budget, while a batch step builds a sparse mask and runs dense attention (left-padded rows) or batched
gathers. The two reduce in a different order, so a row's tokens depended on whether another
request shared its step (and on which one joined first): speculation on and off, or two arrival orders,
produced different seeded streams on Flash-Next from about 2K tokens of context.

Past the sparse start eligible rows share a padded sparse kernel with the singleton reduction order.
Unsupported batches fall back to the singleton path on extracted caches. Before the sparse start
the batched step is already row-invariant and is left alone.
"""

from __future__ import annotations

from typing import Any, cast

import mlx.core as mx

_STATE: dict[str, Any] = {"installed": False, "rows": 0, "batched": 0}


def sparse_start(attention: Any) -> int:
    indexer = attention.indexer
    return int((indexer.block_topk + 1) * indexer.compress_ratio)


def takes_rows(attention: Any, hidden: Any, cache: Any, batch_cache: Any) -> bool:
    """A batched single-token decode step whose longest row may be on the sparse path."""
    return bool(
        isinstance(cache, batch_cache)
        and hidden.shape[0] > 1
        and hidden.shape[1] == 1
        and int(cache._idx) + 1 >= sparse_start(attention)
    )


def run_rows(
    original: Any,
    owner: Any,
    attention: Any,
    hidden: mx.array,
    cache: Any,
    position_ids: Any,
    batch_cache: Any,
) -> mx.array:
    rows = [cache.extract(i) for i in range(hidden.shape[0])]
    outputs = []
    for i, entry in enumerate(rows):
        if position_ids is None:
            pos = None
        elif position_ids.ndim == 3:
            pos = position_ids[:, i : i + 1]
        else:
            pos = position_ids[i : i + 1]
        outputs.append(original(owner, attention, hidden[i : i + 1], entry, pos, None))
    cache.__dict__.update(batch_cache.merge(rows).__dict__)
    _STATE["rows"] += 1
    out: mx.array = mx.concatenate(outputs, axis=0)
    return out


def _index_rows(cache: Any, batch: int) -> list[Any]:
    """Extract only the indexer state; sparse attention uses the batched KV cache."""
    from mlx_vlm.models.qwen4_exp.language import QSAKVCache

    padding = cache.left_padding.tolist()
    rows = []
    for i in range(batch):
        pad = int(padding[i])
        row = QSAKVCache()
        row.offset = int(cache._idx) - pad
        row.index_keys = mx.contiguous(
            cache.index_keys[i : i + 1, pad : cache.index_offset]
        )
        positions = cache.index_position_ids
        if positions.ndim == 3:
            row.index_position_ids = mx.contiguous(
                positions[:, i : i + 1, pad : cache.index_offset]
            )
        else:
            row.index_position_ids = mx.contiguous(
                positions[i : i + 1, pad : cache.index_offset]
            )
        if pad == 0 and cache.index_block_keys is not None:
            row.index_block_keys = mx.contiguous(cache.index_block_keys[i : i + 1])
            row.index_block_ratio = cache.index_block_ratio
        rows.append(row)
    return rows


def run_batch(
    owner: Any, attention: Any, hidden: mx.array, cache: Any, position_ids: Any
) -> mx.array:
    """Run the upstream singleton pipeline with padded batched attention."""
    from .kernels.qwen4_qsa.attention import padded_sparse_attention

    # Batched indexer scoring can change the top-k set near the cutoff.
    # Select exactly as singleton decode does; attention still launches once.
    rows = _index_rows(cache, hidden.shape[0])
    selected = []
    projected_rows = []
    for i, row in enumerate(rows):
        if position_ids is None:
            pos = None
        elif position_ids.ndim == 3:
            pos = position_ids[:, i : i + 1]
        else:
            pos = position_ids[i : i + 1]
        projected_row = owner._linear(
            attention.indexer.index_qk_proj, hidden[i : i + 1]
        )
        projected_rows.append(projected_row)
        row_selection = attention.indexer.select_from_projected(projected_row, row, pos)
        if row_selection is None:
            raise RuntimeError("QSA row was admitted without sparse selection")
        selected.append(row_selection.selected_blocks)
    batch, length, _ = hidden.shape
    projected = mx.concatenate(projected_rows, axis=0).reshape(
        batch,
        length,
        attention.indexer.n_heads + attention.indexer.kv_heads,
        attention.indexer.head_dim,
    )
    raw_keys = projected[:, :, attention.indexer.n_heads :].squeeze(2)
    pos = position_ids
    if pos is None:
        pos = attention.indexer._default_position_ids(batch, cache.offset, length)
    query_ends = mx.full((batch, length), cache.index_offset + 1, dtype=mx.int32)
    cache.update_indexer(raw_keys, pos)
    summaries = [row.index_block_keys for row in rows]
    ratios = [row.index_block_ratio for row in rows]
    if (
        all(value is not None for value in summaries)
        and len({row.offset for row in rows}) == 1
        and len(set(ratios)) == 1
        and len({value.shape[2] for value in summaries}) == 1
    ):
        cache.index_block_keys = mx.concatenate(summaries, axis=0)
        cache.index_block_ratio = ratios[0]
    else:
        cache.clear_index_blocks()
    q, k, v = owner._linears(
        (attention.q_proj, attention.k_proj, attention.v_proj), hidden
    )
    q, k, v, gate, _ = attention._prepare_projected_qkv(
        q, k, v, cache, position_ids, None, None
    )
    out = padded_sparse_attention(
        q,
        k,
        v,
        mx.concatenate(selected, axis=0),
        query_ends,
        cache.left_padding,
        scale=attention.scale,
        block_size=attention.indexer.compress_ratio,
    )
    out = out.transpose(0, 2, 1, 3).reshape(batch, length, -1)
    _STATE["batched"] += 1
    return cast(mx.array, owner._linear(attention.o_proj, out * mx.sigmoid(gate)))


def can_batch(attention: Any, hidden: mx.array, cache: Any, mask: Any) -> bool:
    """Only admit known sparse-compatible rows before mutating either cache."""
    if not (
        mask is None
        or (isinstance(mask, str) and mask in ("causal", "left_padded_decode"))
    ):
        return False
    if (
        hidden.dtype not in (mx.float16, mx.bfloat16)
        or mx.default_device() != mx.gpu
        or not mx.metal.is_available()
    ):
        return False
    if (
        getattr(attention, "head_dim", 0) <= 0
        or attention.head_dim % 32
        or getattr(attention, "num_key_value_heads", 0) <= 0
        or attention.num_attention_heads % attention.num_key_value_heads
        or attention.indexer.compress_ratio <= 0
        or attention.indexer.block_topk <= 0
        or cache.index_keys is None
        or cache.index_offset != cache._idx
    ):
        return False
    # Rows on the dense side of the boundary need the singleton dense reduction.
    return int(cast(Any, mx.min(cache.offset).item())) + 1 >= sparse_start(attention)


def install() -> bool:
    """Patch the Qwen4 batch-invariant forward once; True when active."""
    if _STATE["installed"]:
        return True
    try:
        from mlx_vlm.models.qwen4_exp.language import (
            BatchQSAKVCache,
            Qwen4ExpBatchInvariantForward,
        )
    except ImportError:
        return False
    original = Qwen4ExpBatchInvariantForward._qsa_attention

    def qsa_attention(self, attention, hidden_states, cache, position_ids, mask):
        if takes_rows(attention, hidden_states, cache, BatchQSAKVCache):
            if can_batch(attention, hidden_states, cache, mask):
                return run_batch(self, attention, hidden_states, cache, position_ids)
            return run_rows(
                original,
                self,
                attention,
                hidden_states,
                cache,
                position_ids,
                BatchQSAKVCache,
            )
        return original(self, attention, hidden_states, cache, position_ids, mask)

    Qwen4ExpBatchInvariantForward._qsa_attention = qsa_attention  # type: ignore[method-assign]
    _STATE["installed"] = True
    return True
