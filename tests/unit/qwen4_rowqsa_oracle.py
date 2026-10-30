# Patches upstream (MIT): mlx-vlm Qwen4ExpBatchInvariantForward._qsa_attention runs each row of a batched
# decode step as the singleton step (tracked in vendor.json; Blaizzy/mlx-vlm).
"""Row-exact QSA attention for batched Qwen4 decode.

Upstream's singleton decode selects the indexed sparse attention once the prefix outgrows the indexer
budget, while a batch step builds a sparse mask and runs dense attention (left-padded rows) or batched
gathers. The two reduce in a different order, so a row's tokens depended on whether another
request shared its step (and on which one joined first): speculation on and off, or two arrival orders,
produced different seeded streams on Flash-Next from about 2K tokens of context.

Past the sparse start each row runs through the singleton path on its own extracted cache, and the batch
cache is rebuilt from the updated rows, so a row's arithmetic no longer depends on its neighbours.
Before the sparse start the batched step is already row-invariant and is left alone.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx

_STATE: dict[str, Any] = {"installed": False, "rows": 0}


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
