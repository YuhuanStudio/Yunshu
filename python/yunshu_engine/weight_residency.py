"""Counters for weights that are read from SSD instead of held in RAM.

The qwen4_exp per-layer n-gram embedding (PLE, ~32 GB) is looked up row by row from the checkpoint's
safetensors ranges (mlx_vlm ``QuantizedMMapNGramEmbedding``). A run only counts as "PLE on SSD" when
these counters move, so the engagement check reads them instead of assuming the path was taken.
"""

from __future__ import annotations

import gc
from typing import Any


def _tables() -> list[Any]:
    try:
        from mlx_vlm.models.qwen4_exp.ple_storage import QuantizedMMapNGramEmbedding
    except Exception:  # noqa: BLE001 - mlx_vlm without qwen4_exp
        return []
    return [o for o in gc.get_objects() if isinstance(o, QuantizedMMapNGramEmbedding)]


def ple_lookup_stats() -> dict[str, Any]:
    """Summed lookup counters over every external PLE table in this process."""
    tables = _tables()
    out = {
        "tables": len(tables),
        "lookups": 0,
        "rows": 0,
        "cache_hits": 0,
        "cache_misses": 0,
        "bytes_read": 0,
        "elapsed_seconds": 0.0,
    }
    for t in tables:
        st = t.stats
        out["lookups"] += st.lookups
        out["rows"] += st.rows
        out["cache_hits"] += st.cache_hits
        out["cache_misses"] += st.cache_misses
        out["bytes_read"] += st.bytes_read
        out["elapsed_seconds"] += st.elapsed_seconds
    out["engaged"] = out["tables"] > 0 and out["lookups"] > 0
    return out
