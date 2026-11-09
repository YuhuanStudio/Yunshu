"""Yunshu L5 KV Hierarchy — UMA-resident paged KV cache with tiered storage.

Names are resolved lazily (PEP 562). ``tiered`` imports ``mlx.core`` and the
hash/serialization modules import numpy and xxhash; the production callers
that only need ``yunshu_kv.disk_budget`` / ``cache_gc`` / ``fingerprint`` must
not pay for that at import time (gateway and CLI startup).
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .block import BlockPool, KVBlock
    from .block_table import BlockTable
    from .compression import KVTier, TierConfig, compute_compression_ratio
    from .hash import compute_block_hash, compute_prompt_hashes
    from .manager import KVCacheConfig, KVCacheManager, compute_num_blocks
    from .radix_attention import RadixNode, RadixTree
    from .serialization import KVCacheSerializer
    from .ssd_sqlite_store import SSDSQLiteStore
    from .tiered import BackgroundSSDFlush, SSDCacheStore, TieredKVCacheManager
    from .warm_tier import KVTierConfig, KVWarmTier

_LAZY_ATTRS: dict[str, str] = {
    "BackgroundSSDFlush": ".tiered",
    "BlockPool": ".block",
    "BlockTable": ".block_table",
    "KVBlock": ".block",
    "KVCacheConfig": ".manager",
    "KVCacheManager": ".manager",
    "KVCacheSerializer": ".serialization",
    "KVTier": ".compression",
    "KVTierConfig": ".warm_tier",
    "KVWarmTier": ".warm_tier",
    "RadixNode": ".radix_attention",
    "RadixTree": ".radix_attention",
    "SSDCacheStore": ".tiered",
    "SSDSQLiteStore": ".ssd_sqlite_store",
    "TierConfig": ".compression",
    "TieredKVCacheManager": ".tiered",
    "compute_block_hash": ".hash",
    "compute_compression_ratio": ".compression",
    "compute_num_blocks": ".manager",
    "compute_prompt_hashes": ".hash",
}


def __getattr__(name: str) -> object:
    module_name = _LAZY_ATTRS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


__all__ = [
    "BackgroundSSDFlush",
    "BlockPool",
    "BlockTable",
    "KVBlock",
    "KVCacheConfig",
    "KVCacheManager",
    "KVTier",
    "KVTierConfig",
    "KVWarmTier",
    "RadixNode",
    "SSDSQLiteStore",
    "RadixTree",
    "SSDCacheStore",
    "TierConfig",
    "TieredKVCacheManager",
    "compute_block_hash",
    "compute_compression_ratio",
    "compute_num_blocks",
    "compute_prompt_hashes",
    "KVCacheSerializer",
]
