from __future__ import annotations

"""Yunshu KV Prefix Cache — reuse prefilled KV states across requests.

Stores completed request KV caches keyed by prompt token prefix.
When a new request arrives, finds the longest prefix match and
reuses the cached KV state, only prefilling the remaining tokens.

Major speedup for:
- Multi-turn conversations (system prompt + history already cached)
- Batch requests with same system prompt
- API calls with repeated instructions

Design:
- Hash-chain prefix index: O(matched_blocks) lookup instead of O(n) scan
- Block-level hashing
- Pluggable eviction strategies: LRU, MRU, FILO, SLRU, Priority
- Detached copies to avoid graph reference leaks
- Supports any cache type with keys/values/offset attributes
"""

import contextlib
import gc
import hashlib
import logging
import threading
from copy import copy
from typing import Any

import mlx.core as mx

logger = logging.getLogger(__name__)

# Block size for hash-chain prefix matching (tokens per block).
# Larger blocks = coarser matching but fewer hash computations.
_BLOCK_SIZE = 64


def _canonical_block_hash(
    parent_hash: int | None,
    token_ids: list[int],
) -> int:
    """Compute block hash using the canonical yunshu_kv.hash function.

    Previously this module used its own blake2b-based hash
    producing bytes digests, while yunshu_kv/hash.py used xxhash
    producing int digests.  This prevented cross-subsystem cache hits.
    Now both use the same hash chain.
    """
    from yunshu_kv.hash import compute_block_hash
    return compute_block_hash(parent_hash, token_ids)


def get_prefix_length(prompt: mx.array, cached_prompt: mx.array) -> int:
    """Find the length of the common prefix between two token arrays.

    PERF: done on the CPU via numpy, NOT MLX. The old version ran
    mx.equal+mx.cumprod+mx.sum then `.item()` — a GPU→CPU SYNC on every call, in the
    per-step engine-loop hot path (top non-model Python cost in profiling). A sync there
    stalls the decode pipeline. Token arrays are tiny 1-D ints, so a numpy compare is
    faster and sync-free. argmin of the equality mask gives the first mismatch index =
    prefix length (or n when fully equal)."""
    import numpy as np
    a = np.asarray(prompt)
    b = np.asarray(cached_prompt)
    n = min(a.shape[0], b.shape[0])
    if n == 0:
        return 0
    eq = a[:n] == b[:n]
    if eq.all():
        return int(n)
    return int(np.argmin(eq))


def cache_length(cache: list) -> int:
    """Get the number of tokens in a KV cache."""
    return max((getattr(c, "offset", 0) for c in cache), default=0)


def _detached_copy(a):
    """Create a detached copy of an mx.array (breaks graph references).

    With KV quantization (YUNSHU_KV_QUANT_BITS) a QuantizedKVCache
    stores ``.keys``/``.values`` as a TUPLE ``(packed, scales, biases)``, not a
    single array. ``mx.stop_gradient(tuple)`` raised
    "stop_gradient(): incompatible function arguments ... Invoked with types:
    tuple", crashing every generation into empty output. Detach each element of
    a tuple/list cache representation.
    """
    if isinstance(a, (tuple, list)):
        return type(a)(_detached_copy(x) for x in a)
    return mx.array(mx.stop_gradient(a))


def _token_hash(tokens: mx.array) -> str:
    """Hash token array for fast cache key lookup.

    Pin to int32: bytes() of the SAME tokens as int32 vs int64 differ, so a caller
    passing int64 (some external/batch paths) would miss every int32-keyed entry and
    silently degrade to a scan/full-prefill. The fast path is already int32, so this
    is a no-op there (no cache invalidation).
    """
    return hashlib.blake2b(
        bytes(np_array(tokens).astype("int32")), digest_size=16
    ).hexdigest()


def np_array(arr: mx.array):
    """Convert mx.array to numpy without extra imports at module level."""
    import numpy as np
    return np.array(arr)


def _compute_block_hashes(tokens) -> list[int]:
    """Compute the hash chain for all blocks in a token sequence.

    Returns a list of block hashes where each hash depends on all
    previous blocks, enabling prefix matching at any block boundary.

    Uses the canonical hash from yunshu_kv.hash for cross-subsystem
    compatibility.
    """
    hashes = []
    parent: int | None = None
    for i in range(0, len(tokens), _BLOCK_SIZE):
        block = tokens[i : i + _BLOCK_SIZE]
        parent = _canonical_block_hash(parent, block)
        hashes.append(parent)
    return hashes


# ── Eviction Strategies ──────────────────────────────────────────────────────


def _evictable_indices(n: int, exclude: set[int] | None) -> list[int]:
    """Indices 0..n-1 minus `exclude`. Callers pass the set of entries the
    block-evict checker has pinned this round so a deterministic strategy doesn't keep
    re-proposing the same pinned victim (which made _evict_if_full give up after the
    single oldest entry, leaving the cache above capacity)."""
    if not exclude:
        return list(range(n))
    return [i for i in range(n) if i not in exclude]


class EvictionStrategy:
    """Base class for cache eviction strategies."""

    def select_victim(
        self,
        entries: list,
        last_used: list[int],
        access_counter: int,
        priorities: list[int],
        exclude: set[int] | None = None,
    ) -> int:
        """Return the index of the entry to evict.

        Args:
            entries: List of prompt arrays (for size-aware strategies).
            last_used: Access timestamps for each entry.
            access_counter: Current access counter value.
            priorities: Per-entry priority values (higher = keep longer).
            exclude: Entry indices to skip (pinned by the block-evict checker).
        """
        cand = _evictable_indices(len(entries), exclude)
        return min(cand, key=lambda i: last_used[i])


class LRUStrategy(EvictionStrategy):
    """Evict the least-recently-used entry (default)."""

    def select_victim(self, entries, last_used, access_counter, priorities, exclude=None):
        # Among same-priority entries, evict least recently used
        idxs = _evictable_indices(len(entries), exclude)
        min_priority = min((priorities[i] for i in idxs), default=0)
        candidates = [i for i in idxs if priorities[i] == min_priority] or idxs
        return min(candidates, key=lambda i: last_used[i])


class MRUStrategy(EvictionStrategy):
    """Evict the most-recently-used entry.

    Counter-intuitive but optimal for scan workloads where a large
    result set is read once and never reused (e.g., bulk document analysis).
    Keeps older entries that might be reused in multi-turn conversations.
    """

    def select_victim(self, entries, last_used, access_counter, priorities, exclude=None):
        idxs = _evictable_indices(len(entries), exclude)
        min_priority = min((priorities[i] for i in idxs), default=0)
        candidates = [i for i in idxs if priorities[i] == min_priority] or idxs
        return max(candidates, key=lambda i: last_used[i])


class FILOStrategy(EvictionStrategy):
    """First-In, Last-Out: evict the newest entry.

    Preserves the oldest cached entries (warm system prompts, long-lived
    contexts). Useful when newer entries are speculative and might not
    be reused (e.g., one-shot generations).
    """

    def select_victim(self, entries, last_used, access_counter, priorities, exclude=None):
        idxs = _evictable_indices(len(entries), exclude)
        min_priority = min((priorities[i] for i in idxs), default=0)
        candidates = [i for i in idxs if priorities[i] == min_priority] or idxs
        return max(candidates, key=lambda i: last_used[i])


class SLRUStrategy(EvictionStrategy):
    """Segmented LRU: entries are promoted to a protected segment after N hits.

    Two segments:
    - Probationary: new entries (evicted first)
    - Protected: entries with 2+ accesses (evicted last, within segment by LRU)

    80/20 split: 80% capacity for protected, 20% for probationary.
    Best for workloads with clear hot/cold separation.
    """

    def __init__(self, protected_ratio: float = 0.8, promote_after: int = 2):
        self._protected_ratio = protected_ratio
        self._promote_after = promote_after
        self._access_counts: list[int] = []

    def update_access_counts(self, access_counts: list[int]) -> None:
        self._access_counts = access_counts

    def select_victim(self, entries, last_used, access_counter, priorities, exclude=None):
        idxs = _evictable_indices(len(entries), exclude)
        protected_cap = max(1, int(len(entries) * self._protected_ratio))
        # Separate into probationary and protected (over non-excluded entries)
        probationary = [
            i for i in idxs
            if self._access_counts[i] < self._promote_after
        ]
        protected = [
            i for i in idxs
            if self._access_counts[i] >= self._promote_after
        ]

        # Evict from probationary first (LRU within segment)
        if probationary and len(protected) >= protected_cap:
            return min(probationary, key=lambda i: last_used[i])
        # If probationary is empty or protected is oversized, evict LRU from all
        return min(idxs, key=lambda i: last_used[i])


class PriorityStrategy(EvictionStrategy):
    """Priority-based eviction: higher priority entries are kept longer.

    Within the same priority level, uses LRU as tiebreaker.
    Priority is assigned per-entry via set_priority().
    """

    def select_victim(self, entries, last_used, access_counter, priorities, exclude=None):
        # Evict lowest priority first, then LRU within that tier
        idxs = _evictable_indices(len(entries), exclude)
        min_priority = min((priorities[i] for i in idxs), default=0)
        candidates = [i for i in idxs if priorities[i] == min_priority] or idxs
        return min(candidates, key=lambda i: last_used[i])


def _make_eviction_strategy(name: str) -> EvictionStrategy:
    """Create an eviction strategy by name."""
    strategies = {
        "lru": LRUStrategy,
        "mru": MRUStrategy,
        "fifo": FILOStrategy,
        "filo": FILOStrategy,
        "slru": SLRUStrategy,
        "priority": PriorityStrategy,
    }
    cls = strategies.get(name.lower())
    if cls is None:
        raise ValueError(
            f"Unknown eviction strategy: {name!r}. "
            f"Supported: {', '.join(strategies.keys())}"
        )
    return cls()


class KVPrefixCache:
    """Cache prefilled KV states keyed by prompt token prefix.

    Uses a hash-chain prefix index for O(matched_blocks) lookup instead
    of O(n) linear scan. LRU eviction when at capacity.

    Block dedup + COW:
    - When two entries share identical blocks (same content hash), they
      share the same physical KV data via reference counting.
    - When a snapshot is taken from a shared block, COW creates a copy
      only if the block is shared (refcount > 1).
    - Block refcounts are tracked in _block_refcount: hash → count.
    """

    def __init__(
        self,
        max_entries: int = 64,
        hot_limit: int | None = None,
        min_prefix_length: int = 32,
        eviction: str = "lru",
    ):
        self._prompts: list[mx.array] = []
        self._caches: list[list] = []
        self._block_hashes: list[list[int]] = []
        self._last_used: list[int] = []
        self._access_counter: int = 0
        self._max_entries = max_entries
        # WARM tier: entries beyond hot_limit are kept but 4-bit quantized in RAM
        # (Apple-Silicon UMA value: 4x less memory, still fast — no disk). None
        # disables WARM (all entries full-precision).
        self._hot_limit = hot_limit if hot_limit is not None else max_entries
        self._warm_flags: list[bool] = []
        # HYBRID-model support. Hybrid models (Qwen3.5 etc.)
        # mix trimmable KVCache layers with non-trimmable ArraysCache (linear-
        # attention recurrent state). Such entries may ONLY be reused with
        # trim=0 (an exact full-prefix match landing on the stored boundary) —
        # trimming the tail off a recurrent state corrupts it. When True, get()
        # is replaced by get_no_trim() (boundary-only reuse) and every entry is
        # an exact-length boundary snapshot.
        self._no_trim_mode: bool = False
        self._min_prefix = min_prefix_length
        self._eviction_strategy: EvictionStrategy = _make_eviction_strategy(eviction)
        self._priorities: list[int] = []  # Per-entry priority
        self._access_counts: list[int] = []  # For SLRU promotion tracking
        # Hash-chain prefix index: block_hash → (entry_index, block_index)
        self._hash_index: dict[str, int] = {}
        self._prefix_index: dict[int, list[tuple[int, int]]] = {}
        # Block dedup: block_hash → reference count (COW)
        self._block_refcount: dict[int, int] = {}
        # SSD-tier cache (lazy init)
        self._ssd_cache: Any | None = None
        self._hybrid_ssd: Any | None = None  # whole-snapshot store for hybrid
        self._ssd_model_name: str = ""
        # SSD net-negative guard: reading a prefix back from
        # disk only beats re-prefilling when the prefix is large enough to amortise
        # the I/O — for fast-prefill models a small SSD restore is slower than just
        # prefilling. Skip the disk read when the candidate prefix is below this
        # many tokens. Default 0 = always restore (preserves prior behaviour);
        # operators on fast-prefill models can raise it (e.g. 512).
        import os as _os_kvp
        try:
            self._ssd_restore_min_tokens = int(_os_kvp.environ.get("YUNSHU_SSD_RESTORE_MIN_TOKENS", "0"))
        except (TypeError, ValueError):
            self._ssd_restore_min_tokens = 0
        # SSD prefill-speed auto-gate: for fast-prefill models, reading
        # a prefix back from disk + dequant is SLOWER than just re-prefilling
        # (measured GLM-OCR ~6300 t/s → F-SSD 0.93×, net-negative). When the model's
        # observed prefill throughput exceeds this ceiling, skip the SSD restore and
        # re-prefill instead. Default 4000 t/s keeps SSD for slow-prefill models
        # (0.8B 2947→2.0×, 9B 430→7.8×) and only drops it for blazing ones (GLM).
        # _prefill_tps is fed by the engine via note_prefill_tps(); None = unknown
        # (no gate, preserves prior behaviour).
        try:
            self._ssd_prefill_tps_ceil = float(_os_kvp.environ.get("YUNSHU_SSD_PREFILL_TPS_CEIL", "4000"))
        except (TypeError, ValueError):
            self._ssd_prefill_tps_ceil = 4000.0
        self._prefill_tps: float | None = None
        # Pre-eviction callback for DeltaNet inversion (set by BatchedEngine)
        self._pre_evict_callback: Any | None = None
        # Block eviction checker: callable(block_hash) -> bool (e.g., MemoryGuard.should_evict_block)
        self._block_evict_checker: Any | None = None
        # Hash collision counter (verify token IDs on hash match)
        self._hash_collisions: int = 0
        # Lookup/hit counters (mirrors KVCacheManager API so monitoring shows
        # real cache activity even when fast path bypasses the paged scheduler)
        self._total_lookups: int = 0
        self._total_hits: int = 0
        # Thread safety: protects all mutable state (lists, dicts)
        self._lock = threading.Lock()

    def _maybe_demote_to_warm(self) -> None:
        """Demote oldest full-precision entries to the WARM tier (4-bit, in RAM)
        when the count of full-precision (HOT) entries exceeds hot_limit.

        On Apple-Silicon UMA a WARM CPU tier saves no memory
        (CPU==GPU memory), so WARM here = 4-bit-quantized-IN-PLACE: the entry
        stays cached and matchable but uses ~4x less memory. Runs inside add()
        which is invoked on the MLX executor thread, so c.to_quantized() builds
        arrays on the correct GPU stream (quantizing on another thread caused a
        "no Stream(gpu,1)" / GPU-timeout crash). Hybrid ArraysCache layers have
        no to_quantized → passed through unquantized (still valid).
        """
        if self._hot_limit >= self._max_entries:
            return
        while True:
            cand = [i for i in range(len(self._warm_flags))
                    if not self._warm_flags[i] and i < len(self._last_used)
                    and i < len(self._caches)]
            if len(cand) <= self._hot_limit:
                break
            victim = min(cand, key=lambda i: self._last_used[i])
            try:
                self._caches[victim] = [
                    (c.to_quantized(bits=4) if hasattr(c, "to_quantized") else c)
                    for c in self._caches[victim]
                ]
            except Exception:
                logger.debug("WARM demotion (quantize) failed", exc_info=True)
            # Mark warm regardless so we never re-select it (avoid infinite loop).
            self._warm_flags[victim] = True

    def add(
        self,
        prompt_tokens: mx.array,
        cache: list,
    ) -> None:
        """Store a completed request's KV cache.

        IMPORTANT: cache may contain MORE entries than prompt_tokens when the
        caller has already advanced through generated tokens (BatchedEngine
        appends generated tokens to the same cache during decoding). We trim
        the cache snapshot to match prompt_tokens length, otherwise future
        lookups would replay the prior request's completion as part of the
        prefix and produce nonsense continuations (cross-request
        state leakage).
        """
        if len(prompt_tokens) < self._min_prefix:
            return

        # Compute trim amount: how many tokens cache is ahead of prompt
        prompt_len = int(len(prompt_tokens))
        cache_len = 0
        for c in cache:
            if hasattr(c, "offset"):
                cache_len = max(cache_len, int(getattr(c, "offset", 0) or 0))
            elif hasattr(c, "keys") and c.keys is not None:
                with contextlib.suppress(Exception):
                    cache_len = max(cache_len, int(c.keys.shape[-2]))
        trim_amount = max(0, cache_len - prompt_len)
        logger.info(
            "KV prefix cache add: prompt_len=%d cache_len=%d trim=%d",
            prompt_len, cache_len, trim_amount,
        )

        with self._lock:
            self._evict_if_full()

            prompt_copy = _detached_copy(prompt_tokens)
            cache_copy = self._snapshot_cache(cache, trim=trim_amount)

            # Remove old entry with same hash if exists
            h = _token_hash(prompt_copy)
            if h in self._hash_index:
                old_idx = self._hash_index.pop(h)
                # Consult block evict checker before removing — same path as
                # normal eviction, so shared-block safety checks aren't bypassed.
                if (self._block_evict_checker is not None
                        and old_idx < len(self._block_hashes)):
                    for bh in self._block_hashes[old_idx]:
                        if not self._block_evict_checker(bh):
                            logger.debug(
                                "Block evict checker rejected replacement of hash %s",
                                f"{bh:016x}" if isinstance(bh, int) else bh,
                            )
                self._remove_entry(old_idx)

            # Compute block hashes for prefix index
            block_hashes = _compute_block_hashes(np_array(prompt_copy))

            idx = len(self._prompts)
            self._prompts.append(prompt_copy)
            self._caches.append(cache_copy)
            self._block_hashes.append(block_hashes)
            self._warm_flags.append(False)
            self._access_counter += 1
            self._last_used.append(self._access_counter)
            self._priorities.append(0)
            self._access_counts.append(1)
            self._maybe_demote_to_warm()
            self._hash_index[h] = idx

            # Add to prefix index with block dedup refcounting
            for bi, bh in enumerate(block_hashes):
                if bh not in self._prefix_index:
                    self._prefix_index[bh] = []
                self._prefix_index[bh].append((idx, bi))
                self._block_refcount[bh] = self._block_refcount.get(bh, 0) + 1

            logger.info(
                f"KV prefix cache added: {len(prompt_tokens)} tokens, "
                f"blocks={len(block_hashes)}, total entries={len(self._prompts)}"
            )

        mx.eval(prompt_copy)

    def note_prefill_tps(self, tps: float) -> None:
        """Feed the engine's observed cold-prefill throughput (tokens/sec) so the
        SSD tier can auto-gate: fast-prefill models skip the (slower) disk restore.
        EWMA-smoothed; ignores non-positive/absurd values."""
        if not tps or tps <= 0 or tps > 1e6:
            return
        if self._prefill_tps is None:
            self._prefill_tps = float(tps)
        else:
            self._prefill_tps = 0.7 * self._prefill_tps + 0.3 * float(tps)

    def get(
        self,
        prompt_tokens: mx.array,
        exact_refeed_trim: bool = True,
    ) -> tuple[list | None, int, int]:
        """Find best prefix match and return cached KV state.

        Uses hash-chain index for O(matched_blocks) lookup:
        1. Compute block hashes for query tokens
        2. Walk the chain looking up each hash in prefix_index
        3. Find longest chain match
        Falls back to exact hash match or vectorized scan.

        Args:
            exact_refeed_trim: on a FULL/exact hit (remaining == 0), trim the
                returned snapshot by one token and report matched-1 so a fast-path
                caller that re-feeds ``ids[-1:]`` prefills the last token ONCE.
                The ENGINE-LOOP SCHEDULER
                must pass False: it uses ``matched == len`` to detect a full match
                and fall back to an efficient BATCHED prefill — the L-1 signal
                defeated that detection and routed every exact-hit request through
                the per-request KV-restore path instead, collapsing batched
                throughput at batch≥16 (152→79 tok/s).

        Returns:
            (cached_kv, remaining_token_count, matched_token_count)
            cached_kv is None if no useful match found.
        """
        with self._lock:
            self._total_lookups += 1
            if self._no_trim_mode:
                result = self._get_no_trim_unlocked(prompt_tokens)
            else:
                result = self._get_unlocked(prompt_tokens)
            # result is (cached_kv, remaining, matched); count as hit when a
            # cached KV was returned (matched >= min_prefix in the cache paths).
            if result[0] is not None:
                self._total_hits += 1
                # on a FULL/exact prefix hit (remaining == 0 ⟺
                # matched == prompt length) the returned cache holds the ENTIRE prompt
                # incl. the last token. The fast-path callers then re-feed ids[-1:] to
                # start decoding; if the cache still holds that token, re-feeding
                # DUPLICATES it at a shifted RoPE position → confident-wrong first decode
                # token (breaks greedy-exact). This is directly reachable on the streaming
                # fast path (no prompt_cache shadow there). Trim the (detached) snapshot by
                # one token and report matched-1 so the caller prefills exactly the last
                # token ONCE — mirrors the prompt_cache _refeed=1 path. Partial hits
                # (remaining > 0) leave tokens to prefill and are unaffected.
                _cached, _remaining, _matched = result
                # Only adjust when EVERY layer cache is trimmable — otherwise reporting
                # matched-1 without actually trimming would itself be off-by-one. (Real
                # mlx-lm caches reaching here all support trim; sliding-window caches are
                # bypassed upstream.)
                if (exact_refeed_trim and _remaining == 0 and _matched > 0
                        and all(hasattr(_c, "trim") for _c in _cached)):
                    try:
                        for _c in _cached:
                            _c.trim(1)
                        return _cached, 1, _matched - 1
                    except Exception:
                        logger.debug(
                            "exact-hit refeed trim failed — signalling full prefill",
                            exc_info=True,
                        )
                        return None, 0, 0
            return result

    def _get_no_trim_unlocked(
        self,
        prompt_tokens: mx.array,
    ) -> tuple[list | None, int, int]:
        """HYBRID-safe lookup: return the longest cached entry that is a FULL
        prefix of the query (so reuse needs trim=0). Entries whose stored prompt
        is only partially matched are skipped — trimming their tail would corrupt
        the non-trimmable recurrent (ArraysCache) layers.
        """
        # do NOT early-exit on an empty RAM cache when a hybrid SSD tier
        # exists — the SSD probe below (the whole point of SSD: serving prefix hits
        # after RAM eviction / a process restart) lived AFTER this guard, so a
        # cold/empty RAM cache never consulted SSD and every restore was a silent
        # miss. The empty-cache fall-through is a no-op loop → reaches the SSD probe.
        if not self._prompts and self._hybrid_ssd is None:
            return None, len(prompt_tokens), 0
        best_index = -1
        best_length = 0
        for i, cached_prompt in enumerate(self._prompts):
            clen = len(cached_prompt)
            # Only consider entries that fit and improve on the best so far.
            if clen <= best_length or clen > len(prompt_tokens):
                continue
            matched = get_prefix_length(prompt_tokens, cached_prompt)
            # Require the WHOLE stored entry to be a prefix of the query
            # (matched == clen) → reuse with zero trim.
            if matched == clen and matched >= self._min_prefix:
                best_index = i
                best_length = clen
        if best_index < 0:
            # RAM miss — try the hybrid SSD whole-snapshot store.
            # Probe the EXACT stored token-counts (descending → longest
            # match first), NOT 64-aligned boundaries. The snapshot is keyed by the
            # hash of the FULL stored prompt (almost never a multiple of _BLOCK_SIZE),
            # so the old block-aligned probe `_token_hash(q[:b])` matched only when
            # len(prompt) % 64 == 0 → the snapshot was written but never restorable
            # (the "wrote N MB / restored 0" failure class). The largest stored count c
            # whose prefix-hash matches q[:c] is reusable with trim=0.
            if self._hybrid_ssd is not None:
                q = prompt_tokens
                n = len(q)
                try:
                    _counts = self._hybrid_ssd.candidate_token_counts()
                except Exception:
                    _counts = []
                for c in _counts:
                    if c > n or c < self._min_prefix:
                        continue
                    try:
                        key = _token_hash(q[:c]).encode("utf-8")[:32]
                        if self._hybrid_ssd.has(key):
                            cache_list, tok = self._hybrid_ssd.load(key)
                            if cache_list is not None and tok >= self._min_prefix:
                                logger.info("hybrid SSD snapshot hit: %d/%d tokens", tok, n)
                                return cache_list, n - tok, tok
                    except Exception:
                        logger.debug("hybrid SSD probe failed at c=%d", c, exc_info=True)
            return None, len(prompt_tokens), 0
        # Decline a FULL-EXACT match (the whole query is cached, remaining
        # would be 0). The no_trim path is for hybrid/recurrent caches that CANNOT be
        # trimmed; the fast-path caller then re-feeds ids[-1:] to start decoding,
        # which DUPLICATES the last token (shifted RoPE in KVCache layers + double-fed
        # recurrent ArraysCache state) → wrong output. We can't trim it away here, so
        # fall back to a full prefill. Partial matches (query longer than the stored
        # prefix → remaining > 0, the valuable shared-prefix case) are unaffected.
        # Exact repeats are served by the response cache anyway.
        if best_length >= len(prompt_tokens):
            return None, len(prompt_tokens), 0
        cached = self._caches[best_index]
        cached_len = cache_length(cached)
        # Defensive: only proceed if the stored cache length matches the prompt
        # length (a true boundary snapshot). Otherwise reuse would need a trim.
        if cached_len != best_length:
            logger.debug(
                "no_trim: entry %d cache_len=%d != matched=%d — skipping (would trim)",
                best_index, cached_len, best_length,
            )
            return None, len(prompt_tokens), 0
        try:
            result = self._snapshot_cache(cached, trim=0)
        except Exception:
            logger.warning("no_trim snapshot failed — full prefill", exc_info=True)
            return None, len(prompt_tokens), 0
        self._touch(best_index)
        logger.info("KV prefix cache no_trim hit: %d/%d tokens",
                    best_length, len(prompt_tokens))
        return result, len(prompt_tokens) - best_length, best_length

    def _get_unlocked(
        self,
        prompt_tokens: mx.array,
    ) -> tuple[list | None, int, int]:
        """Internal get without lock — callers must hold _lock."""
        # do NOT early-exit on an empty RAM cache when an SSD tier exists —
        # the per-block SSD restore below (the whole point of SSD: serving prefix hits
        # after RAM eviction / a process restart) lived AFTER this guard, so a
        # cold/empty RAM cache never consulted SSD and every restore was a silent miss
        # (proven: 2 blocks written, 0 read back). The fall-through is empty-safe: the
        # exact-hash and hash-chain lookups no-op on an empty index, then the slow scan
        # is a no-op loop, reaching the SSD restore.
        if not self._prompts and self._ssd_cache is None:
            return None, len(prompt_tokens), 0

        # Fast path: exact hash match
        h = _token_hash(prompt_tokens)
        if h in self._hash_index:
            idx = self._hash_index[h]
            # Verify token IDs to detect hash collisions.
            # A blake2b collision is astronomically unlikely but a bug in
            # the hash function or a corrupted index could cause it, and
            # reusing the wrong KV cache would produce garbage output.
            cached_prompt = self._prompts[idx]
            if len(cached_prompt) != len(prompt_tokens):
                # Length mismatch means the hash collides with a different entry.
                self._hash_collisions += 1
                logger.warning(
                    f"KV prefix cache hash collision detected (length mismatch: "
                    f"query={len(prompt_tokens)}, cached={len(cached_prompt)}). "
                    f"Falling back to scan."
                )
            else:
                actual_prefix = get_prefix_length(prompt_tokens, cached_prompt)
                if actual_prefix != len(prompt_tokens):
                    # Tokens differ despite same hash — collision.
                    self._hash_collisions += 1
                    logger.warning(
                        f"KV prefix cache hash collision detected (token mismatch: "
                        f"matched={actual_prefix}/{len(prompt_tokens)}). "
                        f"Falling back to scan."
                    )
                else:
                    # Verified: tokens match the hash entry.
                    cached = self._caches[idx]
                    matched = len(prompt_tokens)
                    # Trim stale generation tokens from cached KV so the
                    # caller receives only prompt-prefix state, not leftover
                    # KV entries from a previous request's generation.
                    cached_len = cache_length(cached)
                    tokens_to_trim = max(0, cached_len - matched)
                    try:
                        result = self._snapshot_cache(cached, trim=tokens_to_trim)
                    except Exception:
                        logger.warning(
                            "KV prefix cache snapshot failed during exact match "
                            "(%d tokens) — falling back to full prefill",
                            matched, exc_info=True,
                        )
                        return None, len(prompt_tokens), 0
                    self._touch(idx)
                    logger.info(
                        f"KV prefix cache exact hit: {matched}/{len(prompt_tokens)} tokens"
                    )
                    return result, 0, matched

        # Medium path: hash-chain prefix index lookup
        query_blocks = _compute_block_hashes(np_array(prompt_tokens))
        best_index, best_blocks = self._find_prefix_via_hash_chain(query_blocks)

        if best_blocks > 0:
            cached_len = cache_length(self._caches[best_index])
            # the hash chain confirms >= best_blocks*_BLOCK_SIZE tokens
            # match by block hash, but the TRUE reusable prefix is the token-exact
            # common prefix — do NOT floor to the 64-token block boundary. The old
            # code seeded best_length = best_blocks*_BLOCK_SIZE and then only ever
            # min()'d it down, so a 100-token cached prefix reused only 64 tokens
            # (36 valid tokens silently re-prefilled) on every multi-turn /
            # system-prompt hit. The snapshot trims to ANY length, so block
            # alignment was never required. Use the exact prefix, capped by what
            # the cache actually holds and the query length.
            actual_prefix = get_prefix_length(
                prompt_tokens, self._prompts[best_index]
            )
            best_length = min(actual_prefix, cached_len, len(prompt_tokens))

            if best_length >= self._min_prefix:
                # Hash collision guard: if the "prefix" covers the entire
                # cached prompt but the query diverges after that, this is a
                # collision not a prefix match — skip it.
                if best_length == len(self._prompts[best_index]) and best_length < len(prompt_tokens):
                    # Verify the tokens actually match at the boundary. NOTE: a raw
                    # `mx.array != mx.array` inside an `if` raises "Only length-1
                    # arrays can be converted to Python scalars" for best_length>1 —
                    # which fired on the MOST COMMON hit (cached prompt is a strict
                    # prefix of a longer multi-turn query), and the call site caught
                    # it and silently fell back to a full prefill, defeating the
                    # cache. Reduce to a Python bool. (get_prefix_length above already
                    # guarantees the match, so this is a defensive collision guard.)
                    # numpy (CPU) compare — avoids a GPU→CPU sync on the prefix-cache
                    # get() path (consistent with get_prefix_length). Tiny int
                    # arrays; the sync isn't worth stalling the pipeline for.
                    import numpy as _np
                    if not bool(_np.array_equal(
                        _np.asarray(prompt_tokens[:best_length]),
                        _np.asarray(self._prompts[best_index][:best_length]),
                    )):
                        best_length = 0  # collision — invalidate

            if best_length >= self._min_prefix:
                cached = self._caches[best_index]
                tokens_to_trim = cached_len - best_length
                try:
                    result = self._snapshot_cache(cached, trim=tokens_to_trim)
                except Exception:
                    logger.warning(
                        "KV prefix cache snapshot failed during hash-chain match "
                        "(%d tokens, trim=%d) — falling back to full prefill",
                        best_length, tokens_to_trim, exc_info=True,
                    )
                    return None, len(prompt_tokens), 0
                self._touch(best_index)

                remaining = len(prompt_tokens) - best_length
                logger.info(
                    f"KV prefix cache hash-chain hit: matched {best_length}/{len(prompt_tokens)} tokens, "
                    f"remaining={remaining}"
                )
                return result, remaining, best_length

        # Slow path: vectorized scan (fallback for short prefixes)
        best_index = -1
        best_length = 0

        for i, cached_prompt in enumerate(self._prompts):
            if len(cached_prompt) <= best_length:
                continue
            length = get_prefix_length(prompt_tokens, cached_prompt)
            if length > best_length:
                best_index = i
                best_length = length

        if best_length < self._min_prefix:
            # SSD restore: reassemble the longest run of consecutive
            # SSD-resident prefix blocks into a usable contiguous cache.
            # Net-negative guard: skip the disk read when even the best-case
            # restore (all query blocks) is too small to beat a re-prefill.
            _ssd_skip = (self._ssd_restore_min_tokens > 0
                         and len(query_blocks) * _BLOCK_SIZE < self._ssd_restore_min_tokens)
            # Prefill-speed auto-gate: for fast-prefill models the disk
            # restore is slower than re-prefilling (GLM-OCR ~6300 t/s → F-SSD 0.93×).
            # Skip when observed prefill throughput exceeds the ceiling.
            if (not _ssd_skip and self._prefill_tps is not None
                    and self._prefill_tps > self._ssd_prefill_tps_ceil):
                _ssd_skip = True
                logger.info(
                    "KV prefix cache SSD restore gated by prefill speed: "
                    "%.0f t/s > ceil %.0f — re-prefill instead",
                    self._prefill_tps, self._ssd_prefill_tps_ceil,
                )
            if self._ssd_cache is not None and query_blocks and not _ssd_skip:
                try:
                    ssd_cache, n_tok = self.restore_prefix_from_ssd(query_blocks)
                except Exception:
                    logger.warning(
                        "KV prefix cache SSD restore failed — full prefill",
                        exc_info=True,
                    )
                    return None, len(prompt_tokens), 0
                if ssd_cache is not None and n_tok >= self._min_prefix:
                    # NOTE: do NOT increment _total_hits here — the outer get()
                    # counts any non-None result once (line ~492). Incrementing here
                    # too made SSD restores double-count (hit_rate could exceed 1.0).
                    logger.info("KV prefix cache SSD restore: %d tokens", n_tok)
                    return ssd_cache, len(prompt_tokens) - n_tok, n_tok
            return None, len(prompt_tokens), 0

        cached = self._caches[best_index]
        cached_len = cache_length(cached)
        tokens_to_trim = cached_len - best_length

        try:
            result = self._snapshot_cache(cached, trim=tokens_to_trim)
        except Exception:
            logger.warning(
                "KV prefix cache snapshot failed during scan match "
                "(%d tokens, trim=%d) — falling back to full prefill",
                best_length, tokens_to_trim, exc_info=True,
            )
            return None, len(prompt_tokens), 0
        self._touch(best_index)

        remaining = len(prompt_tokens) - best_length
        logger.info(
            f"KV prefix cache scan hit: matched {best_length}/{len(prompt_tokens)} tokens, "
            f"remaining={remaining}"
        )
        return result, remaining, best_length

    def _find_prefix_via_hash_chain(
        self, query_blocks: list[int]
    ) -> tuple[int, int]:
        """Find longest prefix match using hash-chain index.

        Optimised to avoid O(n*k) full-chain verification on every block.
        Instead, maintains a candidate set that is narrowed at each step:
        after processing query block i, only entries whose block i matches
        are retained as candidates for block i+1.  This gives O(total_matches)
        instead of O(n * k).

        Returns (entry_index, matched_blocks).
        """
        if not query_blocks:
            return -1, 0

        # Seed candidates from the first query block's prefix_index entries.
        first_hash = query_blocks[0]
        if first_hash not in self._prefix_index:
            return -1, 0

        # candidate_entry -> number of consecutive matching blocks (from 0)
        candidates: dict[int, int] = {}
        for entry_idx, block_idx in self._prefix_index[first_hash]:
            if block_idx == 0:
                candidates[entry_idx] = 1

        if not candidates:
            return -1, 0

        # Extend the chain one block at a time, pruning entries that
        # diverge from the query.
        for qi in range(1, len(query_blocks)):
            qhash = query_blocks[qi]
            if qhash not in self._prefix_index:
                break

            # Build a lookup: entry_idx -> True for entries that have this
            # hash at position qi.
            entries_at_qi: set[int] = set()
            for entry_idx, block_idx in self._prefix_index[qhash]:
                if block_idx == qi:
                    entries_at_qi.add(entry_idx)

            # Retain only candidates that also match at position qi.
            surviving: dict[int, int] = {}
            for entry_idx, matched in candidates.items():
                if entry_idx in entries_at_qi:
                    surviving[entry_idx] = matched + 1
            if not surviving:
                break
            candidates = surviving

        if not candidates:
            return -1, 0

        # Return the entry with the longest match.
        best_entry, best_blocks = max(candidates.items(), key=lambda x: x[1])
        return best_entry, best_blocks

    @staticmethod
    def _has_recurrent_layer(cache) -> bool:
        """True if the cache holds a linear-attn / recurrent layer — an
        ArraysCache with a fixed-size recurrent state and NO per-token ``.keys``
        tensor (e.g. Qwen3.5 hybrid).

        Only such backbones need the whole-snapshot SSD tier: their recurrent
        state can't be block-decomposed, so the only way to persist a prefix is
        the full boundary snapshot. Sliding-window (RotatingKVCache) models DO
        have per-token keys and are fully served by the in-RAM no_trim reuse, so
        persisting their (large, rarely-restorable) full snapshot to disk is
        net-negative — measured 923 MB written / 0 restored for gemma-4.
        """
        if not cache:
            return False
        for c in cache:
            if not isinstance(c, (tuple, list)) and getattr(c, "keys", None) is None:
                return True
        return False

    @staticmethod
    def _is_block_decomposable(cache) -> bool:
        """Whether a KV cache can be split into fixed 64-token blocks for the
        per-block SSD tier — True ONLY when sequence position == tensor index.

        FALSE for:
          - RotatingKVCache (sliding-window, e.g. gemma-4: 35/42 layers have a
            512-window): the rotating buffer wraps, so ``block_index*64`` does
            NOT map to an absolute sequence position. Slicing it produces garbage
            that the restore can't reassemble. This silently spilled
            923 MB of unrestorable blocks for gemma-4 (vs 44.7 MB) while every
            SSD restore failed (restored=False, 0.76x — slower than not caching),
            because RotatingKVCache has real ``.keys`` tensors so it slipped past
            the existing hybrid (ArraysCache) bypass.
          - ArraysCache / recurrent linear-attn (e.g. Qwen3.5 hybrid): a
            fixed-size recurrent state, not a per-token KV tensor.

        For such backbones the per-block SSD tier is net-negative (wasted disk +
        always-failing restore), so the spill is skipped and reuse falls back to
        the HOT/WARM in-RAM tiers (which work) or a full prefill.
        """
        if not cache:
            return False
        for c in cache:
            # Recurrent / linear-attn state: no per-token keys tensor.
            if not isinstance(c, (tuple, list)) and getattr(c, "keys", None) is None:
                return False
            # Sliding-window RotatingKVCache: has rotating-buffer attrs that the
            # plain KVCache / QuantizedKVCache lack.
            if hasattr(c, "max_size") and hasattr(c, "_idx"):
                return False
        return True

    def _extract_block_cache_data(self, cache, block_index: int):
        """Extract KV data for a single block from a full cache snapshot.

        Two bugs made the SSD spill a silent no-op so the SSD
        tier never stored anything: (1) it sliced ``c.keys[start:end]`` on AXIS 0
        (batch=1 for KVCache shape [B, H, S, D]) instead of the SEQUENCE axis
        (-2); (2) it rebuilt via ``type(c)(keys=…, values=…)`` but mlx-lm's
        KVCache constructor takes NO keys/values kwargs → TypeError swallowed by
        the caller's except → save skipped. Fixed: slice the seq axis and return
        a lightweight .keys/.values holder (save_block only reads those two
        attributes, not a real KVCache).
        """
        if cache is None:
            return None
        from types import SimpleNamespace

        import mlx.core as mx
        bs = _BLOCK_SIZE
        result = []
        for c in cache:
            # WARM-quantized layers store keys/values as a TUPLE (no .ndim), so they'd
            # fall to the else-branch below, be appended whole (the entire layer, for
            # EVERY block index), and save_block would then fail to serialize the tuple
            # → silent no-op SSD spill (WARM entries never reached disk). Dequantize a
            # copy to full precision first so the normal seq-axis block slice applies.
            # No-op when c is not quantized. The stored WARM entry stays quantized.
            c = self._maybe_dequantize(c)
            k = getattr(c, "keys", None)
            v = getattr(c, "values", None)
            if (k is not None and v is not None
                    and hasattr(k, "ndim") and k.ndim >= 2):
                seq_len = k.shape[-2]  # [.., S, D] → seq is the second-to-last axis
                start = block_index * bs
                if start >= seq_len:
                    result.append(SimpleNamespace(keys=None, values=None))
                    continue
                end = min(start + bs, seq_len)
                result.append(SimpleNamespace(
                    keys=mx.array(k[..., start:end, :]),
                    values=mx.array(v[..., start:end, :]),
                ))
            else:
                # Quantized/recurrent (tuple) layers — pass through as-is.
                result.append(c)
        return result

    def _maybe_dequantize(self, c):
        """If c is a WARM (4-bit QuantizedKVCache) layer, dequantize it back to a
        full-precision KVCache for reuse.

        Trimming/reusing a group-quantized cache by a
        non-group-aligned amount corrupts it (produced broken output). The WARM
        tier keeps entries quantized for STORAGE (4x less RAM), but on reuse we
        dequantize a copy to full precision so snapshot/trim/insert_segments are
        correct. The stored entry stays quantized. Runs on the MLX executor
        thread (called from get() in _run).
        """
        k = getattr(c, "keys", None)
        v = getattr(c, "values", None)
        if (isinstance(k, (tuple, list)) and isinstance(v, (tuple, list))
                and hasattr(c, "group_size") and hasattr(c, "bits")):
            try:
                from mlx_lm.models.cache import KVCache
                dk = mx.dequantize(*k, group_size=c.group_size, bits=c.bits)
                dv = mx.dequantize(*v, group_size=c.group_size, bits=c.bits)
                off = int(getattr(c, "offset", dk.shape[-2]))
                nc = KVCache()
                nc.keys = dk[..., :off, :]
                nc.values = dv[..., :off, :]
                nc.offset = off
                return nc
            except Exception:
                logger.debug("WARM dequantize-on-reuse failed", exc_info=True)
        return c

    def _snapshot_cache(self, cache: list, trim: int = 0) -> list:
        """Create a detached snapshot of a KV cache.

        COW optimization: layers whose block hash has
        refcount == 1 (not shared) can reuse the same tensor references
        without copying. Only shared blocks (refcount > 1) need a
        detached copy to prevent aliasing.
        """
        result = []
        for c in cache:
            c = self._maybe_dequantize(c)
            if not (hasattr(c, "keys") and c.keys is not None
                    and hasattr(c, "values") and c.values is not None):
                from copy import deepcopy
                result.append(deepcopy(c))
                continue

            snap = copy(c)
            snap.keys = _detached_copy(c.keys)
            snap.values = _detached_copy(c.values)
            if trim > 0 and hasattr(snap, "trim"):
                snap.trim(trim)
                # KVCache.trim() only decrements offset; the underlying
                # keys/values tensors still hold stale generation K/V at
                # positions >offset. Future writes append into the same
                # buffer, but Metal kernels can still see those stale
                # slots via the buffer's full shape. Slice the tensors
                # to the new offset to guarantee no cross-request leak.
                new_offset = int(getattr(snap, "offset", 0))
                # Only the plain-array cache can be tensor-sliced. A quantized
                # cache stores keys/values as a (packed, scales, biases) tuple
                # with no .ndim — skip the extra slice there (snap.trim already
                # adjusted the offset; quantized blocks aren't tensor-sliceable).
                if (isinstance(snap.keys, mx.array)
                        and snap.keys.ndim >= 3
                        and new_offset < snap.keys.shape[-2]):
                    snap.keys = _detached_copy(snap.keys[..., :new_offset, :])
                    snap.values = _detached_copy(snap.values[..., :new_offset, :])
            elif trim > 0 and hasattr(snap, "offset"):
                # Bug fix: only adjusting offset without trimming the
                # actual tensors leaves stale KV data beyond the logical
                # boundary.  Downstream code that reads by tensor shape
                # (not offset) would see garbage.  Slice the tensors to
                # the new offset to match the semantic contract.
                new_offset = max(0, getattr(c, "offset", 0) - trim)
                snap.offset = new_offset
                # the sequence axis of an mlx KV tensor is -2
                # (shape [batch, heads, seq, head_dim]), NOT axis 0. The old
                # code sliced [:new_offset] (batch) and tested shape[0], which
                # would corrupt KV if this branch ever ran. Mirror the .trim
                # branch above. (Latent today — every mlx-lm cache type has
                # .trim, so this offset-only branch is reached only by an exotic
                # custom cache — but keep it correct.)
                if (isinstance(snap.keys, mx.array)
                        and snap.keys.ndim >= 3
                        and 0 < new_offset < snap.keys.shape[-2]):
                    snap.keys = _detached_copy(snap.keys[..., :new_offset, :])
                    snap.values = _detached_copy(snap.values[..., :new_offset, :])
                elif new_offset > 0:
                    # Non-tensor / low-rank cache: keep offset-adjusted snapshot
                    # without slicing (can't safely slice an unknown layout).
                    result.append(snap)
                    continue
                else:
                    # Trim consumed the entire cache — append a zero-length
                    # detached copy instead of the original reference, to
                    # prevent the caller from corrupting the cached entry.
                    snap.offset = 0
                    snap.keys = _detached_copy(snap.keys[:0])
                    snap.values = _detached_copy(snap.values[:0])
                    result.append(snap)
                    continue
            result.append(snap)
        return result

    def _touch(self, index: int) -> None:
        """Update LRU timestamp and access count for accessed entry."""
        self._access_counter += 1
        self._last_used[index] = self._access_counter
        self._access_counts[index] += 1

    def _remove_entry(self, index: int, rebuild_index: bool = True) -> None:
        """Remove an entry and clean up all indices.

        Uses swap-and-pop for O(1) list removal instead of index-based pop
        which requires O(n) shift. Set ``rebuild_index=False`` when calling
        in a batch loop and rebuild once after all removals.
        """
        # Pre-eviction callback: gives engine a chance to capture inverted
        # DeltaNet SSM state before the KV cache is discarded.
        if self._pre_evict_callback is not None:
            try:
                self._pre_evict_callback(
                    self._prompts[index], self._caches[index],
                )
            except Exception:
                logger.debug("pre-evict callback failed", exc_info=True)
        # truly-recurrent HYBRID models (Qwen3.5 linear-attn)
        # spill the WHOLE boundary snapshot (all layers) to disk keyed by its
        # prefix hash, since ArraysCache recurrent state can't be saved as
        # per-token blocks. Reused with trim=0.
        # Gate on _has_recurrent_layer. Sliding-window models
        # (gemma-4: RotatingKVCache) also land in _no_trim_mode but have per-token
        # keys — persisting their full snapshot wrote 923 MB / restored 0. They
        # fall through to the per-block branch, where _is_block_decomposable then
        # also (correctly) skips them → no SSD write at all. The in-RAM no_trim
        # reuse (HOT/WARM) still serves them losslessly (3.5x).
        _ec_for_spill = self._caches[index] if index < len(self._caches) else None
        if (self._no_trim_mode and self._hybrid_ssd is not None
                and self._has_recurrent_layer(_ec_for_spill)):
            try:
                ep = self._prompts[index] if index < len(self._prompts) else None
                ec = _ec_for_spill
                if ep is not None and ec is not None:
                    key = _token_hash(ep).encode("utf-8")[:32]
                    if not self._hybrid_ssd.has(key):
                        self._hybrid_ssd.save(key, ec, int(len(ep)))
            except Exception:
                logger.debug("hybrid SSD snapshot spill failed", exc_info=True)
            # Skip the per-block SSD path below for hybrid (can't serialize it).
        elif self._ssd_cache is not None:
            try:
                evicted_hashes = self._block_hashes[index] if index < len(self._block_hashes) else []
                evicted_cache = self._caches[index] if index < len(self._caches) else None
                evicted_prompt = self._prompts[index] if index < len(self._prompts) else None
                # Sliding-window (RotatingKVCache) backbones aren't block-
                # decomposable — spilling them wastes disk and the restore always
                # fails. Skip the per-block path (see _is_block_decomposable).
                if evicted_cache is not None and not self._is_block_decomposable(evicted_cache):
                    evicted_hashes = []
                if evicted_hashes and evicted_cache is not None and evicted_prompt is not None:
                    for bi, bh in enumerate(evicted_hashes):
                        bh_bytes = bh.to_bytes(8, "little") if isinstance(bh, int) else bh
                        if not self._ssd_cache.has_block(bh_bytes):
                            try:
                                import numpy as np
                                tokens = np.array(evicted_prompt)
                                start = bi * _BLOCK_SIZE
                                end = min(start + _BLOCK_SIZE, len(tokens))
                                block_cache = self._extract_block_cache_data(evicted_cache, bi)
                                self._ssd_cache.save_block(
                                    block_hash=bh_bytes,
                                    cache_data=block_cache,
                                    token_count=end - start,
                                    model_name=self._ssd_model_name,
                                )
                            except Exception:
                                bh_hex = f"{bh:016x}" if isinstance(bh, int) else bh.hex()[:16]
                                logger.debug(
                                    "SSD spill save failed for block %s",
                                    bh_hex[:16], exc_info=True,
                                )
            except Exception:
                logger.debug("SSD spill failed during eviction", exc_info=True)

        # Swap-and-pop: O(1) removal by swapping with the last element.
        last = len(self._prompts) - 1
        if index != last:
            # Update hash index for the swapped entry before moving it
            swapped_prompt = self._prompts[last]
            swapped_hash = _token_hash(swapped_prompt)
            if swapped_hash in self._hash_index and self._hash_index[swapped_hash] == last:
                self._hash_index[swapped_hash] = index
            self._prompts[index] = self._prompts[last]
            self._caches[index] = self._caches[last]
            self._block_hashes[index] = self._block_hashes[last]
            self._last_used[index] = self._last_used[last]
            self._priorities[index] = self._priorities[last]
            self._access_counts[index] = self._access_counts[last]
            self._warm_flags[index] = self._warm_flags[last]
        # Pop last element (O(1) — no shift needed)
        self._prompts.pop()
        self._caches.pop()
        self._block_hashes.pop()
        self._last_used.pop()
        self._priorities.pop()
        self._access_counts.pop()
        if self._warm_flags:
            self._warm_flags.pop()

        if rebuild_index:
            self._rebuild_hash_index()

    def _rebuild_hash_index(self) -> None:
        """Rebuild hash index, prefix index, and block refcounts after structural changes."""
        self._hash_index.clear()
        self._prefix_index.clear()
        self._block_refcount.clear()
        for i, prompt in enumerate(self._prompts):
            self._hash_index[_token_hash(prompt)] = i
            for bi, bh in enumerate(self._block_hashes[i]):
                if bh not in self._prefix_index:
                    self._prefix_index[bh] = []
                self._prefix_index[bh].append((i, bi))
                self._block_refcount[bh] = self._block_refcount.get(bh, 0) + 1

    def _evict_if_full(self) -> None:
        """Evict entries using the configured strategy when at capacity."""
        _skipped_indices: set[int] = set()
        _evicted_any = False
        while len(self._prompts) >= self._max_entries and len(_skipped_indices) < len(self._prompts):
            if isinstance(self._eviction_strategy, SLRUStrategy):
                self._eviction_strategy.update_access_counts(self._access_counts)
            victim = self._eviction_strategy.select_victim(
                self._prompts, self._last_used, self._access_counter,
                self._priorities, exclude=_skipped_indices,
            )
            if victim in _skipped_indices:
                # Defensive: a correct strategy never returns an excluded index, but
                # if one does, all evictable candidates are pinned — give up.
                break
            if self._block_evict_checker is not None and self._block_hashes[victim]:
                skip = False
                for bh in self._block_hashes[victim]:
                    if not self._block_evict_checker(bh):
                        skip = True
                        break
                if skip:
                    _skipped_indices.add(victim)
                    continue
            # Rebuild index after each removal to keep select_victim's
            # parallel lists consistent after swap-and-pop mutates them.
            self._remove_entry(victim, rebuild_index=True)
            _skipped_indices.clear()  # Reset: indices shifted after removal
            _evicted_any = True
            logger.info(
                f"KV prefix cache evicted entry via {type(self._eviction_strategy).__name__} (capacity)"
            )
        if not _evicted_any and len(self._prompts) >= self._max_entries:
            logger.warning(
                "KV prefix cache at capacity but no entries evictable — cache will exceed max_entries"
            )

    def evict_under_pressure(self, threshold_pct: float = 85.0) -> int:
        """Evict LRU entries when GPU memory is under pressure.

        Checks MLX active memory against max recommended working set.
        Evicts least-recently-used entries until utilization drops below
        threshold or cache is empty.

        Proactive eviction prevents OOM on Apple
        Silicon UMA where GPU and CPU share the same memory pool.

        Args:
            threshold_pct: Memory utilization percentage to trigger eviction.

        Returns:
            Number of entries evicted.
        """
        try:
            info = mx.device_info()
            max_ws = info.get("max_recommended_working_set_size") if isinstance(info, dict) else None
            if max_ws is None or max_ws <= 0:
                return 0
            active = mx.get_active_memory()
            util_pct = (active / max_ws) * 100

            if util_pct < threshold_pct:
                return 0

            with self._lock:
                if not self._prompts:
                    return 0
                evicted = 0
                max_evict = max(1, len(self._prompts) // 4)
                _skipped_indices: set[int] = set()

                while self._prompts and evicted < max_evict and len(_skipped_indices) < len(self._prompts):
                    active = mx.get_active_memory()
                    if (active / max_ws) * 100 < threshold_pct - 5.0:
                        break

                    if isinstance(self._eviction_strategy, SLRUStrategy):
                        self._eviction_strategy.update_access_counts(self._access_counts)
                    # pass exclude=_skipped_indices (same as _evict_if_full).
                    # Without it, a pinned victim is re-selected every iteration and the
                    # `victim in _skipped_indices: break` below aborts ALL pressure eviction
                    # on the FIRST pinned entry — leaving other evictable entries in place and
                    # pressure unrelieved (OOM risk under the engine loop's active-request
                    # pinning). With exclude, select_victim skips past pinned entries.
                    victim = self._eviction_strategy.select_victim(
                        self._prompts, self._last_used, self._access_counter,
                        self._priorities, exclude=_skipped_indices,
                    )
                    if victim is None or victim in _skipped_indices:
                        break
                    if self._block_evict_checker is not None and self._block_hashes[victim]:
                        skip = False
                        for bh in self._block_hashes[victim]:
                            if not self._block_evict_checker(bh):
                                skip = True
                                break
                        if skip:
                            _skipped_indices.add(victim)
                            continue
                    self._remove_entry(victim, rebuild_index=False)
                    _skipped_indices.clear()  # Indices shifted after removal
                    evicted += 1

                if evicted > 0:
                    self._rebuild_hash_index()

            mx.clear_cache()
            logger.info(
                f"KV prefix cache pressure eviction: {evicted} entries freed "
                f"(utilization was {util_pct:.1f}%)"
            )
            return evicted

        except Exception:
            logger.debug("memory pressure check failed", exc_info=True)
            return 0

    def clear(self) -> None:
        """Clear all cached entries."""
        with self._lock:
            self._prompts.clear()
            self._caches.clear()
            self._block_hashes.clear()
            self._warm_flags.clear()  # was missing → desynced
            self._last_used.clear()
            self._priorities.clear()
            self._access_counts.clear()
            self._hash_index.clear()
            self._prefix_index.clear()
            self._block_refcount.clear()
            self._hash_collisions = 0
            self._access_counter = 0
            self._total_lookups = 0
            self._total_hits = 0
        gc.collect()
        mx.clear_cache()

    @property
    def size(self) -> int:
        return len(self._prompts)

    def set_priority(self, index: int, priority: int) -> None:
        """Set eviction priority for a cached entry (higher = kept longer)."""
        with self._lock:
            if 0 <= index < len(self._priorities):
                self._priorities[index] = priority

    def get_stats(self) -> dict:
        with self._lock:
            total_tokens = sum(len(p) for p in self._prompts)
            total_blocks = sum(len(bh) for bh in self._block_hashes)
            unique_blocks = len(self._block_refcount)
            shared_blocks = sum(1 for c in self._block_refcount.values() if c > 1)
            hit_rate = (
                round(self._total_hits / self._total_lookups, 4)
                if self._total_lookups > 0
                else 0.0
            )
            stats = {
                "entries": len(self._prompts),
                "hot_entries": sum(1 for w in self._warm_flags if not w),
                "warm_entries": sum(1 for w in self._warm_flags if w),
                "hot_limit": self._hot_limit,
                "max_entries": self._max_entries,
                "total_cached_tokens": total_tokens,
                "total_cached_blocks": total_blocks,
                "unique_blocks": unique_blocks,
                "shared_blocks": shared_blocks,
                "prefix_index_size": len(self._prefix_index),
                "min_prefix_length": self._min_prefix,
                "block_size": _BLOCK_SIZE,
                "eviction_strategy": type(self._eviction_strategy).__name__,
                "hash_collisions": self._hash_collisions,
                "total_lookups": self._total_lookups,
                "total_hits": self._total_hits,
                "hit_rate": hit_rate,
            }
            if self._ssd_cache is not None:
                try:
                    stats["ssd_cache"] = self._ssd_cache.get_stats()
                except Exception:
                    logger.debug("SSD cache stats unavailable", exc_info=True)
            return stats

    def enable_ssd_cache(
        self,
        cache_dir: str = "~/.cache/yunshu/kv-ssd",
        max_size_bytes: int = 10 * 1024 ** 3,
        model_name: str = "",
    ) -> None:
        """Enable SSD-tier KV cache persistence.

        After enabling, blocks saved to the prefix cache are also persisted
        to disk. On restart, previously cached blocks are recovered.
        """
        from .ssd_kv_cache import SSDKVCache
        # CRITICAL (cross-model KV corruption): the per-block SSD store
        # is keyed purely by the content block-hash (_canonical_block_hash omits
        # model identity for cross-subsystem yunshu_kv compatibility), and
        # has_block()/load_block() never check the stored model_name. With a
        # SINGLE shared cache_dir for every model (the default), two different
        # models that share a prompt prefix produce identical block hashes →
        # has_block() reports a hit on a block written by the OTHER model →
        # foreign KV loaded into attention → garbage output or a shape-mismatch
        # crash. Namespace the on-disk dir per model so each model's persisted KV
        # is physically isolated (covers both the per-block store and the hybrid
        # whole-snapshot store) without perturbing the in-memory hash chain.
        scoped_dir = self._scoped_ssd_dir(cache_dir, model_name)
        self._ssd_cache = SSDKVCache(
            cache_dir=scoped_dir,
            max_size_bytes=max_size_bytes,
        )
        self._ssd_model_name = model_name
        # whole-snapshot SSD store for HYBRID models (the
        # per-block SSD path can't serialize ArraysCache recurrent state).
        try:
            from .hybrid_ssd_snapshot import HybridSnapshotStore
            self._hybrid_ssd = HybridSnapshotStore(scoped_dir)
        except Exception:
            self._hybrid_ssd = None
            logger.debug("hybrid SSD snapshot store init failed", exc_info=True)
        logger.info(f"SSD KV cache enabled: dir={scoped_dir}, max={max_size_bytes / 1024**3:.0f}GB")

    @staticmethod
    def _scoped_ssd_dir(cache_dir: str, model_name: str) -> str:
        """Namespace the SSD cache dir per model so different models never share
        content-hash-keyed KV blocks on disk. Empty model_name → the
        base dir unchanged (back-compat / single-model deployments)."""
        import os
        import re
        base = os.path.expanduser(cache_dir)
        if not model_name:
            return base
        # Stable, filesystem-safe per-model subdir: readable suffix + a short
        # digest of the FULL name to disambiguate names that sanitize identically.
        safe = re.sub(r"[^A-Za-z0-9._-]", "_", model_name).strip("_")[-48:]
        digest = hashlib.blake2b(model_name.encode("utf-8"), digest_size=6).hexdigest()
        return os.path.join(base, f"{safe}-{digest}" if safe else digest)

    def flush_to_ssd(self) -> int:
        """Flush all in-memory cache entries to SSD.

        Returns the number of blocks written.
        """
        if self._ssd_cache is None:
            return 0

        with self._lock:
            snapshot_prompts = list(self._prompts)
            # Deep-copy cache data under lock to prevent GPU-freed tensors
            # from concurrent mx.clear_cache(). Without detached copies,
            # the snapshot holds references to live GPU tensors that may be
            # reclaimed between the lock release and the SSD write loop.
            snapshot_caches = [
                self._snapshot_cache(c) if c is not None else None
                for c in self._caches
            ]
            snapshot_hashes = list(self._block_hashes)

        count = 0
        for i, (prompt, cache) in enumerate(zip(snapshot_prompts, snapshot_caches, strict=False)):
            # Skip non-block-decomposable backbones (sliding-window / recurrent):
            # spilling them wastes disk and the restore always fails.
            if cache is not None and not self._is_block_decomposable(cache):
                continue
            block_hashes = snapshot_hashes[i]
            for bi, bh in enumerate(block_hashes):
                bh_bytes = bh.to_bytes(8, "little") if isinstance(bh, int) else bh
                if self._ssd_cache.has_block(bh_bytes):
                    continue
                tokens = np_array(prompt)
                start = bi * _BLOCK_SIZE
                end = min(start + _BLOCK_SIZE, len(tokens))
                block_cache = self._extract_block_cache_data(cache, bi)
                self._ssd_cache.save_block(
                    block_hash=bh_bytes,
                    cache_data=block_cache,
                    token_count=end - start,
                    model_name=self._ssd_model_name,
                )
                count += 1

        logger.info(f"SSD KV cache flushed {count} blocks")
        return count

    def restore_prefix_from_ssd(self, query_blocks) -> tuple:
        """Reassemble the longest run of consecutive SSD-resident prefix blocks
        into a contiguous KVCache list usable by BatchGenerator.insert_segments.

        The prior SSD restore only loaded a SINGLE block and
        returned a structure that wasn't a valid cache (empty output). This loads
        every consecutive matching block (block 0..N), int8-dequants each layer,
        concatenates along the seq axis (-2), and builds proper KVCache objects
        with the correct offset. int8-LOSSY by design (KV stored quantized), so
        output is coherent but not byte-identical to a full-precision prefill.
        Returns (cache_list, n_tokens) or (None, 0).
        """
        if self._ssd_cache is None or not query_blocks:
            return None, 0
        loaded = []
        for bh in query_blocks:
            bh_bytes = bh.to_bytes(8, "little") if isinstance(bh, int) else bh
            if not self._ssd_cache.has_block(bh_bytes):
                break
            blk = self._ssd_cache.load_block(bh_bytes)
            if not blk:
                break
            loaded.append(blk)
        if not loaded:
            return None, 0
        # HYBRID models (e.g. Qwen3.5) have linear-attention
        # layers backed by ArraysCache — a FIXED-SIZE recurrent state, not a
        # per-token KV tensor. Such state cannot be sliced per block nor
        # concatenated along a sequence axis, so a per-block SSD prefix cannot be
        # reassembled into a correct cache. Detect this up front and decline
        # (clean fall-back to a full prefill) instead of bailing mid-layer or,
        # worse, splicing recurrent state and emitting garbage. SSD prefix
        # restore is therefore a STANDARD-model (all-KVCache) feature.
        for ld in loaded[0]:
            if not isinstance(ld, (tuple, list)) and getattr(ld, "keys", None) is None:
                logger.debug(
                    "SSD prefix restore declined: hybrid model (linear-attn "
                    "ArraysCache layer is not block-decomposable) — full prefill"
                )
                return None, 0
        try:
            from mlx_lm.models.cache import KVCache
            num_layers = len(loaded[0])
            cache_list = []
            total = 0
            for li in range(num_layers):
                ks, vs = [], []
                for blk in loaded:
                    ld = blk[li]
                    if isinstance(ld, (tuple, list)):
                        k, v = ld[0], ld[1]
                    else:
                        k = getattr(ld, "keys", None)
                        v = getattr(ld, "values", None)
                    if k is None or v is None:
                        return None, 0
                    ks.append(k)
                    vs.append(v)
                # Concatenate along the sequence axis. Use -2 (not the absolute
                # 2) so it stays correct if a cache ever exposes 3-D keys/values
                # rather than the usual [B, H, S, D].
                ck = mx.concatenate(ks, axis=-2).astype(mx.bfloat16)
                cv = mx.concatenate(vs, axis=-2).astype(mx.bfloat16)
                c = KVCache()
                c.keys = ck
                c.values = cv
                c.offset = ck.shape[-2]
                cache_list.append(c)
                total = ck.shape[-2]
            return cache_list, total
        except Exception:
            logger.warning("SSD multi-block reassembly failed", exc_info=True)
            return None, 0

    def try_ssd_restore(self, block_hash: bytes) -> list | None:
        """Try to restore a block from SSD cache (GUARDED OFF by default).

        The SSD SAVE path is now fixed+verified (persists
        int8-quantized KV blocks), but the RESTORE is incomplete — the get()
        caller restores only a SINGLE 64-token block and the reconstructed
        single-block structure is not a valid cache for insert_segments, so it
        produced EMPTY output. A correct restore needs multi-block reassembly
        into a contiguous offset-correct cache + int8 dequant (lossy by design).
        Until that exists, return None so a miss falls back to a correct full
        prefill. Enable experimentally with YUNSHU_SSD_KV_RESTORE=1.
        """
        if self._ssd_cache is None:
            return None
        import os
        if os.environ.get("YUNSHU_SSD_KV_RESTORE", "").strip() not in ("1", "true", "yes"):
            return None
        return self._ssd_cache.load_block(block_hash)

    def close(self) -> None:
        """Flush SSD cache and release resources."""
        if self._ssd_cache is not None:
            try:
                self._ssd_cache.close()
            except Exception:
                logger.debug("SSD cache close failed", exc_info=True)
