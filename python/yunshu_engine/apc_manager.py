# Patches upstream mlx-vlm symbols (see vendor.json kind=patches; `just vendor-check` flags source changes)
"""Prefix-cache policy for the VLM runner: what mlx-vlm's APC stores, keeps and reports.

mlx-vlm's ``APCManager`` keeps *exact checkpoints* for hybrid (GDN) models: a full copy of the
attention KV plus the recurrent state at one token position, reusable only by a prompt that
starts with exactly those tokens. Its defaults are tuned for one conversation:

- two entries in total, and each request stores two (the prompt end and one interval
  boundary), so a second agent session (a subagent, a title request, another terminal) evicts
  the first session's cache and every request re-prefills its whole context;
- the older checkpoints of a growing conversation stay resident although the newest one
  supersedes them, wasting most of the byte budget on stale copies;
- a new session that shares only the system prompt and tool list finds nothing to reuse.

``YunshuAPCManager`` changes that policy and nothing else (hits are the same exact states, so
output is identical to a cold prefill):

- ``max_entries`` is large; the byte budget is the limit;
- a request's checkpoints supersede earlier-request checkpoints that are prefixes of it;
- a checkpoint at the end of the system turn is kept for every distinct head (system prompt +
  tools), so a new session in the same project reuses it;
- every lookup is recorded (tier ``ram`` / ``ssd`` / ``none``, cached tokens, reload time) so
  the API can report cache provenance per request.

Registered in vendor.json (patches: the overridden symbols; the byte-size estimates and the
checkpoint lengths are the upstream ones).
"""

from __future__ import annotations

import collections
import contextlib
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any

from mlx_vlm.apc import APCManager, DiskBlockStore, _sequence_hash
from mlx_vlm.apc_coordinator import APCCoordinator

logger = logging.getLogger(__name__)

GIB = 1 << 30

# Entry-count cap: the byte budget decides, this only bounds bookkeeping.
MAX_ENTRIES = 16
# Reserve for the OS, activations and the live decode KV when sizing the budget from memory.
_RESERVE_GIB = 16.0
_MIN_GIB = 4.0
_MAX_GIB = 32.0


def auto_memory_gb(total_bytes: int, weights_bytes: int) -> float:
    """Default APC RAM budget: half of what is left after the weights and a fixed reserve,
    clamped to [4, 32] GiB (128 GB machine + 16 GB model: 32; 64 GB: 16; 32 GB: 4)."""
    free = total_bytes / GIB - weights_bytes / GIB - _RESERVE_GIB
    return float(min(_MAX_GIB, max(_MIN_GIB, 0.5 * free)))


def total_memory_bytes() -> int:
    import os

    return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")


class SpillDiskStore(DiskBlockStore):
    """SSD tier that receives a checkpoint when RAM evicts it, not when it is stored.

    Upstream writes every checkpoint through to disk as it is stored: a growing agent
    conversation then writes its whole context (gigabytes) after every request, tens of
    terabytes a day of SSD wear for one busy session. Here a checkpoint that stays in RAM is
    never written; one that RAM evicts is (unless a newer checkpoint of the same
    conversation superseded it), so the SSD holds what RAM had to give up and the cache
    survives a restart (``YunshuAPCManager.close`` writes what is still resident).
    """

    def save_exact_cache(
        self, cache_hash, token_ids, extra_hash, prompt_cache, *, synchronous=False
    ) -> bool:
        if not synchronous:
            return True  # kept in RAM for now; written on eviction
        # synchronous: too large to retain in RAM, so this is the only copy
        return self.write_now(cache_hash, token_ids, extra_hash, prompt_cache, True)

    def write_now(
        self, cache_hash, token_ids, extra_hash, prompt_cache, synchronous=False
    ) -> bool:
        return super().save_exact_cache(
            cache_hash, token_ids, extra_hash, prompt_cache, synchronous=synchronous
        )


class _SpillingDict(collections.OrderedDict):
    """The exact-checkpoint table; LRU eviction (``popitem(last=False)``) spills to SSD."""

    def __init__(self, spill):
        super().__init__()
        self._spill = spill

    def popitem(self, last=True):
        key, entry = super().popitem(last)
        if not last:
            self._spill(key, entry)
        return key, entry


@dataclass
class Lookup:
    """One cache lookup: what a request found and where it came from."""

    prompt_len: int
    cached: int
    tier: str  # "ram" | "ssd" | "none"
    ms: float
    t: float  # time.perf_counter() when the lookup finished


class _Coordinator(APCCoordinator):
    """Checkpoint positions: the prompt end, one interval boundary, the end of the system turn."""

    def checkpoint_lengths(self, token_ids, media_token_ids):
        final = self.checkpoint_len(token_ids, media_token_ids)
        if final <= 0:
            return []
        mgr = self.manager
        lengths = {final}
        interval = mgr.checkpoint_interval_tokens
        if interval > 0 and mgr.keep_interval_checkpoint:
            from mlx_vlm.apc import adjust_prefix_to_text_suffix_boundary

            block = mgr.block_size
            interval = ((interval + block - 1) // block) * block
            last = ((final - 1) // interval) * interval
            last = adjust_prefix_to_text_suffix_boundary(
                token_ids, last, media_token_ids, max_prefix_tokens=final
            )
            if mgr.exact_cache_min_tokens <= last < final:
                lengths.add(last)
        head = mgr.head_boundary(token_ids)
        if head and mgr.exact_cache_min_tokens <= head < final:
            lengths.add(head)
            mgr.note_head(token_ids[:head])
        mgr.begin_request()
        return sorted(lengths)


class YunshuAPCManager(APCManager):
    def __init__(
        self,
        *args: Any,
        head_marker: tuple[int, int] | None = None,
        keep_interval_checkpoint: bool = True,
        max_entries: int = MAX_ENTRIES,
        **kwargs: Any,
    ):
        overrides = dict(kwargs.pop("overrides", None) or {})
        overrides.setdefault("checkpoint_entries", max_entries)
        super().__init__(*args, overrides=overrides, **kwargs)
        # (<|im_start|>, user) token ids: the end of the system turn is the head boundary.
        self.head_marker = head_marker
        self.keep_interval_checkpoint = keep_interval_checkpoint
        self.lookups: collections.deque[Lookup] = collections.deque(maxlen=64)
        self._generation = 0
        self._born: dict[int, int] = {}
        self._head_keys: set[int] = set()
        self._head_lengths: set[int] = set()
        self._plock = threading.Lock()
        if isinstance(self.disk, SpillDiskStore):
            self._exact_cache = _SpillingDict(self._spill)

    # ── SSD spill ──────────────────────────────────────────────────────
    def _spill(self, key, entry) -> None:
        disk = self.disk
        if not isinstance(disk, SpillDiskStore):
            return
        try:
            disk.write_now(key, entry.token_ids, entry.extra_hash, entry.prompt_cache)
        except Exception:
            logger.warning("APC: SSD spill failed", exc_info=True)

    def close(self) -> None:
        """Write what is still resident so the cache survives a restart."""
        if isinstance(self.disk, SpillDiskStore):
            with self.lock:
                entries = list(self._exact_cache.items())
            for key, entry in entries:
                self._spill(key, entry)
            with contextlib.suppress(Exception):
                self.disk.flush()
        super().close()

    # ── policy ─────────────────────────────────────────────────────────
    def coordinator(self, model: Any) -> APCCoordinator:
        return _Coordinator(self, model)

    def head_boundary(self, token_ids) -> int:
        """Token index where the first user turn starts (the end of the system turn)."""
        if self.head_marker is None:
            return 0
        a, b = self.head_marker
        n = len(token_ids)
        for i in range(1, n - 1):
            if token_ids[i] == a and token_ids[i + 1] == b:
                return i
        return 0

    def begin_request(self) -> None:
        with self._plock:
            self._generation += 1

    def note_head(self, head_tokens, extra_hash: int = 0) -> None:
        with self._plock:
            self._head_lengths.add(len(head_tokens))

    def store_exact_cache(self, token_ids, prompt_cache, *, extra_hash=0) -> bool:
        n = len(token_ids)
        with self._plock:
            gen = self._generation
            is_head = n in self._head_lengths
        self._supersede(token_ids, extra_hash, gen)
        ok = super().store_exact_cache(token_ids, prompt_cache, extra_hash=extra_hash)
        if ok:
            key = _sequence_hash(
                tuple(int(t) for t in token_ids), extra_hash, self.block_size
            )
            with self._plock:
                self._born[key] = gen
                if is_head:
                    self._head_keys.add(key)
        return ok

    def _supersede(self, token_ids, extra_hash: int, gen: int) -> None:
        """Drop RAM checkpoints of earlier requests that are prefixes of this one."""
        tokens = tuple(int(t) for t in token_ids)
        dropped = 0
        with self.lock:
            for key, entry in list(self._exact_cache.items()):
                stored = entry.token_ids
                if (
                    entry.extra_hash != extra_hash
                    or len(stored) >= len(tokens)
                    or key in self._head_keys
                    or self._born.get(key, gen) >= gen
                    or tokens[: len(stored)] != stored
                ):
                    continue
                del self._exact_cache[key]
                self._born.pop(key, None)
                dropped += 1
        if dropped:
            logger.debug("APC: superseded %d earlier checkpoint(s)", dropped)

    # ── provenance ─────────────────────────────────────────────────────
    def lookup_exact_cache(self, token_ids, *args, **kwargs):
        before = self.stats.disk_hits
        t0 = time.perf_counter()
        cache, n = super().lookup_exact_cache(token_ids, *args, **kwargs)
        ms = (time.perf_counter() - t0) * 1000.0
        tier = ("ssd" if self.stats.disk_hits > before else "ram") if n else "none"
        self.lookups.append(
            Lookup(len(token_ids), int(n), tier, round(ms, 1), time.perf_counter())
        )
        return cache, n

    def lookup_prefix(self, token_ids, *args, **kwargs):
        """Block-mode (plain KV) families: the in-RAM block pool."""
        t0 = time.perf_counter()
        matched, n = super().lookup_prefix(token_ids, *args, **kwargs)
        ms = (time.perf_counter() - t0) * 1000.0
        self.lookups.append(
            Lookup(
                len(token_ids),
                int(n),
                "ram" if n else "none",
                round(ms, 1),
                time.perf_counter(),
            )
        )
        return matched, n

    def lookup_prefix_disk_cache(self, token_ids, *args, **kwargs):
        t0 = time.perf_counter()
        cache, n = super().lookup_prefix_disk_cache(token_ids, *args, **kwargs)
        ms = (time.perf_counter() - t0) * 1000.0
        if n:
            self.lookups.append(
                Lookup(len(token_ids), int(n), "ssd", round(ms, 1), time.perf_counter())
            )
        return cache, n

    def take_lookup(
        self, prompt_len: int, cached: int, since: float = 0.0
    ) -> Lookup | None:
        """The most recent lookup (after ``since``, a perf_counter time) of a prompt this
        long that found ``cached`` tokens."""
        for rec in reversed(self.lookups):
            if rec.t < since:
                break
            if rec.prompt_len == prompt_len and rec.cached == cached:
                return rec
        return None

    def snapshot(self) -> dict:
        """Counters and occupancy for /debug/kv-cache and /metrics."""
        snap = self.stats_snapshot()
        with self.lock:
            snap["entries"] = len(self._exact_cache)
            snap["entry_tokens"] = [
                len(e.token_ids) for e in self._exact_cache.values()
            ]
        snap["head_checkpoints"] = len(self._head_keys & set(self._exact_cache))
        return snap
