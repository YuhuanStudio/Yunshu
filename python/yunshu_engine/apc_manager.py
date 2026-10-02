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
# Shutdown spill budget: a service stop must not hang on a 32 GiB cache.
CLOSE_FLUSH_SECONDS = 20.0
# What the OS, activations and the live decode KV need, as a share of the machine, between
# these bounds (8 / 16 GB machines: 4 GiB, 32 GB: 8, 64 GB and up: 16).
_RESERVE_SHARE = 0.25
_RESERVE_MIN_GIB = 4.0
_RESERVE_MAX_GIB = 16.0
# The cache never takes more than this share of the machine, whatever is left over.
_MAX_SHARE = 0.25
# Under this the cache is not worth its footprint (about 8K tokens of a 27B checkpoint).
_MIN_GIB = 1.0
_MAX_GIB = 32.0


def auto_memory_gb(total_bytes: int, weights_bytes: int) -> float:
    """Default APC RAM budget in GiB: half of what is left after the weights and the OS /
    activation reserve, at most a quarter of the machine and 32 GiB; 0 (cache off) when less
    than 1 GiB would be left to it, so a small machine never gives away memory the model
    needs (128 GB + 16 GB model: 32; 64 GB: 16; 32 GB: 4; 16 GB with 16 GB of weights or an
    8 GB machine with a 4 GB model: 0)."""
    total = total_bytes / GIB
    reserve = min(_RESERVE_MAX_GIB, max(_RESERVE_MIN_GIB, _RESERVE_SHARE * total))
    free = total - weights_bytes / GIB - reserve
    budget = min(_MAX_GIB, _MAX_SHARE * total, 0.5 * free)
    return float(budget) if budget >= _MIN_GIB else 0.0


def total_memory_bytes() -> int:
    import os

    return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")


def check_loaded_cache(
    token_ids, prompt_cache, template=None, kv_heads=None, head_dim=None
):
    """Structural check of an exact checkpoint read back from disk; True when it is
    consistent with the live model.

    ``template`` is ``model.make_cache()`` (same layer count / cache classes); ``kv_heads``
    and ``head_dim`` (optional) pin the attention geometry. Offsets must agree with the
    token count and every tensor must have a plausible rank, dtype and batch of one. A
    corrupt or foreign file fails here and the caller falls back to a cold prefill.
    """
    n_tokens = len(token_ids)
    if not prompt_cache or n_tokens <= 0:
        return False
    if template is not None and len(template) != len(prompt_cache):
        return False
    for i, c in enumerate(prompt_cache):
        if template is not None and type(template[i]).__name__ != type(c).__name__:
            return False
        keys, values = getattr(c, "keys", None), getattr(c, "values", None)
        if keys is not None or values is not None:
            if keys is None or values is None or isinstance(keys, (tuple, list)):
                return False
            offset = int(getattr(c, "offset", 0))
            if (
                keys.ndim != 4
                or keys.shape[0] != 1
                or keys.shape[:3] != values.shape[:3]
                or keys.dtype != values.dtype
                or not 0 < offset <= min(keys.shape[2], n_tokens)
                or (kv_heads is not None and keys.shape[1] != kv_heads)
                or (head_dim is not None and keys.shape[3] != head_dim)
            ):
                return False
            continue
        state = getattr(c, "cache", None)
        if state is None:
            state = getattr(c, "state", None)
        if state is None:
            continue  # an empty layer
        if template is not None:
            tstate = getattr(template[i], "cache", None)
            if tstate is not None and len(tstate) != len(state):
                return False
        for a in state:
            if a is None:
                continue
            if a.ndim < 2 or a.shape[0] != 1 or a.size == 0:
                return False
    return True


class SpillDiskStore(DiskBlockStore):
    """SSD tier that receives a checkpoint when RAM evicts it, not when it is stored.

    Upstream writes every checkpoint through to disk as it is stored: a growing agent
    conversation then writes its whole context (gigabytes) after every request, tens of
    terabytes a day of SSD wear for one busy session. Here a checkpoint that stays in RAM is
    never written; one that RAM evicts is (unless a newer checkpoint of the same
    conversation superseded it), so the SSD holds what RAM had to give up and the cache
    survives a restart (``YunshuAPCManager.close`` writes what is still resident).
    """

    # Set by the engine: ``validator(token_ids, prompt_cache) -> bool`` on a loaded
    # checkpoint (see ``check_loaded_cache``). None accepts what the format checks pass.
    validator = None
    # The root's shared DiskBudget (yunshu_kv.disk_budget): global cap across namespaces,
    # free-space reserve, pause after a failed write. None keeps upstream's per-namespace cap.
    budget = None

    def attach_budget(self, budget) -> None:
        """Put this namespace under the root-wide budget."""
        self.budget = budget
        ns = self.dir.name
        budget.register_owner(ns, self._evict_path, self._in_flight_paths)
        self._budget_ns = ns

    def _in_flight_paths(self) -> set:
        with self._in_flight_lock:
            hashes = set(self._in_flight)
        with self._index_lock:
            paths = {self._index[h][0] for h in hashes if h in self._index}
            paths.update(self._exact_index[h] for h in hashes if h in self._exact_index)
        return paths

    def _evict_path(self, path) -> bool:
        """Drop one of this namespace's files on the budget's behalf (index and all)."""
        if path in self._in_flight_paths():
            return False
        try:
            size = path.stat().st_size
        except OSError:
            size = 0
        self._drop_index_for_path(path)
        path.unlink(missing_ok=True)
        self._disk_bytes = max(0, self._disk_bytes - size)
        self.evictions += 1
        return True

    def _maybe_evict(self) -> int:
        budget = self.budget
        if budget is None:
            return super()._maybe_evict()
        before = self.evictions
        try:
            budget.enforce(keep={self._budget_ns})
        except Exception:
            logger.warning("APC disk: budget enforcement failed", exc_info=True)
        return self.evictions - before

    def _write_payload(self, shard_id, block_hashes, payload) -> bool:
        budget = self.budget
        if budget is None:
            return super()._write_payload(shard_id, block_hashes, payload)
        from mlx_vlm.apc import _cache_nbytes

        path = self._shard_path(shard_id)
        try:
            exact = hasattr(payload, "prompt_cache")
            size = _cache_nbytes(
                payload.prompt_cache
                if exact
                else payload.layer_keys + payload.layer_values
            )
        except Exception:
            size = 0
        if not budget.allow_write(size):
            self._notify_write_callback(self._write_failure_callback, len(block_hashes))
            return False
        try:
            # mirrors upstream's _write_payload, but keeps the error: a failed write drops
            # this checkpoint, leaves no temp file, and pauses spilling (one warning)
            with self._write_lock:
                if hasattr(payload, "prompt_cache"):
                    self._write_exact_cache_snapshot(path, payload)
                else:
                    self._write_layer_major_snapshot(path, payload)
        except Exception as e:
            from yunshu_kv.disk_budget import sweep_tmp

            sweep_tmp(path.parent, path.stem)
            if isinstance(e, OSError):
                budget.record_failure(e)
            else:  # this one checkpoint cannot be serialized; the disk is fine
                logger.warning("APC disk: checkpoint not saved (%s)", e)
            self._notify_write_callback(self._write_failure_callback, len(block_hashes))
            return False
        budget.note_written(path)
        self._maybe_evict()  # now that the new file is counted
        self._notify_write_callback(self._write_success_callback, len(block_hashes))
        return True

    def quarantine(self, path, why: str) -> None:
        """Invalidate one persisted file: delete it and forget it, so it is never read
        again and the lookup that hit it prefills cold."""
        logger.warning(
            "APC disk: %s unusable (%s), dropped", getattr(path, "name", path), why
        )
        self.invalidated = getattr(self, "invalidated", 0) + 1
        try:
            self._drop_index_for_path(path)
        finally:
            try:
                size = path.stat().st_size
                path.unlink()
                self._disk_bytes = max(0, self._disk_bytes - size)
            except OSError:
                pass

    def _file_complete(self, path) -> bool:
        """The file's tensor payload is all there (a torn write or truncation is not)."""
        parsed = self._open_shard_header(path)
        if parsed is None:
            return False
        entries, _meta, data_start = parsed
        try:
            end = max(
                (
                    int(e["data_offsets"][1])
                    for e in entries.values()
                    if "data_offsets" in e
                ),
                default=0,
            )
            return bool(path.stat().st_size >= data_start + end)
        except (OSError, KeyError, TypeError, ValueError, IndexError):
            return False

    def find_exact_prefix(self, *args, **kwargs):
        try:
            return super().find_exact_prefix(*args, **kwargs)
        except Exception:
            logger.warning("APC disk: index lookup failed, going cold", exc_info=True)
            return None

    def load_exact_cache(self, cache_hash, **kwargs):
        with self._index_lock:
            path = self._exact_index.get(cache_hash)
        try:
            loaded = super().load_exact_cache(cache_hash, **kwargs)
        except Exception as e:
            if path is not None:
                self.quarantine(path, f"load raised {type(e).__name__}")
            return None
        if loaded is None:
            if path is not None and path.exists() and not self._file_complete(path):
                self.quarantine(path, "torn or truncated")
            return None
        if self.validator is not None:
            try:
                ok = bool(self.validator(loaded[0], loaded[2]))
            except Exception:
                ok = False
            if not ok:
                if path is not None:
                    self.quarantine(path, "structure does not match the model")
                return None
        return loaded

    def load_layer_major_prefix(self, *args, **kwargs):
        try:
            return super().load_layer_major_prefix(*args, **kwargs)
        except Exception:
            logger.warning("APC disk: block restore failed, going cold", exc_info=True)
            return None

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


@dataclass
class _Held:
    """What the SSD writer needs of a checkpoint that is not (or no longer) a HOT entry."""

    token_ids: tuple
    extra_hash: int
    prompt_cache: list


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
    tier: str  # "ram" | "warm" | "ssd" | "none"
    ms: float
    t: float  # time.perf_counter() when the lookup finished
    device: str | None = None  # storage tier (volume name) of an ssd hit


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
        if head:
            from mlx_vlm.apc import adjust_prefix_to_text_suffix_boundary

            # media tokens (an image inside the system turn) must not be cut in half
            if (
                adjust_prefix_to_text_suffix_boundary(
                    token_ids, head, media_token_ids, max_prefix_tokens=final
                )
                != head
            ):
                head = 0
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
        warm_mode: str = "off",
        warm_bytes: int = 0,
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
        self.warm = None
        self._promoted: tuple[int, float] | None = None
        self.tier_hits: collections.Counter = collections.Counter()
        if warm_mode != "off" and warm_bytes > 0:
            from .apc_warm import WarmTier

            try:
                self.warm = WarmTier(warm_mode, warm_bytes)
            except ImportError as e:
                logger.warning(
                    "APC WARM tier '%s' needs the 'compression' extra (%s); WARM off",
                    warm_mode,
                    e,
                )
        if isinstance(self.disk, SpillDiskStore) or self.warm is not None:
            self._exact_cache = _SpillingDict(self._demote)

    # ── demotion: HOT -> WARM -> SSD ───────────────────────────────────
    def _spill(self, key, entry) -> None:
        disk = self.disk
        if not isinstance(disk, SpillDiskStore):
            return
        try:
            disk.write_now(key, entry.token_ids, entry.extra_hash, entry.prompt_cache)
        except Exception as e:
            logger.warning("APC: SSD spill failed (%s)", e)

    def _demote(self, key, entry) -> None:
        """A HOT entry was evicted: it moves to WARM (compact form in RAM) when there is one and
        to SSD otherwise. Lossy WARM keeps the SSD copy exact, so the exact bytes go to SSD now."""
        warm = self.warm
        if warm is None:
            self._spill(key, entry)
            return
        self._settle_warm()
        with self._plock:
            born = self._born.get(key, 0)
            head = key in self._head_keys
        if warm.lossy:
            self._spill(key, entry)  # exact states stay the SSD's invariant
        try:
            taken = warm.demote(key, entry, born=born, head=head)
        except Exception as e:
            logger.warning("APC warm: demotion failed (%s)", e)
            taken = False
        if not taken and not warm.lossy:
            self._spill(key, entry)

    def _settle_warm(self) -> None:
        """Absorb finished compressions; entries the WARM budget pushes out go to SSD (lossless
        mode: decoded back to exact arrays) or are dropped (lossy mode: SSD already has them)."""
        warm = self.warm
        if warm is None:
            return
        for key, ent in warm.drain():
            if warm.lossy:
                warm.stats.dropped += 1
                continue
            try:
                cache = warm.decode_entry(ent)
            except Exception as e:
                warm.stats.corrupt += 1
                logger.warning("APC warm: evicted entry unusable (%s), dropped", e)
                continue
            self._spill(key, _Held(ent.token_ids, ent.extra_hash, cache))
            warm.stats.evicted_to_ssd += 1
        for key, inf in warm.failed():
            self._spill(key, _Held(inf.entry_tokens, inf.extra_hash, inf.prompt_cache))

    def _promote_warm(
        self, tokens, extra_hash, max_prefix_tokens, min_prefix_tokens
    ) -> None:
        """If WARM holds a longer reusable prefix than HOT, decode it into HOT so the regular
        lookup serves it (the same exact states a HOT hit would use in lossless mode)."""
        warm = self.warm
        self._promoted = None
        if warm is None:
            return
        self._settle_warm()
        max_len = len(tokens) - 1
        if max_prefix_tokens is not None and max_prefix_tokens > 0:
            max_len = min(max_len, int(max_prefix_tokens))
        hot_best = 0
        with self.lock:
            for entry in self._exact_cache.values():
                n = len(entry.token_ids)
                if (
                    entry.extra_hash == extra_hash
                    and hot_best < n <= max_len
                    and tokens[:n] == entry.token_ids
                ):
                    hot_best = n
        found = warm.find(tokens, extra_hash, max_len, max(min_prefix_tokens, hot_best))
        if found is None:
            return
        key, n, where = found
        t0 = time.perf_counter()
        if where == "inflight":
            inf = warm.take_inflight(key)
            if inf is None:
                return
            cache, born, head = inf.prompt_cache, inf.born, inf.head
        else:
            try:
                got = warm.take(key)
            except Exception:
                return  # corrupt: dropped, the lookup goes on to SSD / recompute
            if got is None:
                return
            cache, ent = got
            born, head = ent.born, ent.head
        from mlx_vlm.apc import APCExactCacheEntry, _cache_nbytes

        size = _cache_nbytes(cache)
        if size > self.memory_max_bytes or not self._make_room(size, retain_bytes=size):
            # no HOT room for the decoded copy: it is discarded and the lookup goes on to
            # SSD / recompute
            return
        with self.lock:
            self._exact_cache[key] = APCExactCacheEntry(
                tokens[:n], int(extra_hash), cache
            )
            self._exact_cache.move_to_end(key)
            while len(self._exact_cache) > self._exact_cache_max:
                self._exact_cache.popitem(last=False)
        with self._plock:
            self._born[key] = born
            if head:
                self._head_keys.add(key)
        self._promoted = (n, (time.perf_counter() - t0) * 1000.0)

    def close(self) -> None:
        """Write what is still resident so the cache survives a restart."""
        if isinstance(self.disk, SpillDiskStore):
            with self.lock:
                entries = list(self._exact_cache.items())
            # newest first, within a time budget: a service stop must not hang on a
            # 32 GiB cache; what does not fit in the budget is simply re-prefilled later
            deadline = time.monotonic() + CLOSE_FLUSH_SECONDS
            for key, entry in reversed(entries):
                if time.monotonic() > deadline:
                    logger.info("APC: shutdown spill budget spent; rest stays RAM-only")
                    break
                self._spill(key, entry)
            if self.warm is not None and not self.warm.lossy:
                self.warm.wait_idle(max(1.0, deadline - time.monotonic()))
                self._settle_warm()
                for key, ent in reversed(list(self.warm.entries.items())):
                    if time.monotonic() > deadline:
                        break
                    try:
                        cache = self.warm.decode_entry(ent)
                    except Exception:
                        continue
                    self._spill(key, _Held(ent.token_ids, ent.extra_hash, cache))
            with contextlib.suppress(Exception):
                self.disk.flush()
        if self.warm is not None:
            self.warm.close()
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
        if self.warm is not None:
            with self._plock:
                keep = set(self._head_keys)
            dropped += self.warm.supersede(tokens, extra_hash, gen, keep)
        if dropped:
            logger.debug("APC: superseded %d earlier checkpoint(s)", dropped)

    # ── provenance ─────────────────────────────────────────────────────
    def lookup_exact_cache(self, token_ids, *args, **kwargs):
        before = self.stats.disk_hits
        t0 = time.perf_counter()
        if self.warm is not None:
            tokens = tuple(int(t) for t in token_ids)
            names = ("extra_hash", "max_prefix_tokens", "min_prefix_tokens")
            defaults = (0, None, 0)
            vals = [
                kwargs.get(k, args[i] if i < len(args) else d)
                for i, (k, d) in enumerate(zip(names, defaults, strict=True))
            ]
            self._promote_warm(tokens, int(vals[0]), vals[1], vals[2])
        cache, n = super().lookup_exact_cache(token_ids, *args, **kwargs)
        ms = (time.perf_counter() - t0) * 1000.0
        tier = ("ssd" if self.stats.disk_hits > before else "ram") if n else "none"
        if n and self._promoted is not None and self._promoted[0] == n:
            tier = "warm"
        self._promoted = None
        self.tier_hits[tier] += 1
        device = getattr(self.disk, "last_device", None) if tier == "ssd" else None
        self.lookups.append(
            Lookup(
                len(token_ids), int(n), tier, round(ms, 1), time.perf_counter(), device
            )
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
        snap["tier_hits"] = dict(self.tier_hits)
        tiers = getattr(self.disk, "snapshot", None)
        if callable(tiers):
            snap["storage_tiers"] = tiers()
        if self.warm is not None:
            snap.update(self.warm.snapshot())
        return snap
