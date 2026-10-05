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
import contextvars
import logging
import threading
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import mlx_vlm.apc as _upstream_apc
from mlx_vlm.apc import APCManager, DiskBlockStore, _sequence_hash
from mlx_vlm.apc_coordinator import APCCoordinator

logger = logging.getLogger(__name__)

_CANONICAL_LOOKUP: contextvars.ContextVar[
    tuple[int, int, int, int, frozenset[int]] | None
] = contextvars.ContextVar("yunshu_canonical_apc_lookup", default=None)

GIB = 1 << 30

# ``APCManager``'s clone of a prompt cache (restoring a checkpoint, storing one) differs
# from upstream in two ways, both pure scheduling of the same copies:
# - ``store_exact_cache`` clones the cache it is given, so a checkpoint that is already a
#   private snapshot (the runner's deferred captures) was copied twice and both copies were
#   live until the call returned. While ``_OWNED_SNAPSHOT`` is set the snapshot itself is
#   handed over;
# - the copies of all layers were evaluated at once, so a restore held the source, the
#   copy and the capacity-padded copy of every layer together (3x the cache at its peak).
#   They are evaluated a few layers at a time, so only a few layers' intermediates exist.
_OWNED_SNAPSHOT: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "yunshu_apc_owned_snapshot", default=False
)
CLONE_EVAL_BYTES = 512 << 20


def _restore_kv(c, min_capacity_tokens, targets):
    """Copy a stored K/V checkpoint into a fresh cache with room for ``min_capacity_tokens``.

    Upstream copies the rows into a fresh buffer and then pads that copy (concatenate with
    zeros) into a second one; when the request needs more capacity than the checkpoint
    holds (always, for a follow-up turn) the prefix is written twice. Here the rows are
    read once and written once into the padded buffer: same values, same zero padding,
    same capacity (rounded up to ``step`` from the larger of rows and the request).
    """
    import mlx.core as mx
    from mlx_vlm.models.cache import KVCache

    rows = int(c.offset)
    out = KVCache()
    capacity = max(rows, int(min_capacity_tokens))
    if out.step > 0:
        capacity = ((capacity + out.step - 1) // out.step) * out.step
    pad = capacity - rows

    def build(a):
        part = a[..., :rows, :]
        if pad <= 0:
            return mx.contiguous(mx.array(part, dtype=part.dtype))
        zeros = mx.zeros((*part.shape[:2], pad, part.shape[3]), dtype=part.dtype)
        return mx.concatenate([part, zeros], axis=2)

    out.keys, out.values = build(c.keys), build(c.values)
    out.offset = rows
    targets += [out.keys, out.values]
    return out


def _install_owned_snapshot_clone() -> None:
    current = _upstream_apc._clone_prompt_cache_for_apc
    if getattr(current, "_yunshu_owned_snapshot", False):
        return
    clone_entry = getattr(_upstream_apc, "_clone_cache_entry_for_apc", None)

    def clone(prompt_cache, *, min_capacity_tokens=None):
        if _OWNED_SNAPSHOT.get():
            return list(prompt_cache)
        if clone_entry is None:
            return current(prompt_cache, min_capacity_tokens=min_capacity_tokens)
        import mlx.core as mx
        from mlx_vlm.models.cache import KVCache

        out, targets = [], []
        for c in prompt_cache:
            if (
                min_capacity_tokens is not None
                and type(c) is KVCache
                and c.keys is not None
                and c.values is not None
                and c.keys.shape[-2] >= c.offset > 0
                and c.values.shape[-2] >= c.offset
            ):
                copied = _restore_kv(c, min_capacity_tokens, targets)
            else:
                copied = clone_entry(
                    c, min_capacity_tokens=min_capacity_tokens, eval_targets=targets
                )
            if copied is None:
                return None
            out.append(copied)
            if sum(t.nbytes for t in targets) >= CLONE_EVAL_BYTES:
                mx.eval(targets)
                targets.clear()
        if targets:
            mx.eval(targets)
        return out

    clone._yunshu_owned_snapshot = True  # type: ignore[attr-defined]
    clone._upstream = current  # type: ignore[attr-defined]
    _upstream_apc._clone_prompt_cache_for_apc = clone


_install_owned_snapshot_clone()

# Request-generation bookkeeping (_born) is trimmed past this many entries, keeping the newest.
_BORN_MAX = 20000
_BORN_KEEP_GENERATIONS = 5000
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

    # Called on the writer thread after a checkpoint file landed: ``(cache_hash, token_ids,
    # extra_hash)``. The manager uses it to drop the files that checkpoint supersedes.
    on_exact_written: Any = None

    def _write_payload(self, shard_id, block_hashes, payload) -> bool:
        ok = self._write_payload_impl(shard_id, block_hashes, payload)
        h = getattr(payload, "cache_hash", None)
        if ok and h is not None:
            tokens = tuple(int(t) for t in payload.token_ids)
            self._tok_cache()[int(h)] = (tokens, int(payload.extra_hash))
            hook = self.on_exact_written
            if hook is not None:
                try:
                    hook(int(h), tokens, int(payload.extra_hash))
                except Exception:
                    logger.warning("APC disk: supersede hook failed", exc_info=True)
        return ok

    def _tok_cache(self) -> dict:
        cache: dict = self.__dict__.setdefault("_tok_by_hash", {})
        return cache

    def exact_prefixes_of(self, tokens, extra_hash, exclude=None) -> list:
        """Hashes of the stored checkpoints whose tokens are a strict prefix of ``tokens``."""
        cache = self._tok_cache()
        with self._index_lock:
            items = list(self._exact_index.items())
        out = []
        for h, path in items:
            if h == exclude:
                continue
            got = cache.get(h)
            if got is None:
                parsed = self._open_shard_header(path)
                if parsed is None:
                    continue
                meta = parsed[1]
                try:
                    got = (
                        tuple(
                            int(x) for x in meta.get("token_ids", "").split(",") if x
                        ),
                        int(meta.get("extra_hash", "0")),
                    )
                except (TypeError, ValueError):
                    continue
                cache[h] = got
            stored, extra = got
            if extra == extra_hash and 0 < len(stored) < len(tokens):
                if tokens[: len(stored)] == stored:
                    out.append(h)
        return out

    def drop_exact(self, cache_hash) -> bool:
        """Delete one checkpoint file (index and all); False when it is not stored here or busy."""
        with self._index_lock:
            path = self._exact_index.get(cache_hash)
        if path is None:
            return False
        ok = bool(self._evict_path(path))
        if ok:
            self._tok_cache().pop(cache_hash, None)
        return ok

    def _write_payload_impl(self, shard_id, block_hashes, payload) -> bool:
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


def _single_native_arrays_row(source):
    """A private native restore row with upstream merge's metadata reset.

    Return None for non-native or multi-row state. Views use separate array
    handles, so subsequent live cache mutation cannot change the restored source.
    This avoids ArraysCache.merge's zeros allocation and whole-row assignment.
    """
    from mlx_vlm.models.cache import ArraysCache

    if type(source) is not ArraysCache or any(
        a is not None and (not a.ndim or a.shape[0] != 1) for a in source.cache
    ):
        return None
    row = ArraysCache(len(source.cache))
    if source.empty():
        import mlx.core as mx

        # Upstream initializes zero padding for its one empty row.
        row.left_padding = mx.array([0])
    else:
        row.cache = [a.view(a.dtype) if a is not None else None for a in source.cache]
    return row


MATERIALIZE_BYTES = 512 << 20
ANCHOR_MIN_GAP = 8192
_BORN_KEEP = 4096
# Retained anchors may hold at most this share of the APC byte budget, and never more than
# this many bytes, counting only memory they own (rows viewed from a newer checkpoint are free).
ANCHOR_BUDGET_FRACTION = 0.15
ANCHOR_BUDGET_MAX_BYTES = 4 << 30
# Freed anchor buffers would sit in MLX's allocator pool and count in the process footprint
# until idle; give them back at once when this much was freed.
RELEASE_FREED_BYTES = 256 << 20


def release_freed_buffers(freed_bytes: int) -> None:
    if freed_bytes >= RELEASE_FREED_BYTES:
        import mlx.core as mx

        mx.clear_cache()


def thin_chain(chain: dict, end: int) -> list:
    """Keys of ``chain`` (key -> length) to keep before a checkpoint of length ``end``.

    A checkpoint is redundant when its two kept neighbours are closer than ``anchor_gap``:
    dropping it costs a branch between them less than that gap. The newest old checkpoint is
    judged against ``end`` only once a longer one exists, so a conversation growing by small
    steps accumulates anchors a gap apart instead of dropping each turn as it is superseded.
    """
    seq: list[tuple[int, int | None]] = [(0, None)]
    for key, n in sorted(chain.items(), key=lambda kv: kv[1]):
        seq.append((n, key))
        while len(seq) >= 3 and seq[-1][0] - seq[-3][0] < anchor_gap(seq[-1][0]):
            if seq[-2][1] is None:
                break
            del seq[-2]
    seq.append((end, None))
    while len(seq) >= 3 and seq[-1][0] - seq[-3][0] < anchor_gap(seq[-1][0]):
        if seq[-2][1] is None:
            break
        del seq[-2]
    return [key for _, key in seq if key is not None]


def anchor_gap(length: int) -> int:
    """Smallest distance an earlier checkpoint must lie below the next kept one to be kept."""
    return max(ANCHOR_MIN_GAP, length // 3)


def materialize(target_lists, limit_bytes: int = MATERIALIZE_BYTES) -> None:
    """Evaluate the arrays of several checkpoints in groups of about ``limit_bytes``.

    Targets are in layer order and the checkpoints of one request share their layers, so
    position i of every list is the same layer: evaluating a few layers at a time lets the
    live buffers each copy pinned go before the next layer's copy is made. Groups are sized
    by bytes so a small cache is one evaluation (each one is a GPU synchronization).
    """
    import mlx.core as mx

    width = max((len(t) for t in target_lists), default=0)
    batch, size = [], 0
    for i in range(width):
        for t in target_lists:
            if i < len(t):
                batch.append(t[i])
                size += t[i].nbytes
        if size >= limit_bytes:
            mx.eval(batch)
            batch, size = [], 0
    if batch:
        mx.eval(batch)


SHARE_MAX_TAIL = 4096


def share_prefix_rows(pending, max_tail: int = SHARE_MAX_TAIL) -> list:
    """Make a shorter checkpoint's K/V rows views of a longer one's of the same request.

    A later checkpoint holds the earlier one's rows unchanged (K/V rows are written once, in
    order), so its first ``m`` rows ARE the earlier checkpoint's: the earlier one is stored
    as views of them and its own copy is never made. Same bits, one physical buffer. The
    longer checkpoint's buffer stays alive while a view of it does, so only checkpoints at
    most ``max_tail`` tokens apart share (that bounds what a surviving view can pin beyond
    its own size). Recurrent state differs per position and is never shared. Returns the
    views to evaluate once the longer checkpoint's arrays exist.
    """
    from mlx_vlm.models.cache import KVCache

    views: list = []
    for short in pending:
        tokens, snapshot, extra_hash, targets, generation = short[:5]
        n = len(tokens)
        donors = [
            p
            for p in pending
            if p is not short
            and p[2] == extra_hash
            and p[4] == generation
            and 0 < len(p[0]) - n <= max_tail
            and p[0][:n] == tokens
        ]
        if not donors:
            continue
        donor = max(donors, key=lambda p: len(p[0]))
        for mine, theirs in zip(snapshot, donor[1], strict=False):
            if type(mine) is not KVCache or type(theirs) is not KVCache:
                continue
            if mine.keys is None or theirs.keys is None or mine.values is None:
                continue
            rows = mine.keys.shape[-2]
            if (
                mine.values.shape[-2] != rows
                or theirs.keys.shape[-2] < rows
                or theirs.values.shape[-2] < rows
                or mine.keys.shape[:-2] != theirs.keys.shape[:-2]
                or mine.keys.dtype != theirs.keys.dtype
            ):
                continue
            own = {id(mine.keys), id(mine.values)}
            targets[:] = [t for t in targets if id(t) not in own]
            mine.keys = theirs.keys[..., :rows, :]
            mine.values = theirs.values[..., :rows, :]
            views += [mine.keys, mine.values]
    return views


class _Coordinator(APCCoordinator):
    """Checkpoint positions: the prompt end, one interval boundary, the end of the system turn."""

    def set_request(self, token_ids, policy):
        if policy is None:
            return
        # Bounded by live queued/prefilling rows, not a lossy LRU of active
        # descriptors. Identical tokens may have different write/retention intent.
        policies = self.__dict__.setdefault("_requests", {})
        policies.setdefault(tuple(token_ids), collections.deque()).append(policy)

    def request(self, token_ids):
        pending = self.__dict__.get("_requests", {}).get(tuple(token_ids))
        return pending[0] if pending else None

    def release_request(self, token_ids, policy):
        policies = self.__dict__.get("_requests", {})
        tokens = tuple(token_ids)
        pending = policies.get(tokens)
        if pending is None:
            return
        for i, queued in enumerate(pending):
            if queued is policy:
                del pending[i]
                break
        if not pending:
            policies.pop(tokens, None)

    def lookup(self, token_ids, **kwargs):
        policy = self.request(token_ids)
        if not policy or not self.is_checkpoint:
            return self._lookup_canonical(token_ids, **kwargs)
        points = self.checkpoint_lengths(token_ids, set(), begin=False)
        candidates = policy.get("lookup_points", [n for n, _ in policy["points"]])
        maximum = max(candidates, default=0)
        extra_hash = kwargs["extra_hash"]
        while maximum > 0:
            cache, n = self.manager.lookup_exact_cache(
                token_ids,
                extra_hash=extra_hash,
                max_prefix_tokens=maximum,
                min_prefix_tokens=kwargs.get("safe_lookup_min", 0),
                refresh_retention=False,
            )
            if cache is None or not n:
                return None
            key = _sequence_hash(
                tuple(token_ids[:n]), extra_hash, self.manager.block_size
            )
            # Restoring at a noncanonical point, or from a different earlier
            # split plan, changes the hybrid recurrence's numerical spans.
            signature = tuple(p for p in points if p < n and p % 2048)
            source = self.manager._span_plans.get(key)
            if (
                (n in candidates)
                and source == signature
                and kwargs["suffix_is_text_only"](n)
            ):
                self.manager.touch_boundary(token_ids[:n], extra_hash)
                return {
                    "matched_blocks": [],
                    "warm_cache": cache,
                    "prefix_len": n,
                    "extra_hash": extra_hash,
                    "full_input_ids": list(token_ids),
                }
            maximum = n - 1
        return None

    def _publish_checkpoint_policy(self, tokens, extra_hash, policy, signature):
        if policy is None:
            return
        duration = next(
            ttl for n, ttl in policy.get("writes", policy["points"]) if n == len(tokens)
        )
        self.manager.protect_boundary(tokens, extra_hash, duration)
        key = _sequence_hash(tokens, extra_hash, self.manager.block_size)
        self.manager._span_plans[key] = signature
        while len(self.manager._span_plans) > _BORN_MAX:
            self.manager._span_plans.pop(next(iter(self.manager._span_plans)))
        policy["written"] = max(policy.get("written", 0), len(tokens))

    # Enabled only around a runner generator step. Round-driver and external APC
    # callers retain synchronous admission until they supply the same flush boundary.
    defer_checkpoint_stores = False

    def _lookup_canonical(self, token_ids, **kwargs):
        stride = getattr(self.manager, "prefill_stride", 0)
        # Media processors may expand / reshape positions; retain their existing
        # boundary policy. Text prefixes use the same grid in cold and warm runs.
        if not stride or kwargs["prefix_has_media"](len(token_ids)):
            return super().lookup(token_ids, **kwargs)
        policy = (
            id(self.manager),
            stride,
            len(token_ids) - self.manager.exact_cache_guard_tokens,
            self.manager.head_boundary(token_ids),
            frozenset(self.manager.semantic_boundaries(token_ids)),
        )
        context = _CANONICAL_LOOKUP.set(policy)
        try:
            return super().lookup(token_ids, **kwargs)
        finally:
            _CANONICAL_LOOKUP.reset(context)

    def merge_rows(self, picks, prefix_lens, *, kv_quant_config=None):
        """Keep the reserved capacity of a private, single-row native restore."""
        from mlx_vlm.apc_adapters import merge_cache_entries
        from mlx_vlm.models.cache import ArraysCache, BatchKVCache, KVCache

        hit = picks[0] if len(picks) == 1 else None
        rows = hit.get("warm_cache") if hit is not None else None
        prefix = int(prefix_lens[0]) if len(prefix_lens) == 1 else 0
        if (
            self.enabled
            and self.is_checkpoint
            and kv_quant_config is None
            and prefix > 0
            and rows
            and all(
                type(c) is ArraysCache
                or (
                    type(c) is KVCache
                    and c.offset == prefix
                    and c.keys is not None
                    and c.values is not None
                    and c.keys.shape[0] == 1
                )
                for c in rows
            )
        ):
            import mlx.core as mx

            lengths = self.manager.memory_plan.lengths
            # The measured suffix-prefill win does not extend to one-token
            # revisits. Keep their upstream merge; unknown plans do likewise.
            arrays_view = len(lengths) == 1 and lengths[0] - prefix >= 64
            merged = []
            for c in rows:
                if type(c) is KVCache:
                    batch = BatchKVCache([0])
                    # New array handles share the already-detached restore's
                    # buffers. MLX copy-on-write protects any retained source
                    # handles; no physical row copy or capacity truncation.
                    batch.keys = c.keys.view(c.keys.dtype)
                    batch.values = c.values.view(c.values.dtype)
                    batch._idx = prefix
                    batch.offset += prefix
                    merged.append(batch)
                else:
                    row = _single_native_arrays_row(c) if arrays_view else None
                    merged.append(
                        row if row is not None else merge_cache_entries([c], [prefix])
                    )
            if all(c is not None for c in merged):
                mx.eval([c.state for c in merged])
                with self.manager.lock:
                    self.manager.stats.restored_tokens += prefix
                return merged, prefix
        return super().merge_rows(picks, prefix_lens, kv_quant_config=kv_quant_config)

    def store_checkpoint(
        self, token_ids, prompt_cache, *, extra_hash=0, batch_idx=None
    ) -> bool:
        tokens = tuple(token_ids)
        full_ids = getattr(self, "_current_ids", tokens)
        policy = self.request(full_ids)
        signature = ()
        if policy is not None:
            if len(tokens) not in {
                n for n, _ in policy.get("writes", policy["points"])
            }:
                return False
            points = self.checkpoint_lengths(full_ids, set(), begin=False)
            signature = tuple(p for p in points if p < len(tokens) and p % 2048)
        if (
            policy is None
            and getattr(self, "_split_lengths", False)
            and len(token_ids) not in self._store_lengths
        ):
            return False  # deterministic forward boundary, not an extra snapshot
        from mlx_vlm.apc import _cache_nbytes
        from mlx_vlm.apc_adapters import clone_cache_entry
        from mlx_vlm.models.cache import ArraysCache, BatchKVCache, KVCache

        # Multi-row extraction and custom contracts stay on the upstream path.
        # Native single-row caches (including the spec lane's BatchKVCache) only
        # build lazy, detached arrays and copy Python metadata in these adapters.
        can_defer = (
            self.defer_checkpoint_stores
            and self.enabled
            and self.is_checkpoint
            and batch_idx in (None, 0)
            and bool(prompt_cache)
            and all(
                type(c) in (ArraysCache, KVCache)
                or (type(c) is BatchKVCache and c.is_single_row())
                for c in prompt_cache
            )
        )
        size = _cache_nbytes(prompt_cache) if can_defer else 0
        pending_bytes = getattr(self, "_deferred_bytes", 0)
        # Captures are outside the manager's resident accounting until flush.
        # Keep their combined footprint inside the same APC budget; a full
        # tier falls back to upstream eviction / reserve checks before copying.
        resident = self.manager.resident_bytes() if can_defer else 0
        if (
            not can_defer
            or resident + pending_bytes + size > self.manager.memory_max_bytes
        ):
            ok = bool(
                super().store_checkpoint(
                    token_ids, prompt_cache, extra_hash=extra_hash, batch_idx=batch_idx
                )
            )
            if ok:
                self._publish_checkpoint_policy(tokens, extra_hash, policy, signature)
            return ok
        targets: list[Any] = []
        snapshot = [
            clone_cache_entry(c, min_capacity_tokens=None, eval_targets=targets)
            for c in prompt_cache
        ]
        if any(c is None for c in snapshot):
            ok = bool(
                super().store_checkpoint(
                    token_ids, prompt_cache, extra_hash=extra_hash, batch_idx=batch_idx
                )
            )
            if ok:
                self._publish_checkpoint_policy(tokens, extra_hash, policy, signature)
            return ok
        pending = self.__dict__.setdefault("_deferred_checkpoints", [])
        pending.append(
            (
                tuple(token_ids),
                snapshot,
                int(extra_hash),
                targets,
                getattr(
                    self,
                    "_request_generation",
                    getattr(self.manager, "_generation", None),
                ),
                policy,
                signature,
            )
        )
        self._deferred_bytes = pending_bytes + size
        self.deferred_checkpoint_count = (
            getattr(self, "deferred_checkpoint_count", 0) + 1
        )
        # "stored" means captured here: the runner guarantees a flush after its
        # first emitted token. A cancelled prefill explicitly discards its captures.
        return True

    def flush_deferred_checkpoints(self) -> None:
        pending = self.__dict__.pop("_deferred_checkpoints", [])
        copy_bytes = getattr(self, "_deferred_bytes", 0)
        self._deferred_bytes = 0
        if not pending:
            return
        # The prefill's freed buffers sit in MLX's allocator pool (several GiB after a long
        # prefill) and the copies below are allocated on top of live memory: give the pool
        # back first so the process footprint at this point is active memory, not active
        # plus pool.
        release_freed_buffers(copy_bytes)
        # The copies below are lazy until evaluated, so each still pins the live
        # cache's buffers as they were at capture. Free what these checkpoints supersede
        # first (the same drops ``store_exact_cache`` makes), then copy a few arrays at a
        # time across all checkpoints so the pinned buffers go as the copies appear.
        # Same operations and bits as evaluating everything at once; only the peak differs.
        release = getattr(self.manager, "release_superseded", None)
        for tokens, _, extra_hash, _, generation, _, _ in pending:
            if release is not None:
                release(tokens, extra_hash, _generation=generation)
        views = share_prefix_rows(pending)
        longest = max(pending, key=lambda p: len(p[0]))
        share_lazy = getattr(self.manager, "share_anchor_rows_lazy", None)
        if share_lazy is not None:
            # retained anchors give up their own rows before the new copy is made
            views = views + share_lazy(longest[0], longest[1], longest[2])
        materialize([entry[3] for entry in pending])
        if views:
            import mlx.core as mx

            mx.eval(views)
        for (
            tokens,
            snapshot,
            extra_hash,
            _targets,
            generation,
            policy,
            signature,
        ) in pending:
            kwargs = {"extra_hash": extra_hash, "_owned": True}
            if generation is not None:
                kwargs["_generation"] = generation
            if self.manager.store_exact_cache(tokens, snapshot, **kwargs):
                self._publish_checkpoint_policy(tokens, extra_hash, policy, signature)
        finish = getattr(self.manager, "finish_anchor_sharing", None)
        if finish is not None:
            finish()

    def discard_deferred_checkpoints(self) -> None:
        self.__dict__.pop("_deferred_checkpoints", None)
        self._deferred_bytes = 0

    def checkpoint_lengths(self, token_ids, media_token_ids, *, begin=True):
        if begin:
            self._current_ids = tuple(token_ids)
        final = self.checkpoint_len(token_ids, media_token_ids)
        if final <= 0:
            return []
        mgr = self.manager
        stride = getattr(mgr, "prefill_stride", 0)
        if media_token_ids.intersection(token_ids):
            stride = 0
        policy = self.request(token_ids)
        if policy:
            from mlx_vlm.apc import adjust_prefix_to_text_suffix_boundary

            lengths = []
            for n, _ in policy["points"]:
                safe = adjust_prefix_to_text_suffix_boundary(
                    token_ids, n, media_token_ids, max_prefix_tokens=final
                )
                if safe != n:
                    raise ValueError(
                        "cache breakpoint cuts a media span or final guard"
                    )
                if n >= mgr.exact_cache_min_tokens:
                    lengths.append(n)
            if begin:
                mgr.begin_request()
                self._request_generation = mgr._generation
                self._split_lengths = False
            return sorted(set(lengths))
        lengths = {final}
        interval = mgr.checkpoint_interval_tokens
        if interval > 0 and mgr.keep_interval_checkpoint:
            from mlx_vlm.apc import adjust_prefix_to_text_suffix_boundary

            block = mgr.block_size
            interval = ((interval + block - 1) // block) * block
            last = ((final - 1) // interval) * interval
            if stride:
                last = (last // stride) * stride
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
        if begin:
            mgr.begin_request()
        self._request_generation = mgr._generation
        self._store_lengths = frozenset(lengths)
        self._split_lengths = bool(stride)
        if stride:
            lengths.update(range(stride, final, stride))
            lengths.update(n for n in mgr.semantic_boundaries(token_ids) if n < final)
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
        prefill_stride: int = 0,
        prefill_boundary_tokens: tuple[int, ...] = (),
        prefill_assistant_header: tuple[int, int] | None = None,
        prefill_message_end: int | None = None,
        **kwargs: Any,
    ):
        overrides = dict(kwargs.pop("overrides", None) or {})
        overrides.setdefault("checkpoint_entries", max_entries)
        super().__init__(*args, overrides=overrides, **kwargs)
        # (<|im_start|>, user) token ids: the end of the system turn is the head boundary.
        self.head_marker = head_marker
        self.keep_interval_checkpoint = keep_interval_checkpoint
        self.prefill_stride = int(prefill_stride)
        self.prefill_boundary_tokens = frozenset(prefill_boundary_tokens)
        self.prefill_assistant_header = prefill_assistant_header
        self.prefill_message_end = prefill_message_end
        self.lookups: collections.deque[Lookup] = collections.deque(maxlen=64)
        self._generation = 0
        # Explicit endpoints are protected against superseding until their
        # inactivity TTL expires, always within the existing byte/entry budgets.
        self._retention: dict[int, tuple[float, int]] = {}
        self._span_plans: dict[int, tuple[int, ...]] = {}
        self._expired_boundaries: set[int] = set()
        self._born: dict[int, int] = {}
        self._head_keys: set[int] = set()
        self._last_hit: tuple[tuple, int] | None = None
        self._anchors: dict[int, int] = {}  # retained superseded checkpoint -> length
        # entry key -> (root buffer owner key, K/V bytes it views instead of owning), and the
        # K/V bytes of each root buffer: lets resident_bytes count a shared buffer once.
        self._kv_share: dict[int, tuple[int, int]] = {}
        self._roots: dict[int, int] = {}
        self._head_lengths: set[int] = set()
        self._plock = threading.Lock()
        self.warm = None
        self._promoted: tuple[int, float] | None = None
        self.tier_hits: collections.Counter = collections.Counter()
        self.disk_superseded = 0
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
        if isinstance(self.disk, SpillDiskStore):
            self.disk.on_exact_written = self._disk_superseded_by

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

    def _disk_superseded_by(self, cache_hash, tokens, extra_hash) -> None:
        """A checkpoint reached the SSD tiers: the checkpoints of earlier requests that are
        prefixes of it (the conversation grew past them) are garbage there too, exactly as in
        RAM; without this a tier fills with stale copies of every session and the cache the next
        request needs is the one LRU pushes out first. Heads are kept, and so is anything this
        process did not store or restore itself (it may serve another session)."""
        with self._plock:
            newest = self._born.get(cache_hash)
            heads = set(self._head_keys) | set(self._retention) | set(self._anchors)
        disk = self.disk
        if newest is None or disk is None:
            return
        dropped = 0
        for h in disk.exact_prefixes_of(tokens, extra_hash, exclude=cache_hash):
            if h in heads:
                continue
            with self._plock:
                born = self._born.get(h)
            if born is None or born >= newest:
                continue
            if disk.drop_exact(h):
                dropped += 1
                with self._plock:
                    self._born.pop(h, None)
        if dropped:
            self.disk_superseded += dropped
            logger.debug("APC: dropped %d superseded SSD checkpoint(s)", dropped)

    def _note_disk_hit(self, tokens, n, extra_hash) -> None:
        """A checkpoint came back from an SSD tier: it belongs to this request's generation (its
        next, longer checkpoint supersedes it), and it is a head when it ends the system turn."""
        key = _sequence_hash(
            tuple(int(t) for t in tokens[:n]), extra_hash, self.block_size
        )
        with self._plock:
            self._born.setdefault(key, self._generation)
            if self.head_marker is not None and self.head_boundary(tokens) == n:
                self._head_keys.add(key)

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
        if self.disk is not None:
            # a longer prefix on SSD wins the lookup anyway: decoding this one would be wasted
            longer = self.disk.find_exact_prefix(
                tokens,
                extra_hash=extra_hash,
                max_prefix_tokens=max_prefix_tokens,
                min_prefix_tokens=n,
                block_size=self.block_size,
            )
            if longer is not None:
                return
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
        if n >= 2 and token_ids[0] == a and token_ids[1] == b:
            return 0
        for i in range(1, n - 1):
            if token_ids[i] == a and token_ids[i + 1] == b:
                return i
        return 0

    def checkpoint_ready(self, token_ids, extra_hash):
        """Publication receipt without cloning/evaluating a checkpoint."""
        if extra_hash is None:
            return False
        self._expire_boundaries()
        tokens = tuple(token_ids)
        key = _sequence_hash(tokens, extra_hash, self.block_size)
        with self.lock:
            entry = self._exact_cache.get(key)
            return bool(
                entry is not None
                and entry.token_ids == tokens
                and entry.extra_hash == extra_hash
            )

    def protect_boundary(self, token_ids, extra_hash, duration):
        key = _sequence_hash(tuple(token_ids), extra_hash, self.block_size)
        self._retention[key] = (time.monotonic() + duration, duration)
        self._expired_boundaries.discard(key)
        # Only resident/bookkept entries need intent. Bound descriptor memory.
        if len(self._retention) > _BORN_MAX:
            self._retention = {
                k: v for k, v in self._retention.items() if v[0] > time.monotonic()
            }

    def touch_boundary(self, token_ids, extra_hash):
        key = _sequence_hash(tuple(token_ids), extra_hash, self.block_size)
        if key in self._retention:
            duration = self._retention[key][1]
            self._retention[key] = (time.monotonic() + duration, duration)

    def _expire_boundaries(self):
        now = time.monotonic()
        expired = [k for k, (deadline, _) in self._retention.items() if deadline <= now]
        for key in expired:
            self._retention.pop(key, None)
            self._expired_boundaries.add(key)
            self._span_plans.pop(key, None)
            with self.lock:
                self._exact_cache.pop(key, None)
            if self.warm is not None:
                self.warm.take(key)
            if isinstance(self.disk, SpillDiskStore):
                self.disk.drop_exact(key)

    def begin_request(self) -> None:
        with self._plock:
            self._generation += 1
            if (
                len(self._born) > _BORN_MAX
            ):  # bookkeeping only: forget the oldest generations
                cut = self._generation - _BORN_KEEP_GENERATIONS
                self._born = {k: g for k, g in self._born.items() if g >= cut}

    def note_head(self, head_tokens, extra_hash: int = 0) -> None:
        with self._plock:
            self._head_lengths.add(len(head_tokens))

    def store_exact_cache(
        self,
        token_ids,
        prompt_cache,
        *,
        extra_hash=0,
        _generation=None,
        _owned=False,
    ) -> bool:
        n = len(token_ids)
        with self._plock:
            # A deferred checkpoint belongs to the request that captured it,
            # even when another group began prefill before publication.
            gen = self._generation if _generation is None else _generation
            is_head = n in self._head_lengths
        self._supersede(token_ids, extra_hash, gen)
        key = _sequence_hash(
            tuple(int(t) for t in token_ids), extra_hash, self.block_size
        )
        # recorded first: a checkpoint too big for RAM is written to the SSD inside the call
        # below, and the write hook needs its generation to tell what it supersedes
        with self._plock:
            had = key in self._born
            self._born[key] = gen
            if is_head:
                self._head_keys.add(key)
        token = _OWNED_SNAPSHOT.set(bool(_owned))
        try:
            ok = super().store_exact_cache(
                token_ids, prompt_cache, extra_hash=extra_hash
            )
        finally:
            _OWNED_SNAPSHOT.reset(token)
        if not ok and not had:
            with self._plock:
                self._born.pop(key, None)
                if is_head:
                    self._head_keys.discard(key)
        return ok

    def release_superseded(
        self, token_ids, extra_hash: int = 0, *, _generation=None
    ) -> None:
        """Drop the RAM checkpoints ``token_ids`` supersedes before its copy exists."""
        with self._plock:
            gen = self._generation if _generation is None else _generation
        self._supersede(token_ids, extra_hash, gen)

    def _supersede(self, token_ids, extra_hash: int, gen: int) -> None:
        """Thin the checkpoints of earlier requests that are prefixes of this one.

        A longer checkpoint makes the earlier ones redundant for a linear follow-up, but an
        agent that retries, edits an earlier turn or forks a sub-agent branches somewhere
        inside the old conversation, and only an earlier checkpoint can serve that. The chain
        of earlier checkpoints is therefore kept at geometric spacing (see ``anchor_gap``)
        instead of dropped: a branch loses at most about a third of the shared prefix and the
        retained set stays a handful of checkpoints. Anchors sit at the oldest end of the LRU,
        so the byte and entry budgets evict them before anything newer.
        """
        tokens = tuple(int(t) for t in token_ids)
        with self._plock:
            protected = set(self._head_keys) | set(self._retention)
            anchors = dict(self._anchors)
        chain: dict[int, int] = {}  # candidate key -> length
        dropped_by_filter: dict[int, int] = {}
        with self.lock:
            for key, entry in self._exact_cache.items():
                stored = entry.token_ids
                if (
                    entry.extra_hash != extra_hash
                    or len(stored) >= len(tokens)
                    or key in protected
                    or self._born.get(key, gen) >= gen
                    or tokens[: len(stored)] != stored
                ):
                    continue
                chain[key] = len(stored)
        if self.prefill_stride:
            # Hybrid state is only exact at a position every request runs a forward boundary
            # at (a stride multiple or a semantic boundary): any other anchor could not be
            # restored without changing the recurrence's numerical spans.
            semantic = self.semantic_boundaries(tokens)
            stride = self.prefill_stride
            dropped_by_filter = {
                k: n for k, n in chain.items() if not (n % stride == 0 or n in semantic)
            }
            for k in dropped_by_filter:
                del chain[k]
        for key, n in anchors.items():
            if key not in chain and key not in protected and n < len(tokens):
                if _sequence_hash(tokens[:n], extra_hash, self.block_size) == key:
                    chain[key] = n
        keep = set(thin_chain(chain, len(tokens)))
        chain.update(dropped_by_filter)
        dropped = 0
        with self.lock:
            for key in chain:
                if key in keep:
                    continue
                if self._exact_cache.pop(key, None) is not None:
                    dropped += 1
                # _born stays: a copy of this checkpoint may still be on the SSD, and the
                # disk supersede needs its generation to know it is an earlier request's
        with self._plock:
            for key in chain:
                if key in keep:
                    self._anchors[key] = chain[key]
                else:
                    self._anchors.pop(key, None)
            if len(self._anchors) > _BORN_KEEP:
                self._anchors = dict(list(self._anchors.items())[-_BORN_KEEP:])
            held = set(self._head_keys) | set(self._retention) | set(self._anchors)
        if self.warm is not None:
            dropped += self.warm.supersede(tokens, extra_hash, gen, held)
        if dropped:
            logger.debug("APC: superseded %d earlier checkpoint(s)", dropped)

    def _resident_bytes_locked(self) -> int:
        """Resident bytes with a K/V buffer shared by several checkpoints counted once."""
        logical = super()._resident_bytes_locked()
        saved, pinned = self._share_accounting()
        return max(0, logical - saved + pinned)

    def _share_accounting(self) -> tuple[int, int]:
        """(bytes viewed rather than owned, bytes of root buffers held only through views)."""
        cache = self._exact_cache
        for key in [k for k in self._kv_share if k not in cache]:
            del self._kv_share[key]
        saved = sum(v for _, v in self._kv_share.values())
        used = {root for root, _ in self._kv_share.values()}
        for root in [r for r in self._roots if r not in used]:
            del self._roots[root]
        # a root whose owner is still a normal entry is already in the logical total
        pinned = sum(
            self._roots.get(root, 0)
            for root in used
            if root not in cache or root in self._kv_share
        )
        return saved, pinned

    def anchor_bytes(self) -> int:
        """Bytes the retained anchors own: their state plus K/V rows no newer checkpoint holds."""
        from mlx_vlm.apc import _cache_nbytes

        with self.lock:
            _, pinned = self._share_accounting()
            own = 0
            for key in self._anchors:
                entry = self._exact_cache.get(key)
                if entry is not None:
                    own += _cache_nbytes(entry.prompt_cache)
                    own -= self._kv_share.get(key, (0, 0))[1]
            return max(0, own + pinned)

    def anchor_budget_bytes(self) -> int:
        return min(
            ANCHOR_BUDGET_MAX_BYTES, int(ANCHOR_BUDGET_FRACTION * self.memory_max_bytes)
        )

    def _drop_anchor(self) -> bool:
        """Evict the shortest retained anchor (the least useful branch point)."""
        with self.lock:
            live = [
                (n, k)
                for k, n in self._anchors.items()
                if k in self._exact_cache and k not in self._head_keys
            ]
            if not live:
                return False
            _, key = min(live)
            self._exact_cache.pop(key, None)
        with self._plock:
            self._anchors.pop(key, None)
        return True

    def enforce_anchor_budget(self) -> int:
        dropped = 0
        budget = self.anchor_budget_bytes()
        while self.anchor_bytes() > budget and self._drop_anchor():
            dropped += 1
        return dropped

    def _memory_headroom(self) -> int:
        memo = getattr(self, "_headroom_memo", None)
        return super()._memory_headroom() if memo is None else memo

    def _make_room(self, allocation_bytes: int = 0, *, retain_bytes: int = 0) -> bool:
        """Evict anchors before anything else, so they never raise the peak of a big request.

        The free-memory reading (a system query) is taken once per call and shared with the
        upstream pass: dropping anchors moves resident and headroom by the same bytes, so the
        eviction target is unchanged.
        """
        self._headroom_memo = None
        try:
            if self._anchors:
                self._headroom_memo = self._memory_headroom()
                required = self.memory_reserve_bytes + (
                    self._prefill_reserve_bytes + allocation_bytes
                    if retain_bytes
                    else max(self._prefill_reserve_bytes, allocation_bytes)
                )
                dropped = 0
                while True:
                    with self.lock:
                        resident = self._resident_bytes_locked()
                        target = max(
                            0,
                            min(
                                self.memory_max_bytes - retain_bytes,
                                resident + self._headroom_memo - required,
                            ),
                        )
                    if resident <= target or not self._drop_anchor():
                        break
                    dropped += 1
                if dropped:
                    release_freed_buffers(1 << 40)
            return super()._make_room(allocation_bytes, retain_bytes=retain_bytes)
        finally:
            self._headroom_memo = None

    def share_anchor_rows(self, donor_tokens, extra_hash: int = 0) -> int:
        """Share rows with the checkpoint just stored, then bring anchors under their budget."""
        try:
            shared, views, freed = self._share_anchor_rows(donor_tokens, extra_hash)
            if views:
                import mlx.core as mx

                mx.eval(views)
                release_freed_buffers(freed)
            return shared
        finally:
            self.enforce_anchor_budget()

    def share_anchor_rows_lazy(self, donor_tokens, donor_cache, extra_hash: int = 0):
        """Re-point anchors at a checkpoint that is still being copied (not yet stored).

        Done before the copy is evaluated, an anchor's own buffer is released at once, the
        way ``release_superseded`` releases what a checkpoint replaces, so the old and the new
        copy never coexist. Returns the views to evaluate with the copy; call
        ``finish_anchor_sharing`` afterwards.
        """
        _, views, freed = self._share_anchor_rows(
            donor_tokens, extra_hash, donor_cache=donor_cache
        )
        self._lazy_freed = getattr(self, "_lazy_freed", 0) + freed
        return views

    def finish_anchor_sharing(self) -> None:
        freed, self._lazy_freed = getattr(self, "_lazy_freed", 0), 0
        release_freed_buffers(freed)
        self.enforce_anchor_budget()

    def _share_anchor_rows(self, donor_tokens, extra_hash: int = 0, donor_cache=None):
        """Re-point the K/V rows of retained anchors at the checkpoint just stored.

        A request that restored a checkpoint at ``h`` holds, for every position up to ``h``,
        the very rows of that restore, so a retained anchor at or below ``h`` that is a prefix
        of the new checkpoint stores the same bits again. Making it a view of the new
        checkpoint's buffer frees its own copy: a whole chain of anchors costs one buffer.
        Anchors above the restore point were computed by an earlier request and keep their own.
        """
        from mlx_vlm.models.cache import KVCache

        donor_tokens = tuple(int(t) for t in donor_tokens)
        hit = self._last_hit
        if hit is None or hit[1] != extra_hash or donor_tokens[: len(hit[0])] != hit[0]:
            return 0, [], 0
        limit = len(hit[0])
        views: list = []
        shared = 0
        freed = 0
        with self.lock:
            donor_key = _sequence_hash(donor_tokens, extra_hash, self.block_size)
            if donor_cache is not None:
                donor = SimpleNamespace(
                    prompt_cache=donor_cache, token_ids=donor_tokens
                )
            else:
                donor = self._exact_cache.get(donor_key)
            if donor is None or donor.token_ids != donor_tokens:
                return 0, [], 0
            root = self._kv_share.get(donor_key, (donor_key, 0))[0]
            if root == donor_key:
                self._roots[root] = sum(
                    c.keys.nbytes + c.values.nbytes
                    for c in donor.prompt_cache
                    if type(c) is KVCache
                    and c.keys is not None
                    and c.values is not None
                )
            with self._plock:
                keys = list(self._anchors)
            for key in keys:
                entry = self._exact_cache.get(key)
                n = len(entry.token_ids) if entry is not None else 0
                if (
                    entry is None
                    or entry is donor
                    or n > limit
                    or n >= len(donor_tokens)
                    or entry.extra_hash != extra_hash
                    or donor_tokens[:n] != entry.token_ids
                ):
                    continue
                viewed = 0
                for mine, theirs in zip(
                    entry.prompt_cache, donor.prompt_cache, strict=False
                ):
                    if type(mine) is not KVCache or type(theirs) is not KVCache:
                        continue
                    if mine.keys is None or theirs.keys is None or mine.values is None:
                        continue
                    rows = mine.keys.shape[-2]
                    if (
                        mine.values.shape[-2] != rows
                        or theirs.keys.shape[-2] < rows
                        or theirs.values.shape[-2] < rows
                        or mine.keys.shape[:-2] != theirs.keys.shape[:-2]
                        or mine.keys.dtype != theirs.keys.dtype
                    ):
                        continue
                    mine.keys = theirs.keys[..., :rows, :]
                    mine.values = theirs.values[..., :rows, :]
                    views += [mine.keys, mine.values]
                    viewed += mine.keys.nbytes + mine.values.nbytes
                if viewed:
                    self._kv_share[key] = (root, viewed)
                    freed += viewed
                shared += 1
        return shared, views, freed

    # ── provenance ─────────────────────────────────────────────────────
    def semantic_boundaries(self, token_ids):
        boundaries = set()
        header = self.prefill_assistant_header
        assistant = header is None
        previous = None
        for i, token in enumerate(token_ids):
            if header is not None:
                if token == header[0] or token == self.prefill_message_end:
                    assistant = False
                elif previous == header[0]:
                    assistant = token == header[1]
            if assistant and token in self.prefill_boundary_tokens:
                boundaries.add(i + 1)
            previous = token
        return boundaries

    def _canonical_bounds(self, tokens, extra_hash, maximum, minimum, policy):
        _, stride, final, head, semantic = policy

        def allowed(n):
            return n > 0 and (n in (final, head) or n in semantic or n % stride == 0)

        def previous(n):
            return max(
                (n - 1) // stride * stride,
                head if head < n else 0,
                final if final < n else 0,
                max((b for b in semantic if b < n), default=0),
            )

        best = 0
        with self.lock:
            for entry in self._exact_cache.values():
                n = len(entry.token_ids)
                if (
                    minimum < n <= maximum
                    and n > best
                    and allowed(n)
                    and entry.extra_hash == extra_hash
                    and tokens[:n] == entry.token_ids
                ):
                    best = n
        if self.warm is not None:
            limit = maximum
            while limit > max(minimum, best):
                hit = self.warm.find(tokens, extra_hash, limit, max(minimum, best))
                if hit is None:
                    break
                n = hit[1]
                if allowed(n):
                    best = max(best, n)
                    break
                limit = previous(n)
        if self.disk is not None:
            limit = maximum
            while limit > max(minimum, best):
                hit = self.disk.find_exact_prefix(
                    tokens,
                    extra_hash=extra_hash,
                    max_prefix_tokens=limit,
                    min_prefix_tokens=max(minimum, best),
                    block_size=self.block_size,
                )
                if hit is None:
                    break
                n = hit[1]
                if allowed(n):
                    best = max(best, n)
                    break
                limit = previous(n)
        return best

    def lookup_exact_cache(self, token_ids, *args, **kwargs):
        refresh_retention = kwargs.pop("refresh_retention", True)
        self._expire_boundaries()
        policy = _CANONICAL_LOOKUP.get()
        if policy is not None and policy[0] == id(self):
            names = ("extra_hash", "max_prefix_tokens", "min_prefix_tokens")
            defaults = (0, None, 0)
            vals = [
                kwargs.get(k, args[i] if i < len(args) else d)
                for i, (k, d) in enumerate(zip(names, defaults, strict=True))
            ]
            maximum = len(token_ids) - 1
            if vals[1] is not None and vals[1] > 0:
                maximum = min(maximum, int(vals[1]))
            n = self._canonical_bounds(
                tuple(token_ids), int(vals[0]), maximum, int(vals[2]), policy
            )
            if not n:
                self.tier_hits["none"] += 1
                self.lookups.append(
                    Lookup(len(token_ids), 0, "none", 0.0, time.perf_counter(), None)
                )
                return None, 0
            # Require this exact boundary, including if eviction races selection.
            # Falling back to a smaller arbitrary guard would reintroduce drift.
            args = ()
            kwargs.update(
                extra_hash=int(vals[0]),
                max_prefix_tokens=n,
                min_prefix_tokens=max(int(vals[2]), n - 1),
            )
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
        extra = self._extra_of(args, kwargs)
        # A background spill may finish after drop_exact saw a busy writer.
        # Tombstones veto resurrection without touching a request's live state.
        while (
            n
            and _sequence_hash(tuple(token_ids[:n]), extra, self.block_size)
            in self._expired_boundaries
        ):
            maximum = n - 1
            retry = dict(kwargs, max_prefix_tokens=maximum)
            positional = list(args)
            if len(positional) > 1:
                positional[1] = maximum
                retry.pop("max_prefix_tokens")
            cache, n = super().lookup_exact_cache(token_ids, *positional, **retry)
        self._last_hit = (tuple(int(t) for t in token_ids[:n]), extra) if n else None
        if n and refresh_retention:
            self.touch_boundary(token_ids[:n], extra)
        ms = (time.perf_counter() - t0) * 1000.0
        tier = ("ssd" if self.stats.disk_hits > before else "ram") if n else "none"
        if n and self._promoted is not None and self._promoted[0] == n:
            tier = "warm"
        self._promoted = None
        if tier == "ssd" and n:
            self._note_disk_hit(
                tuple(int(t) for t in token_ids), int(n), self._extra_of(args, kwargs)
            )
        self.tier_hits[tier] += 1
        device = getattr(self.disk, "last_device", None) if tier == "ssd" else None
        self.lookups.append(
            Lookup(
                len(token_ids), int(n), tier, round(ms, 1), time.perf_counter(), device
            )
        )
        return cache, n

    @staticmethod
    def _extra_of(args, kwargs) -> int:
        return int(kwargs.get("extra_hash", args[0] if args else 0) or 0)

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
        snap["disk_superseded"] = self.disk_superseded
        tiers = getattr(self.disk, "snapshot", None)
        if callable(tiers):
            snap["storage_tiers"] = tiers()
        if self.warm is not None:
            snap.update(self.warm.snapshot())
        return snap
