"""Zero-forward exact repeat for the round driver: replay a prompt's final block.

The canonical prefill ends with a tail of 1-64 rows after the last 64-aligned boundary. An APC
hit restores the aligned checkpoint and still runs that tail forward (about 65 ms fixed at 8K).
When the same prompt is sent again, the result of that forward is already known: the tail's
K/V rows, the recurrent state after the prompt, the final hidden rows (the sampler and the lane
head run on them as on a fresh forward) and the drafter's captured tail features. They are
stored once and written back instead of recomputed; they are the outputs of the same forward a
hit would run, so everything downstream sees identical values. The same idea as
``yunshu_engine.prompt_end_replay`` (which serves the stock batch path), on the driver's caches.
"""

from __future__ import annotations

import collections
import logging
from dataclasses import dataclass
from typing import Any

import mlx.core as mx

logger = logging.getLogger(__name__)
TAIL_MAX = 64
MAX_ENTRIES = 3
MAX_BYTES = 768 << 20
_STORE: collections.OrderedDict = collections.OrderedDict()
_STATS = {"stored": 0, "replayed": 0, "evicted": 0}


@dataclass
class Entry:
    base: int
    rows: int
    slots: list  # per layer ("a", state dict), ("k", (keys, values)) or ("q", (keys, values, indexer))
    hidden: mx.array  # final-norm hidden rows [rows, D]
    captures: list  # drafter capture rows per layer ([rows, D] each), may be empty
    nbytes: int


def stats() -> dict[str, int]:
    return {**_STATS, "entries": len(_STORE), "bytes": resident_bytes()}


def resident_bytes() -> int:
    return sum(e.nbytes for e in _STORE.values())


def clear() -> None:
    _STORE.clear()
    for k in _STATS:
        _STATS[k] = 0


def _owned(a):
    return None if a is None else a * 1


def _is_kv(c: Any) -> bool:
    return getattr(c, "keys", None) is not None


def _has_indexer(c: Any) -> bool:
    """A Flash-Next QSA cache: the indexer's raw keys and positions ride with the K/V rows."""
    return getattr(c, "index_keys", None) is not None


def eligible(start: int, end: int, n: int) -> bool:
    """The span is the prompt's final block of at most ``TAIL_MAX`` rows (the span
    plan makes it the same span for a cold run and a restored one)."""
    return end == n and 0 < end - start <= TAIL_MAX


def lookup(key, cache: list, start: int, end: int) -> Entry | None:
    entry: Entry | None = _STORE.get(key)
    if entry is None or (entry.base, entry.rows, len(entry.slots)) != (
        start,
        end - start,
        len(cache),
    ):
        return None
    for c, (tag, payload) in zip(cache, entry.slots, strict=True):
        if tag in ("k", "q"):
            if (
                not _is_kv(c)
                or int(c.offset) != start
                or (tag == "q") != _has_indexer(c)
            ):
                return None
        elif _is_kv(c) or len(payload["cache"]) != len(c.cache):
            return None
    return entry


def restore(entry: Entry, key, cache: list) -> None:
    """Write the stored block into ``cache`` (copies: decode updates state in place)."""
    for c, (tag, payload) in zip(cache, entry.slots, strict=True):
        if tag == "k":
            c.update_and_fetch(*payload)
        elif tag == "q":
            keys, values, indexer = payload
            c.update_and_fetch(keys, values)
            c.update_indexer(indexer["keys"], indexer["positions"])
            # the block summaries after the stored forward (None: rebuilt from the raw keys, as after a trim)
            c.index_block_keys = _owned(indexer["blocks"])
            c.index_block_ratio = indexer["ratio"]
        else:
            c.__dict__.update(payload)
            c.cache = [_owned(a) for a in payload["cache"]]
    _STORE.move_to_end(key)
    _STATS["replayed"] += 1
    logger.info("round driver replay: %d tail rows after %d", entry.rows, entry.base)


def store(key, cache: list, start: int, end: int, hidden, captures) -> None:
    """Keep the block a real forward just produced (``cache`` is after that forward)."""
    slots: list = []
    copies: list = []
    nbytes = 0
    for c in cache:
        if _is_kv(c):
            if int(c.offset) != end:
                return
            rows = (
                _owned(c.keys[..., start:end, :]),
                _owned(c.values[..., start:end, :]),
            )
            if _has_indexer(c):
                if int(c.index_keys.shape[1]) != end:
                    return
                indexer = {
                    "keys": _owned(c.index_keys[:, start:end]),
                    "positions": _owned(c.index_position_ids[..., start:end]),
                    "blocks": _owned(c.index_block_keys),
                    "ratio": c.index_block_ratio,
                }
                extra = [a for a in indexer.values() if isinstance(a, mx.array)]
                slots.append(("q", (*rows, indexer)))
                copies.extend([*rows, *extra])
                nbytes += sum(int(a.nbytes) for a in extra)
            else:
                slots.append(("k", rows))
                copies.extend(rows)
            nbytes += int(rows[0].nbytes) + int(rows[1].nbytes)
        elif hasattr(c, "cache"):
            meta = dict(c.__dict__)
            meta["cache"] = [_owned(a) for a in c.cache]
            slots.append(("a", meta))
            live = [a for a in meta["cache"] if a is not None]
            copies.extend(live)
            nbytes += sum(int(a.nbytes) for a in live)
        else:
            return
    hidden = _owned(hidden)
    caps = [_owned(x) for x in captures]
    nbytes += int(hidden.nbytes) + sum(int(x.nbytes) for x in caps)
    if nbytes > MAX_BYTES:
        return
    mx.eval(copies, hidden, caps)
    _STORE[key] = Entry(start, end - start, slots, hidden, caps, nbytes)
    _STORE.move_to_end(key)
    _STATS["stored"] += 1
    while len(_STORE) > MAX_ENTRIES or (
        resident_bytes() > MAX_BYTES and len(_STORE) > 1
    ):
        _STORE.popitem(last=False)
        _STATS["evicted"] += 1
