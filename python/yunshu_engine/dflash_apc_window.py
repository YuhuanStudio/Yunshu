"""Keep the DFlash drafter's context window across APC prefix hits.

The DFlash2 drafter reads the target's captured hidden states of the last
``context_window`` (about 2047) prompt positions. mlx-vlm's ``SpeculativePrefill`` builds them
only from the chunks one prefill call runs, so after an APC hit (KV restored, only the
suffix prefilled) the drafter starts with a handful of positions and drafts badly (8K code
warm: 3.7 instead of 5.2 commits/round). The target verifies every draft, so tokens are
unchanged; only acceptance suffers.

At each APC checkpoint store the window that ends exactly there, keyed by the checkpoint's
extra hash and token prefix; on a hit seed the new prefill's chunk list with it. A missing
window (evicted, SSD-restored checkpoint, other drafter) leaves the previous behaviour.
"""

from __future__ import annotations

import collections
import hashlib
import logging
from typing import Any

logger = logging.getLogger(__name__)

_STATE: dict[str, Any] = {"installed": False, "current": None, "budget": 0}
_STORE: collections.OrderedDict = collections.OrderedDict()
_STATS = {"stored": 0, "seeded": 0, "evicted": 0, "missed": 0}


def key_for(extra_hash: int, token_ids) -> tuple:
    import numpy as np

    arr = np.asarray(list(token_ids), dtype=np.int64)
    return (
        int(extra_hash),
        len(arr),
        hashlib.blake2b(arr.tobytes(), digest_size=16).digest(),
    )


def resident_bytes() -> int:
    return sum(nb for _, nb in _STORE.values())


def stats() -> dict:
    return {**_STATS, "entries": len(_STORE), "bytes": resident_bytes()}


def clear() -> None:
    _STORE.clear()


def set_budget(nbytes: int) -> None:
    _STATE["budget"] = int(nbytes)
    _trim()


def _trim() -> None:
    while _STORE and resident_bytes() > _STATE["budget"]:
        _STORE.popitem(last=False)
        _STATS["evicted"] += 1


def put(key: tuple, layers: list, nbytes: int) -> bool:
    if nbytes > _STATE["budget"]:
        return False
    _STORE.pop(key, None)
    _STORE[key] = (layers, nbytes)
    _STATS["stored"] += 1
    _trim()
    return key in _STORE


def get(key: tuple):
    hit = _STORE.get(key)
    if hit is None:
        _STATS["missed"] += 1
        return None
    _STORE.move_to_end(key)
    return hit[0]


def window_from_chunks(chunks: list, keep: int) -> tuple[list, int] | None:
    """Last ``keep`` positions of per-layer chunk lists as ``(layers, nbytes)``."""
    import mlx.core as mx

    if not chunks:
        return None
    layers = [mx.concatenate(parts, axis=1) for parts in zip(*chunks, strict=True)]
    if int(layers[0].shape[1]) > keep:
        layers = [x[:, -keep:] for x in layers]
    layers = [mx.contiguous(x) for x in layers]
    mx.eval(layers)
    return layers, sum(int(x.nbytes) for x in layers)


def capture(token_ids, extra_hash: int) -> bool:
    """Called from the coordinator's ``store_checkpoint`` while a prefill step is running."""
    batch = _STATE["current"]
    if batch is None or _STATE["budget"] <= 0:
        return False
    sp = getattr(batch, "_speculative_prefill", None)
    keep = getattr(sp, "_yunshu_keep", None)
    if sp is None or not sp.kwargs or keep is None or len(batch.uids) != 1:
        return False
    built = window_from_chunks(list(sp.chunks), keep)
    if built is None:
        return False
    layers, nbytes = built
    return put(key_for(extra_hash, token_ids), layers, nbytes)


def seed(batch) -> bool:
    """On a prefix hit, start the drafter-context chunks with the stored window."""
    sp = getattr(batch, "_speculative_prefill", None)
    if sp is None or not getattr(sp, "kwargs", None) or len(batch.uids) != 1:
        return False
    metas = getattr(batch, "_apc_meta", None) or []
    meta = metas[0] if metas else None
    prefix = int((meta or {}).get("prefix_len") or 0)
    if prefix <= 0 or sp.chunks:
        return False
    layers = get(key_for(meta.get("extra_hash", 0), meta["full_input_ids"][:prefix]))
    if layers is None:
        return False
    sp.chunks = [list(layers)]
    _STATS["seeded"] += 1
    return True


def install(budget_bytes: int | None = None) -> bool:
    if budget_bytes is not None:
        set_budget(budget_bytes)
    if _STATE["installed"]:
        return True
    from mlx_vlm.generate import ar

    cls = ar.PromptProcessingBatch
    orig_init, orig_step = cls.__init__, cls.prompt_step

    def __init__(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        try:
            seed(self)
        except Exception:  # lossless fallback: drafter keeps the suffix-only context
            logger.debug("drafter window seed failed", exc_info=True)

    def prompt_step(self):
        _STATE["current"] = self
        try:
            return orig_step(self)
        finally:
            _STATE["current"] = None

    cls.__init__, cls.prompt_step = __init__, prompt_step
    _STATE["installed"] = True
    return True
