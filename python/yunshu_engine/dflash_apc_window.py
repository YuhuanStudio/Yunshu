# Upstream (patches): Blaizzy/mlx-vlm (MIT) mlx_vlm/generate/ar.py PromptProcessingBatch @ v0.7.3
"""Keep the DFlash drafter's context window across APC prefix hits.

The DFlash2 drafter reads the target's captured hidden states of the last
``context_window`` (about 2047) prompt positions. mlx-vlm's ``SpeculativePrefill`` builds them
only from the chunks one prefill call runs, so after an APC hit (KV restored, only the
suffix prefilled) the drafter starts with a handful of positions and drafts badly (8K code
warm: 3.7 instead of 5.2 commits/round). The target verifies every draft, so tokens are
unchanged; only acceptance suffers.

At each APC checkpoint store the window that ends exactly there, keyed by the checkpoint's
extra hash and token prefix; on a hit seed the new prefill's chunk list with it. A missing
window (evicted, other drafter) leaves the previous behaviour.

The RAM budget holds about two 32K windows, and an SSD-restored checkpoint (another process,
or evicted from RAM) had none: 32K decode after such a hit ran ~10% slower (tf110 1021h2,
prose 69.7 -> 62.9 tok/s, same tokens). With an APC disk tier the window that ends at the
prompt is also written beside it (``set_disk``), read back on a RAM miss.
"""

from __future__ import annotations

import collections
import hashlib
import logging
from typing import Any

logger = logging.getLogger(__name__)

_STATE: dict[str, Any] = {"installed": False, "current": None, "budget": 0}
DISK_CAP = 4 << 30  # bytes of windows kept on disk per namespace (~40 at 32K)
# disk store: directory, the volume's APC budget (its free-space reserve), the drafter tag
# (its target layers: the window is their captured states) and the writer thread
_DISK: dict[str, Any] = {"dir": None, "budget": None, "tag": None, "queue": None}
_STORE: collections.OrderedDict = collections.OrderedDict()
_STATS = {
    "stored": 0,
    "seeded": 0,
    "evicted": 0,
    "missed": 0,
    "disk_written": 0,
    "disk_read": 0,
    "disk_skipped": 0,
}


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
        layers = _disk_get(key)
        if layers is None:
            _STATS["missed"] += 1
            return None
        nbytes = sum(int(x.nbytes) for x in layers)
        if nbytes <= _STATE["budget"]:
            put(key, layers, nbytes)
        return layers
    _STORE.move_to_end(key)
    return hit[0]


# -- disk store ----------------------------------------------------------------------


def set_disk(directory, budget=None, tag=None) -> None:
    """Keep prompt-end windows under ``directory`` (None: RAM only). ``budget`` is the
    volume's APC DiskBudget (its free-space reserve applies); ``tag`` names the drafter's
    captured layers and is part of every file name."""
    from pathlib import Path

    _DISK["dir"] = None if directory is None else Path(directory)
    _DISK["budget"] = budget
    if tag is not None:
        _DISK["tag"] = tag


def set_tag(tag) -> None:
    _DISK["tag"] = tag


def _disk_path(key: tuple):
    directory, tag = _DISK["dir"], _DISK["tag"]
    if directory is None or tag is None:
        return None
    name = hashlib.blake2b(
        repr((tag, key[0], key[1])).encode() + key[2], digest_size=16
    )
    return directory / f"{name.hexdigest()}.bf16.npy"


def persist(key: tuple) -> bool:
    """Write the RAM window under ``key`` to disk (the prompt-end one; called by the APC
    store). The copy to host memory runs here, on the MLX thread; the file write runs on a
    writer thread."""
    path = _disk_path(key)
    hit = _STORE.get(key)
    if path is None or hit is None:
        return False
    import mlx.core as mx
    import numpy as np

    layers = hit[0]
    if any(x.dtype != mx.bfloat16 for x in layers):
        return False
    budget = _DISK["budget"]
    nbytes = hit[1]
    if budget is not None and not budget.allow_write(nbytes):
        _STATS["disk_skipped"] += 1
        return False
    stacked = mx.stack(layers).view(mx.uint16)
    mx.eval(stacked)
    host = np.array(stacked, copy=True)
    _writer().put((path, host))
    return True


def _writer():
    import queue
    import threading

    q = _DISK["queue"]
    if q is None:
        q = _DISK["queue"] = queue.Queue()
        threading.Thread(target=_write_loop, args=(q,), daemon=True).start()
    return q


def _write_loop(q) -> None:
    import numpy as np

    while True:
        path, host = q.get()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(path.name + ".tmp")
            with open(tmp, "wb") as f:
                np.save(f, host)
            tmp.replace(path)
            _STATS["disk_written"] += 1
            budget = _DISK["budget"]
            if budget is not None:
                budget.note_written(path)
            _disk_trim(path.parent)
        except Exception:  # the window is an optimisation only
            logger.debug("drafter window write failed", exc_info=True)
        finally:
            q.task_done()


def _disk_trim(directory) -> None:
    files = sorted(directory.glob("*.bf16.npy"), key=lambda f: f.stat().st_mtime)
    total = sum(f.stat().st_size for f in files)
    while files and total > DISK_CAP:
        oldest = files.pop(0)
        total -= oldest.stat().st_size
        oldest.unlink(missing_ok=True)


def _disk_get(key: tuple):
    path = _disk_path(key)
    if path is None or not path.exists():
        return None
    import os

    import mlx.core as mx
    import numpy as np

    try:
        host = np.load(path)
        stacked = mx.array(host).view(mx.bfloat16)
        os.utime(path)  # recently used: the trim keeps it
    except Exception:
        logger.debug("drafter window read failed", exc_info=True)
        return None
    _STATS["disk_read"] += 1
    return [stacked[i] for i in range(int(stacked.shape[0]))]


def flush() -> None:
    """Wait for queued window writes (tests, shutdown)."""
    q = _DISK["queue"]
    if q is not None:
        q.join()


def window_from_chunks(chunks: list, keep: int) -> tuple[list, int] | None:
    """Last ``keep`` positions of per-layer chunk lists as ``(layers, nbytes)``."""
    import mlx.core as mx

    if not chunks:
        return None
    layers = [
        mx.concatenate(list(parts), axis=1) for parts in zip(*chunks, strict=True)
    ]
    if int(layers[0].shape[1]) > keep:
        layers = [x[:, -keep:] for x in layers]
    # Never block here: this runs inside the prefill step, ahead of the first token.
    mx.async_eval(layers)
    return layers, sum(int(x.nbytes) for x in layers)


def capture(token_ids, extra_hash: int, *, prompt_end: bool = False) -> bool:
    """Called from the coordinator's ``store_checkpoint`` while a prefill step is running;
    ``prompt_end``: the checkpoint ends the request's prompt (its window goes to disk too)."""
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
    key = key_for(extra_hash, token_ids)
    stored = put(key, layers, nbytes)
    if stored and prompt_end:
        try:
            persist(key)
        except Exception:  # the window is an optimisation only
            logger.debug("drafter window persist failed", exc_info=True)
    return stored


def seed(batch) -> bool:
    """On a prefix hit, start the drafter-context chunks with the stored window."""
    sp = getattr(batch, "_speculative_prefill", None)
    if sp is None or not getattr(sp, "kwargs", None) or len(batch.uids) != 1:
        return False
    metas = getattr(batch, "_apc_meta", None) or []
    meta = metas[0] if metas else None
    prefix = int((meta or {}).get("prefix_len") or 0)
    if meta is None or prefix <= 0 or sp.chunks:
        return False
    layers = get(key_for(meta.get("extra_hash", 0), meta["full_input_ids"][:prefix]))
    if layers is None:
        return False
    sp.chunks = [list(layers)]
    _STATS["seeded"] += 1
    logger.info(
        "DFlash drafter window restored from APC hit: %d positions (prefix %d)",
        int(layers[0].shape[1]),
        prefix,
    )
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
