"""Release the speculative round generator of a finished request.

mlx-vlm's ``GenerationBatch._start_rounds`` builds a ``stop_check`` closure over
the batch and passes it to the rounds generator that the batch itself stores in
``_rounds_iter``. After the last token the consumer never resumes that generator,
so it stays suspended inside a reference cycle (batch -> generator -> frame ->
closure -> batch) holding the request's KV caches and the last round's arrays
until the cyclic GC runs. With the fast DFlash tree those arrays carry live
Metal events and a server died with "Failed to create Metal shared event".
"""

from __future__ import annotations

import contextlib
from typing import Any


def release_rounds(gen: Any) -> bool:
    """Close and detach the rounds generator; True when one was released."""
    batch: Any = getattr(gen, "_generation_batch", None)
    rounds = getattr(batch, "_rounds_iter", None)
    if rounds is None:
        return False
    batch._rounds_iter = None
    with contextlib.suppress(Exception):
        rounds.close()
    return True


def snapshot_caches(gen: Any, uids: list) -> list | None:
    """The live speculative batch's cache objects, taken before a step that may finish its last row.

    mlx-vlm 259802e (v0.7.9+) ``SpeculativeGenerationBatch._release()`` rebinds
    ``batch.prompt_cache`` to ``[]`` (and drops ``hidden`` / ``shared_kv_states`` /
    ``prompt_tokens``) the moment the last row finishes, so a read of the batch after that step
    sees nothing. The runner offers the finished KV set to the restore-buffer pool; it reads this
    snapshot instead. Only the batch that is exactly this request's (never an older finished one)
    is captured, and the runner drops the snapshot at ``_finish`` so it never pins a dead context.
    """
    batch: Any = getattr(gen, "_generation_batch", None)
    if batch is None or list(getattr(batch, "uids", None) or ()) != list(uids):
        return None
    caches = getattr(batch, "prompt_cache", None)
    return list(caches) if caches else None


def finished_caches(gen: Any, snapshot: list | None) -> list:
    """Caches of a finished speculative batch: the batch's own if still held (before 259802e),
    else the snapshot taken before its release."""
    batch: Any = getattr(gen, "_generation_batch", None)
    return list(getattr(batch, "prompt_cache", None) or snapshot or [])
