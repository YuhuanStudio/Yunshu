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
    batch = getattr(gen, "_generation_batch", None)
    rounds = getattr(batch, "_rounds_iter", None)
    if rounds is None:
        return False
    batch._rounds_iter = None
    with contextlib.suppress(Exception):
        rounds.close()
    return True
