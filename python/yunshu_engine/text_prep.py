"""Run a request's chat-template render + tokenization off the event loop.

The text fast paths used to render the Jinja chat template and encode the whole prompt
inside the request coroutine: 5-50 ms for a tool-heavy history and 20-80 ms for a 30k-token
prompt, during which the loop served no other connection (no SSE frame, keepalive or
disconnect poll for the stream decoding next to it). Both are pure CPU on the tokenizer
(the Rust encoder releases the GIL), so they run on one helper thread.

One thread, not a pool: prep for concurrent requests stays serialized exactly as it was on
the loop, so two requests never drive the shared tokenizer's encode state at once, and the
MLX thread (decode only) is not involved. The caller's context variables (the per-request
tool list the template reads) travel with the call.
"""

from __future__ import annotations

import asyncio
import contextvars
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

_POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix="yunshu-textprep")


async def run_text_prep[T](fn: Callable[[], T]) -> T:
    """``fn()`` on the text-prep thread, with the caller's contextvars; awaits its result."""
    ctx = contextvars.copy_context()
    return await asyncio.get_running_loop().run_in_executor(_POOL, ctx.run, fn)
