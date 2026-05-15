"""Yunshu Global MLX Executor — single-thread GPU serialization.

Studied from oMLX's engine_core.py:
- ALL MLX GPU work must run on one thread (mlx-lm's generation_stream is module-level)
- mx.new_thread_local_stream(mx.default_device()) creates a thread-safe Metal stream
- Without this, mx.eval() from other threads fails with "There is no Stream(gpu, N) in current thread"

This is the foundation for safe multi-model serving.
"""


import logging
import sys
from concurrent.futures import ThreadPoolExecutor

import mlx.core as mx

logger = logging.getLogger(__name__)

_executor: ThreadPoolExecutor | None = None


def _init_mlx_thread() -> None:
    """Initialize thread-local Metal stream on the executor thread.

    mlx-lm's generation_stream is created at import time in whichever thread
    imported it first (main thread). Arrays produced inside
    ``with mx.stream(generation_stream):`` carry that stream reference.
    If the stream was created on the main thread, subsequent .item() /
    mx.synchronize() calls from the executor thread fail with
    "There is no Stream(gpu, 0) in current thread".

    Fix: create a thread-local stream here and replace the module-level
    generation_stream in mlx_lm.generate.
    """
    stream = mx.new_thread_local_stream(mx.default_device())

    gen_mod = sys.modules.get("mlx_lm.generate")
    if gen_mod is not None:
        gen_mod.generation_stream = stream

    logger.info(f"MLX executor thread initialized: generation_stream = {stream}")


def get_mlx_executor() -> ThreadPoolExecutor:
    """Get or create the global MLX executor (lazy singleton).

    mlx-lm's BatchGenerator uses a module-level Metal stream (generation_stream),
    so ALL MLX GPU operations across all models MUST be serialized onto one thread
    to prevent Metal command buffer races that cause segfaults.
    """
    global _executor
    if _executor is None:
        _executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="mlx-global",
            initializer=_init_mlx_thread,
        )
    return _executor


def sync_and_clear_cache() -> None:
    """Synchronize in-flight GPU work before clearing the Metal buffer cache.

    Without synchronization, mx.clear_cache() can release Metal buffers that
    are still referenced by in-flight command buffers submitted via
    mx.async_eval(). This causes the GPU driver to hit a
    'completeMemory() prepare count underflow' kernel panic on M4 hardware
    (and SIGSEGV/SIGABRT on M3).

    Studied from oMLX scheduler.py:_sync_and_clear_cache().
    """
    try:
        gen_mod = sys.modules.get("mlx_lm.generate")
        if gen_mod is not None and hasattr(gen_mod, "generation_stream"):
            mx.synchronize(gen_mod.generation_stream)
    except RuntimeError:
        pass
    mx.synchronize()
    mx.clear_cache()
