"""Yunshu Global MLX Executor — single-thread GPU serialization.

Studied from oMLX's engine_core.py:
- ALL MLX GPU work must run on one thread (mlx-lm's generation_stream is module-level)
- mx.new_thread_local_stream(mx.default_device()) creates a thread-safe Metal stream
- Without this, mx.eval() from other threads fails with "There is no Stream(gpu, N) in current thread"

This is the foundation for safe multi-model serving.
"""

import contextlib
import logging
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import mlx.core as mx

logger = logging.getLogger(__name__)

_executor: ThreadPoolExecutor | None = None
_executor_lock = threading.Lock()


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

    Thread-safe: concurrent callers will not create duplicate executors.

    mlx-lm's BatchGenerator uses a module-level Metal stream (generation_stream),
    so ALL MLX GPU operations across all models MUST be serialized onto one thread
    to prevent Metal command buffer races that cause segfaults.
    """
    global _executor
    if _executor is None:
        with _executor_lock:
            if _executor is None:
                _executor = ThreadPoolExecutor(
                    max_workers=1,
                    thread_name_prefix="mlx-global",
                    initializer=_init_mlx_thread,
                )
    return _executor


def reset_mlx_executor() -> ThreadPoolExecutor:
    """Reset the global MLX executor, recreating it from scratch.

    Used by EngineCore's BrokenThreadPool recovery to ensure all
    subsequent get_mlx_executor() calls return the new executor,
    maintaining the single-thread GPU serialization guarantee.
    """
    global _executor
    with _executor_lock:
        if _executor is not None:
            with contextlib.suppress(Exception):
                _executor.shutdown(wait=False)
        _executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="mlx-global",
            initializer=_init_mlx_thread,
        )
        logger.info("MLX executor reset after thread pool failure")
        return _executor


def shutdown_mlx_executor(wait: bool = True) -> None:
    """Shut down the global MLX executor thread.

    Called during engine shutdown to release the thread and its resources.
    After shutdown, get_mlx_executor() will create a fresh executor if needed.

    Args:
        wait: If True, block until all submitted GPU work completes.
    """
    global _executor
    with _executor_lock:
        if _executor is not None:
            if wait:
                try:
                    sync_and_clear_cache()
                except Exception:
                    logger.debug(
                        "sync_and_clear_cache during shutdown failed", exc_info=True
                    )
            _executor.shutdown(wait=wait)
            _executor = None
            logger.info("MLX executor shut down")


def sync_and_clear_cache() -> None:
    """Synchronize in-flight GPU work before clearing the Metal buffer cache.

    Without synchronization, mx.clear_cache() can release Metal buffers that
    are still referenced by in-flight command buffers submitted via
    mx.async_eval(). This causes the GPU driver to hit a
    'completeMemory() prepare count underflow' kernel panic on M4 hardware
    (and SIGSEGV/SIGABRT on M3).

    Safe to call even if the MLX executor has not been started or was
    already shut down — synchronization happens on the current thread.

    Studied from oMLX scheduler.py:_sync_and_clear_cache().
    """
    try:
        gen_mod = sys.modules.get("mlx_lm.generate")
        if gen_mod is not None and hasattr(gen_mod, "generation_stream"):
            stream = getattr(gen_mod, "generation_stream", None)
            if stream is not None:
                mx.synchronize(stream)
    except RuntimeError:
        logger.debug(
            "generation_stream synchronize failed, falling back to global sync",
            exc_info=True,
        )
    mx.synchronize()
    mx.clear_cache()
