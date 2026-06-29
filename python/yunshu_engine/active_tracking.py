"""In-flight request tracking for the non-LLM engines.

The ModelManager only auto-evicts (LRU / memory-pressure) a model it can PROVE is
idle — i.e. whose engine exposes a callable ``has_active_requests()`` returning
False (see model_manager._find_lru_victim + _unload_model_locked's fail-safe).
LLM/VLM engines have this (BatchedEngine._active_fast_path_count). The media engines
(Image/Video/TTS/ASR/STS/OCR) did NOT, so they were eviction-EXEMPT: under memory
pressure their idle weights could never be reclaimed (→ MemoryError on the next LLM
load) and the selector had to skip them to avoid a livelock.

This module gives those engines a real, leak-proof in-flight counter so they can
participate in safe LRU eviction. The try/finally that guarantees the decrement
lives in ONE place (the decorators), so no per-method hand-written counter can leak
(a leaked counter would make a model permanently un-evictable — the soft version of
the bug we are fixing).

CRITICAL: decorate EVERY public generation entry point — including the streaming
(async-generator) ones. A method left untracked reports idle while it runs, so the
model could be torn down mid-generation (the crash the fail-safe prevents). Decorating a
method that internally calls another decorated method is fine: the count just rises
to 2 and falls back symmetrically; has_active_requests only tests > 0.
"""

from __future__ import annotations

import functools
import threading


class ActiveRequestMixin:
    """Mix in to give an engine a thread-safe in-flight request counter and the
    ``has_active_requests()`` the ModelManager probes. State is created lazily so
    no ``__init__`` cooperation is required from the engine."""

    def _ensure_active_state(self) -> None:
        # Double-checked lazy init. The first-time race is benign (two threads may
        # both create the lock); guard with a module-level lock to be exact.
        if not hasattr(self, "_active_count"):
            with _INIT_LOCK:
                if not hasattr(self, "_active_count"):
                    self._active_lock = threading.Lock()
                    self._active_count = 0

    def has_active_requests(self) -> bool:
        self._ensure_active_state()
        with self._active_lock:
            return self._active_count > 0

    def _active_inc(self) -> None:
        self._ensure_active_state()
        with self._active_lock:
            self._active_count += 1

    def _active_dec(self) -> None:
        self._ensure_active_state()
        with self._active_lock:
            self._active_count = max(0, self._active_count - 1)


_INIT_LOCK = threading.Lock()


def tracks_active(method):
    """Wrap an ``async def`` (coroutine) entry point so the engine counts it as an
    in-flight request for its whole duration; the decrement is guaranteed via
    finally (success, exception, or cancellation)."""

    @functools.wraps(method)
    async def _wrapper(self, *args, **kwargs):
        self._active_inc()
        try:
            return await method(self, *args, **kwargs)
        finally:
            self._active_dec()

    return _wrapper


def tracks_active_gen(method):
    """Wrap an ``async def`` async-generator entry point. The decrement runs when
    the wrapper generator is closed — normal exhaustion, an exception, or the
    consumer breaking early (which triggers aclose on GC). Preserves streaming."""

    @functools.wraps(method)
    async def _wrapper(self, *args, **kwargs):
        self._active_inc()
        try:
            async for item in method(self, *args, **kwargs):
                yield item
        finally:
            self._active_dec()

    return _wrapper
