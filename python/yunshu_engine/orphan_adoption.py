"""Orphan adoption: a generation whose client vanished is kept for a short window.

Agent clients (Claude Code, Codex) give up on a long streamed reply after their own
timeout and re-send the identical request (often non-streaming). Cancelling at
disconnect throws away minutes of decoding that the retry then repeats from scratch.
Here the generation keeps running into a buffer for ``YUNSHU_ORPHAN_GRACE_SECONDS``;
an identical request arriving inside the window is attached to it (buffered tokens
replay at once, the rest follow live). At the window's end nothing adopted it, so it
is cancelled. Two LIVE identical requests are never merged: only an orphan is adopted.

Everything here runs on the event loop; the engine's executor threads only touch
``real_cancel`` (a threading.Event the runner already polls).
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import threading
import time
from collections.abc import AsyncIterator, Callable
from typing import Any

logger = logging.getLogger(__name__)

# Request fields that do not change what is generated.
_VOLATILE = frozenset(
    {
        "cancel_event",
        "timeout_seconds",
        "timeout",
        "priority",
        "prompt_cache_key",
        "prompt_cache_retention",
        "prompt_cache_options",
        "prompt_cache_plan",
        "user",
        "metadata",
        "request_id",
        "stream",
        "stream_options",
    }
)


class AutoSeed(int):
    """A seed the server made up (the caller gave none): not part of a request's identity."""


# What a field means when a request omits it (the engine's defaults), so a stream and a
# non-streaming request that spell the same thing differently still match.
_DEFAULTS: dict[str, Any] = {
    "temperature": 0.7,
    "top_p": 1.0,
    "top_k": 0,
    "min_p": 0.0,
    "repetition_penalty": 1.0,
    "xtc_probability": 0.0,
    "xtc_threshold": 0.0,
    "logprobs": False,
    "top_logprobs": None,
    "stop": None,
    "stop_token_ids": None,
    "enable_thinking": None,
}


def _normalize(params: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in sorted(params.items()):
        if k in _VOLATILE:
            continue
        if k == "seed" and (v is None or isinstance(v, AutoSeed)):
            v = "auto"
        if k in _DEFAULTS and (
            v == _DEFAULTS[k] or (v in ([], ()) and _DEFAULTS[k] is None)
        ):
            continue
        out[k] = v
    out.setdefault("seed", "auto")
    return out


def key_parts(model: str, messages: Any, params: dict[str, Any]) -> dict[str, str]:
    """Short hash per request field, to log which field made two requests differ."""
    out = {"model": model, "messages": messages, **_normalize(params)}
    return {
        k: hashlib.sha1(
            json.dumps(v, sort_keys=True, default=repr).encode()
        ).hexdigest()[:6]
        for k, v in out.items()
    }


def request_key(model: str, messages: Any, params: dict[str, Any]) -> str | None:
    """Stable identity of a generation request, or None when it cannot be keyed
    (an object without a stable JSON form, such as a callable): no adoption then."""
    body = {"model": model, "messages": messages, "params": _normalize(params)}
    try:
        text = json.dumps(body, sort_keys=True, default=_unstable)
    except _UnkeyableError:
        return None
    return hashlib.sha256(text.encode()).hexdigest()


class _UnkeyableError(Exception):
    pass


def _unstable(obj: Any) -> Any:
    if isinstance(obj, (set, frozenset)):
        return sorted(obj, key=repr)
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "__dataclass_fields__"):
        return {f: getattr(obj, f) for f in obj.__dataclass_fields__}
    raise _UnkeyableError(type(obj).__name__)


class Generation:
    """One engine generation: a pump task fills ``buffer``; consumers read by index."""

    def __init__(self, key: str, real_cancel: threading.Event):
        self.key = key
        self.real_cancel = real_cancel
        self.buffer: list[Any] = []
        self.finished = False
        self.error: BaseException | None = None
        self.attached = True
        self.created = time.monotonic()
        self.detached_at: float | None = None
        self.deadline: float | None = None
        self.changed = asyncio.Event()
        self.pump: asyncio.Task | None = None
        self.timer: asyncio.TimerHandle | None = None
        self.adopted = 0
        self.parts: dict[str, str] = {}

    def notify(self) -> None:
        ev, self.changed = self.changed, asyncio.Event()
        ev.set()

    @property
    def tokens(self) -> int:
        return (
            max((getattr(o, "completion_tokens", 0) or 0) for o in self.buffer)
            if self.buffer
            else 0
        )

    @property
    def adoptable(self) -> bool:
        """Finished-cleanly or still running; a cancelled / failed run is not."""
        if self.error is not None or self.real_cancel.is_set():
            return False
        if self.finished and self.buffer:
            last = self.buffer[-1]
            return not getattr(last, "error", None) and last.finish_reason not in (
                "cancel",
                "error",
                "timeout",
            )
        return True


class OrphanRegistry:
    """Detached generations by request key (attached ones are not adoptable)."""

    def __init__(self) -> None:
        self._orphans: dict[str, Generation] = {}
        self.adopted_total = 0
        self.expired_total = 0
        self.saved_tokens = 0

    # ── introspection ──
    def snapshot(self) -> list[dict[str, Any]]:
        now = time.monotonic()
        return [
            {
                "key": g.key[:12],
                "tokens": g.tokens,
                "running": not g.finished,
                "orphaned_s": round(now - (g.detached_at or now), 1),
                "expires_in_s": round(max(0.0, (g.deadline or now) - now), 1),
            }
            for g in self._orphans.values()
        ]

    def stats(self) -> dict[str, Any]:
        return {
            "orphans": self.snapshot(),
            "adopted_total": self.adopted_total,
            "expired_total": self.expired_total,
            "saved_tokens": self.saved_tokens,
        }

    # ── lifecycle ──
    def take(self, key: str) -> Generation | None:
        gen = self._orphans.get(key)
        if gen is None:
            return None
        if not gen.adoptable:
            self._drop(gen, cancel=True)
            return None
        del self._orphans[key]
        if gen.timer is not None:
            gen.timer.cancel()
            gen.timer = None
        gen.attached = True
        gen.adopted += 1
        gen.detached_at = gen.deadline = None
        self.adopted_total += 1
        self.saved_tokens += gen.tokens
        return gen

    def detach(self, gen: Generation, grace: float, limit: int) -> None:
        """The consumer is gone: keep the generation for ``grace`` seconds."""
        gen.attached = False
        if grace <= 0 or not gen.adoptable:
            self._drop(gen, cancel=True)
            return
        loop = asyncio.get_running_loop()
        now = time.monotonic()
        gen.detached_at = now
        gen.deadline = now + grace
        older = self._orphans.get(gen.key)
        if older is not None and older is not gen:
            self._drop(older, cancel=True)
        self._orphans[gen.key] = gen
        gen.timer = loop.call_later(grace, self._expire, gen)
        while len(self._orphans) > max(1, limit):
            oldest = min(self._orphans.values(), key=lambda g: g.detached_at or 0.0)
            self._drop(oldest, cancel=True)
        logger.info(
            "orphan kept: %d tokens so far, %s, window %.0fs",
            gen.tokens,
            "finished" if gen.finished else "still decoding",
            grace,
        )

    def _expire(self, gen: Generation) -> None:
        if self._orphans.get(gen.key) is gen and not gen.attached:
            self.expired_total += 1
            logger.info("orphan expired unadopted: %d tokens wasted", gen.tokens)
            self._drop(gen, cancel=True)

    def _drop(self, gen: Generation, cancel: bool) -> None:
        if self._orphans.get(gen.key) is gen:
            del self._orphans[gen.key]
        if gen.timer is not None:
            gen.timer.cancel()
            gen.timer = None
        if cancel:
            gen.real_cancel.set()
        if not gen.attached:
            gen.buffer = []

    def cancel_all(self) -> None:
        for gen in list(self._orphans.values()):
            self._drop(gen, cancel=True)


_registry = OrphanRegistry()


def get_registry() -> OrphanRegistry:
    return _registry


async def _pump(gen: Generation, agen: AsyncIterator[Any]) -> None:
    try:
        async for out in agen:
            if not gen.attached and gen.real_cancel.is_set():
                break
            gen.buffer.append(out)
            gen.notify()
    except asyncio.CancelledError:
        raise
    except BaseException as exc:  # surfaced to whoever reads the generation
        gen.error = exc
    finally:
        gen.finished = True
        gen.notify()
        aclose = getattr(agen, "aclose", None)
        if aclose is not None:
            try:
                await aclose()
            except BaseException:  # noqa: BLE001 - cleanup only
                logger.debug("orphan pump aclose failed", exc_info=True)


def _publish_stats(
    gen: Generation, caller_cancel: Any, out: Any, replay_n: int
) -> None:
    """Give the request's tracker the run statistics it reports to the console.

    The first consumer shares the engine's live ``RunStats``. An adopter gets a rebased copy:
    its clock starts at adoption and ``generated`` counts only what was decoded after it, so
    replayed tokens never read as a burst of speed or a negative TTFT (``replayed`` keeps the
    count for the per-response stats)."""
    orig = getattr(gen.real_cancel, "run_stats", None)
    if orig is None:
        return
    if replay_n < 0:
        caller_cancel.run_stats = orig
        return
    mine = getattr(caller_cancel, "run_stats", None)
    now = time.perf_counter()
    if mine is None or mine is orig:
        import dataclasses

        mine = dataclasses.replace(orig)
        mine.t_submit = mine.t_admit = mine.t_first = mine.t_last = now
        mine.prefill_done = mine.prefill_total
        mine.generated = 0
        mine.replayed = replay_n  # type: ignore[attr-defined]
        caller_cancel.run_stats = mine
    done = int(getattr(out, "completion_tokens", 0) or 0)
    if done > replay_n:
        mine.generated = done - replay_n
        mine.t_last = now
    if getattr(out, "finished", False):
        mine.finish_reason = getattr(out, "finish_reason", None) or "stop"


def explicitly_cancelled(cancel_event: Any) -> bool:
    """Someone asked for the cancel (the cancel endpoints, a deadline), as opposed to a
    disconnect: that must stop the work at once."""
    return bool(getattr(cancel_event, "explicit_cancel", False))


async def run(
    key: str | None,
    start: Callable[[threading.Event], AsyncIterator[Any]],
    caller_cancel: threading.Event,
    grace: float,
    limit: int,
    registry: OrphanRegistry | None = None,
    parts: dict[str, str] | None = None,
) -> AsyncIterator[Any]:
    """Serve one request: adopt an orphan with ``key`` or start ``start(real_cancel)``.

    Yields the generation's outputs. If the caller goes away mid-way (its cancel event is
    set or the iterator is closed) the generation is parked for ``grace`` seconds instead
    of being cancelled.
    """
    reg = registry or _registry
    gen = reg.take(key) if (key and grace > 0) else None
    if gen is not None:
        logger.info(
            "orphan adopted: %d tokens replayed, %s",
            gen.tokens,
            "already finished" if gen.finished else "still decoding",
        )
    else:
        if key and grace > 0 and reg._orphans and parts:
            for other in reg._orphans.values():
                diff = sorted(
                    k
                    for k in set(parts) | set(other.parts)
                    if parts.get(k) != other.parts.get(k)
                )
                logger.info("orphan not adopted: request differs in %s", diff)
        real_cancel = threading.Event()
        for attr in ("scheduling_priority", "client_request_id"):
            if hasattr(caller_cancel, attr):
                setattr(real_cancel, attr, getattr(caller_cancel, attr))
        gen = Generation(key or "", real_cancel)
        gen.parts = parts or {}
        gen.pump = asyncio.ensure_future(_pump(gen, start(real_cancel).__aiter__()))
    index = 0
    clean = False
    replay_n = gen.tokens if gen.adopted else -1  # tokens the adopter gets at once
    try:
        while True:
            if caller_cancel.is_set():
                return
            while index < len(gen.buffer):
                out = gen.buffer[index]
                index += 1
                _publish_stats(gen, caller_cancel, out, replay_n)
                yield out
                if caller_cancel.is_set():
                    return
            if gen.finished and index >= len(gen.buffer):
                clean = True
                if gen.error is not None:
                    raise gen.error
                return
            waiter = gen.changed
            if index < len(gen.buffer) or gen.finished:
                continue
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(waiter.wait(), timeout=0.25)
            if caller_cancel.is_set():
                return
    finally:
        if clean:
            reg._drop(gen, cancel=False)
        elif explicitly_cancelled(caller_cancel) or not key or grace <= 0:
            reg._drop(gen, cancel=True)
        else:
            reg.detach(gen, grace, limit)
