"""Server-side change feed for the console: one sampler per topic, shared by every open tab.

A topic is a name plus an async ``fetch`` (the same function the polled endpoint runs, called
in-process) and a cadence. While at least one subscriber wants a topic, one task samples it,
compares the stable JSON and publishes a new *version* only when the content changed (or a
heartbeat is due, so ages and uptime stay fresh). A subscriber keeps one cursor per topic and
always receives the newest payload: there is no per-tab queue, so a slow tab skips versions
(reported as ``skipped``) and memory stays O(topics) however many tabs are open.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

HEARTBEAT_S = 10.0
KEEPALIVE_S = 15.0


@dataclass
class Topic:
    name: str
    fetch: Callable[[Any], Awaitable[Any]]
    #: seconds until the next sample, given the last payload
    interval: Callable[[Any], float]
    permission: str = "can_infer"
    #: top-level keys ignored when deciding whether the content changed
    volatile: frozenset[str] = frozenset()
    heartbeat_s: float = HEARTBEAT_S


@dataclass
class _State:
    version: int = 0
    data: str = ""  # serialized payload of the newest version
    stable: str = ""
    sent_at: float = 0.0
    error: str | None = None


@dataclass
class _Sampler:
    task: asyncio.Task | None = None
    wanted: int = 0
    request: Any = None
    loop: asyncio.AbstractEventLoop | None = None
    nudge: asyncio.Event | None = None


@dataclass
class Hub:
    topics: dict[str, Topic] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._state: dict[str, _State] = {}
        self._samplers: dict[str, _Sampler] = {}
        self._cond: asyncio.Condition | None = None
        self.seq = 0
        self.published = 0

    def _condition(self) -> asyncio.Condition:
        if self._cond is None:
            self._cond = asyncio.Condition()
        return self._cond

    def register(self, topic: Topic) -> None:
        self.topics[topic.name] = topic

    # ── sampling ───────────────────────────────────────────────────────
    async def _publish(self, topic: Topic, payload: Any, force: bool) -> bool:
        st = self._state.setdefault(topic.name, _State())
        body = payload
        if topic.volatile and isinstance(payload, dict):
            body = {k: v for k, v in payload.items() if k not in topic.volatile}
        stable = json.dumps(body, sort_keys=True, separators=(",", ":"), default=str)
        now = time.monotonic()
        due = now - st.sent_at >= topic.heartbeat_s
        if not (force or st.version == 0 or stable != st.stable or due):
            return False
        st.version += 1
        st.stable = stable
        st.data = json.dumps(payload, separators=(",", ":"), default=str)
        st.sent_at = now
        st.error = None
        self.seq += 1
        self.published += 1
        cond = self._condition()
        async with cond:
            cond.notify_all()
        return True

    async def _run(self, topic: Topic) -> None:
        sampler = self._samplers[topic.name]
        payload: Any = None
        while sampler.wanted > 0:
            if sampler.nudge is not None:
                sampler.nudge.clear()  # before the sample, so a poke during it is not lost
            try:
                payload = await topic.fetch(sampler.request)
                await self._publish(topic, payload, False)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - a bad sample must not kill the feed
                st = self._state.setdefault(topic.name, _State())
                st.error = f"{type(exc).__name__}: {exc}"[:200]
            try:
                delay = max(0.05, float(topic.interval(payload)))
            except Exception:  # noqa: BLE001
                delay = 2.0
            # sleep until the next sample is due, or until poke() says something changed
            nudge = sampler.nudge
            if nudge is None:
                await asyncio.sleep(delay)
                continue
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(nudge.wait(), timeout=delay)

    def poke(self, *names: str) -> None:
        """Sample these topics now (thread-safe): something they report just changed."""
        for n in names:
            s = self._samplers.get(n)
            if s is not None and s.loop is not None and s.nudge is not None:
                with contextlib.suppress(RuntimeError):  # the loop is closing
                    s.loop.call_soon_threadsafe(s.nudge.set)

    def _want(self, name: str, request: Any) -> None:
        s = self._samplers.setdefault(name, _Sampler())
        s.wanted += 1
        s.request = request
        if s.task is None or s.task.done():
            s.loop = asyncio.get_running_loop()
            s.nudge = asyncio.Event()
            s.task = s.loop.create_task(self._run(self.topics[name]))

    def _release(self, name: str) -> None:
        s = self._samplers[name]
        s.wanted -= 1
        if s.wanted <= 0 and s.task is not None:
            s.task.cancel()
            s.task = None
            # the next subscriber starts from a fresh snapshot
            self._state.pop(name, None)

    def samplers_running(self) -> int:
        return sum(1 for s in self._samplers.values() if s.task is not None)

    # ── subscribing ────────────────────────────────────────────────────
    async def subscribe(
        self,
        names: Iterable[str],
        request: Any,
        *,
        boot_id: str,
        is_disconnected: Callable[[], Awaitable[bool]] | None = None,
        keepalive_s: float = KEEPALIVE_S,
    ):
        """Yield SSE text frames: ``hello``, then a snapshot and every change per topic."""
        names = [n for n in dict.fromkeys(names) if n in self.topics]
        for n in names:
            self._want(n, request)
        seen = {n: 0 for n in names}
        cond = self._condition()
        yield _frame("hello", self.seq, {"boot_id": boot_id, "topics": names})
        last_write = time.monotonic()
        try:
            while True:
                if is_disconnected is not None and await is_disconnected():
                    return
                wrote = False
                for n in names:
                    st = self._state.get(n)
                    if st is None or st.version == seen[n]:
                        continue
                    skipped = st.version - seen[n] - 1 if seen[n] else 0
                    seen[n] = st.version
                    yield (
                        f"event: {n}\nid: {self.seq}\n"
                        f'data: {{"v":{st.version},"skipped":{skipped},"d":{st.data}}}\n\n'
                    )
                    wrote = True
                if wrote:
                    last_write = time.monotonic()
                    continue

                def moved() -> bool:
                    return any(
                        (st := self._state.get(n)) is not None and st.version != seen[n]
                        for n in names
                    )

                try:
                    async with cond:
                        await asyncio.wait_for(cond.wait_for(moved), timeout=1.0)
                except TimeoutError:
                    pass
                if time.monotonic() - last_write > keepalive_s:
                    yield ": keepalive\n\n"
                    last_write = time.monotonic()
        finally:
            for n in names:
                self._release(n)


def _frame(event: str, seq: int, data: dict) -> str:
    return f"event: {event}\nid: {seq}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n"
