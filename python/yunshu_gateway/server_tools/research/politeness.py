"""Origin admission, robots and failure cooldown. All downloads use webfetch."""

import asyncio
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from urllib.parse import urlsplit, urlunsplit
from urllib.robotparser import RobotFileParser

from ..webfetch import FetchError, _fetch_url


@dataclass
class Host:
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    slots: asyncio.Semaphore = field(default_factory=lambda: asyncio.Semaphore(2))
    next_request: float = 0
    interval: float = 1
    failures: int = 0
    blocked_until: float = 0
    robots: RobotFileParser | None = None
    robots_expires: float = 0


class Politeness:
    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self.hosts: OrderedDict[str, Host] = OrderedDict()

    def host(self, url: str) -> Host:
        p = urlsplit(url)
        key = urlunsplit((p.scheme, p.netloc.lower(), "", "", ""))
        if key not in self.hosts:
            # Don't evict active admission state (which would bypass host limits).
            if len(self.hosts) >= 2048:
                raise FetchError("too_many_requests", "Origin admission table is full")
            self.hosts[key] = Host()
        return self.hosts[key]

    async def admit(self, state: Host):
        async with state.lock:
            now = self.clock()
            if now < state.blocked_until:
                raise FetchError("too_many_requests", "Origin cooling down")
            await asyncio.sleep(max(0, state.next_request - now))
            state.next_request = self.clock() + state.interval

    def failed(self, state: Host, error: FetchError):
        state.failures += 1
        if error.code == "too_many_requests" or state.failures >= 3:
            state.blocked_until = self.clock() + max(300, error.retry_after or 0)

    async def allowed(self, url: str, state: Host, **kwargs) -> bool:
        async with state.lock:
            if state.robots_expires <= self.clock():
                p = urlsplit(url)
                robot_url = urlunsplit((p.scheme, p.netloc, "/robots.txt", "", ""))
                now = self.clock()
                if now < state.blocked_until:
                    return False
                await asyncio.sleep(max(0, state.next_request - now))
                state.next_request = self.clock() + state.interval
                parser = RobotFileParser()
                try:
                    result = await _fetch_url(robot_url, **kwargs)
                    if result.truncated:
                        return False  # missing rules must not accidentally authorize a fetch
                    parser.parse(result.text.splitlines())
                except FetchError as exc:
                    if exc.status_code in (404, 410):
                        parser.parse([])
                    else:
                        self.failed(state, exc)
                        return False  # unknown robots: skip enrichment, keep snippets
                state.robots, state.robots_expires = parser, self.clock() + 86400
            if state.robots:
                delay = state.robots.crawl_delay("YunshuFetch") or 1
                rate = state.robots.request_rate("YunshuFetch")
                state.interval = max(
                    1, float(delay), rate.seconds / rate.requests if rate else 1
                )
            return bool(state.robots and state.robots.can_fetch("YunshuFetch", url))


politeness = Politeness()
