"""Per-process bounded LRU cache, no disk persistence."""

import time
from collections import OrderedDict
from dataclasses import dataclass
from urllib.parse import urlsplit, urlunsplit

from ..webfetch import FetchResult


def normalized_url(url: str) -> str:
    p = urlsplit(url)
    return urlunsplit((p.scheme.lower(), p.netloc.lower(), p.path or "/", p.query, ""))


@dataclass
class Entry:
    value: FetchResult
    expires: float
    size: int


class PageCache:
    def __init__(self, max_bytes: int = 64 * 1024 * 1024, clock=time.monotonic):
        self.max_bytes, self.clock = max_bytes, clock
        self.rows: OrderedDict[str, Entry] = OrderedDict()
        self.bytes = 0

    def get(self, url: str, *, stale: bool = False) -> FetchResult | None:
        key = normalized_url(url)
        row = self.rows.get(key)
        if row is None or (not stale and row.expires <= self.clock()):
            return None
        self.rows.move_to_end(key)
        return row.value

    def put(self, url: str, value: FetchResult, ttl: float = 900) -> None:
        key = normalized_url(url)
        size = (
            len(value.text.encode())
            + len(key.encode())
            + len(value.title.encode())
            + 512
        )
        if old := self.rows.pop(key, None):
            self.bytes -= old.size
        if size > self.max_bytes:
            return
        self.rows[key] = Entry(value, self.clock() + ttl, size)
        self.bytes += size
        while self.bytes > self.max_bytes:
            _, old = self.rows.popitem(last=False)
            self.bytes -= old.size


pages = PageCache()
