"""Bounded host metadata for APC lifecycle; no cache tensors or token ids retained."""

from __future__ import annotations

import collections
import time


class Observation:
    def __init__(self):
        self.events = collections.deque(maxlen=512)
        self.hits = collections.OrderedDict()
        self.request_id = None
        self.reason = "superseded"
        self.serial = 0

    def event(self, action, key, tokens, tier, reason, request_id=None):
        self.serial += 1
        self.events.append(
            {
                "id": self.serial,
                "t": time.time(),
                "action": action,
                "entry_id": str(key),
                "tokens": tokens,
                "tier": tier,
                "reason": reason,
                "request_id": request_id or self.request_id,
            }
        )

    def hit(self, key):
        count, _ = self.hits.pop(key, (0, None))
        self.hits[key] = (count + 1, time.time())
        while len(self.hits) > 4096:
            self.hits.popitem(last=False)

    def row(self, key, namespace, tokens, logical, physical, tier):
        hits, last = self.hits.get(key, (0, None))
        return {
            "id": str(key),
            "namespace": str(namespace),
            "tokens": tokens,
            "bytes_logical": logical,
            "bytes_physical": physical,
            "tier": tier,
            "last_hit": last,
            "hits": hits,
        }


class ObservedCache(collections.OrderedDict):
    def __init__(self, observation, spill=None):
        super().__init__()
        self.observation = observation
        self.spill = spill

    def popitem(self, last=True):
        key, entry = super().popitem(last=last)
        self.observation.event("eviction", key, len(entry.token_ids), "ram", "lru")
        if not last and self.spill is not None:
            self.spill(key, entry)
        return key, entry

    def pop(self, key, default=None):
        entry = super().pop(key, default)
        if entry is not default:
            self.observation.event(
                "eviction", key, len(entry.token_ids), "ram", self.observation.reason
            )
        return entry

    def clear(self):
        for key, entry in self.items():
            self.observation.event(
                "eviction", key, len(entry.token_ids), "ram", "admin_clear"
            )
        super().clear()
