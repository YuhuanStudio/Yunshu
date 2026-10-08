"""Bounded host metadata for APC lifecycle; no cache tensors or token ids retained."""

from __future__ import annotations

import collections
import time

_AUTO = object()


class Observation:
    def __init__(self):
        self.events = collections.deque(maxlen=512)
        self.hits = collections.OrderedDict()
        self.request_id = None
        self.reason = "superseded"
        self.serial = 0

    def event(self, action, key, tokens, tier, reason, request_id=_AUTO, device=None):
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
                "request_id": self.request_id if request_id is _AUTO else request_id,
                **({"device": device} if device else {}),
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


_MISSING = object()


class ObservedCache(collections.OrderedDict):
    def __init__(self, observation, spill=None):
        super().__init__()
        self.observation = observation
        self.spill = spill
        self.last_access_key = None

    def move_to_end(self, key, last=True):
        super().move_to_end(key, last=last)
        if last:
            self.last_access_key = key

    def popitem(self, last=True):
        key, entry = super().popitem(last=last)
        self.observation.event(
            "eviction",
            key,
            len(entry.token_ids),
            "ram",
            "memory_pressure"
            if self.observation.reason == "memory_pressure"
            else "lru",
        )
        if not last and self.spill is not None:
            self.spill(key, entry)
        return key, entry

    def pop(self, key, default=_MISSING):
        entry = super().pop(key) if default is _MISSING else super().pop(key, default)
        if default is _MISSING or entry is not default:
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


def disk_event(store, action, key, tokens, reason):
    """Cold storage operations append metadata only; no tensor or file reads."""
    owner = getattr(store, "owner", None) or store
    observation = getattr(owner, "observation", None)
    if observation is None:
        return
    origin = getattr(owner, "event_origin", None)
    request_id = origin(key) if origin else None
    observation.event(
        action,
        key,
        tokens,
        "ssd",
        reason,
        request_id=request_id,
        device=getattr(store, "name", None),
    )
