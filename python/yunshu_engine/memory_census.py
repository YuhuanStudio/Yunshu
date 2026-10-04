"""Who holds MLX memory: live ``mx.array`` objects grouped by the object that references them.

MLX's active-memory gauge says how much is held, not by whom. This walks the Python
object graph once (``gc``), finds every reachable array at or above a size floor, and
names its holder as a short path (``KVCache.keys``, ``dict['cache']`` under its owner).
Views that share a buffer are each counted, so the total is an upper bound; compare it
with ``mx.get_active_memory()``. Diagnostic only: it pauses the interpreter for a
moment, so it is not called on the serving path.
"""

from __future__ import annotations

import gc
from typing import Any

_MIB = 1 << 20


def _label(obj: Any) -> str:
    return type(obj).__qualname__


def _owner_of_dict(d: dict) -> Any | None:
    for ref in gc.get_referrers(d):
        if getattr(ref, "__dict__", None) is d:
            return ref
    return None


def _attr(owner: Any, value: Any) -> str | None:
    """Attribute of ``owner`` (an instance) holding ``value``, if any."""
    try:
        items = vars(owner).items()
    except TypeError:
        return None
    return next((k for k, v in items if v is value), None)


def _name(container: Any, value: Any) -> str | None:
    """'Owner.attr' or 'dict[key]' when ``container`` holds ``value`` directly."""
    if isinstance(container, dict):
        key = next((k for k, v in container.items() if v is value), None)
        if key is None:
            return None
        owner = _owner_of_dict(container)
        return f"{_label(owner)}.{key}" if owner is not None else f"dict[{key!r}]"
    if not isinstance(container, (list, tuple)):
        key = _attr(container, value)
        return f"{_label(container)}.{key}" if key is not None else None
    return None


def _holder(container: Any, array: Any) -> str:
    """'Owner.attr' / 'Owner.attr[list]' / 'dict[key]' for one referencing container."""
    direct = _name(container, array)
    if direct is not None:
        return direct
    if isinstance(container, (list, tuple)):
        for parent in gc.get_referrers(container):
            name = _name(parent, container)
            if name is not None:
                return f"{name}[{_label(container)}]"
        return f"{_label(container)}[]"
    return _label(container)


def census(min_mib: float = 64.0, top: int = 40) -> dict[str, Any]:
    import mlx.core as mx

    gc.collect()
    floor = int(min_mib * _MIB)
    seen: dict[int, tuple[Any, int]] = {}
    for obj in gc.get_objects():
        try:
            refs = gc.get_referents(obj)
        except Exception:  # noqa: BLE001
            continue
        for r in refs:
            if isinstance(r, mx.array) and id(r) not in seen:
                n = r.nbytes
                if n >= floor:
                    seen[id(r)] = (r, n)
    internal = {id(seen)} | {id(v) for v in seen.values()}
    groups: dict[str, dict[str, Any]] = {}
    for r, n in seen.values():
        holders = [
            _holder(c, r)
            for c in gc.get_referrers(r)
            if id(c) not in internal
            and (isinstance(c, (dict, list, tuple)) or hasattr(c, "__dict__"))
        ]
        key = " | ".join(sorted(set(holders))[:3]) or "(no python holder)"
        g = groups.setdefault(
            key, {"holder": key, "arrays": 0, "mib": 0.0, "shapes": []}
        )
        g["arrays"] += 1
        g["mib"] += n / _MIB
        if len(g["shapes"]) < 3:
            g["shapes"].append(f"{tuple(r.shape)} {r.dtype}")
    rows = sorted(groups.values(), key=lambda g: -g["mib"])
    for g in rows:
        g["mib"] = round(g["mib"], 1)
    return {
        "active_mib": round(mx.get_active_memory() / _MIB, 1),
        "cache_mib": round(mx.get_cache_memory() / _MIB, 1),
        "counted_mib": round(sum(n for _, n in seen.values()) / _MIB, 1),
        "min_mib": min_mib,
        "holders": rows[:top],
    }
