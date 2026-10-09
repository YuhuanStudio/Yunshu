# Patches upstream (MIT): mlx-vlm qwen4_exp language/cache modules (module-level ``mx`` view that memoizes
# constant ``mx.arange``; tracked in vendor.json; Blaizzy/mlx-vlm).
"""Memoized constant index tensors for the Qwen4-Exp decode step.

A Flash-Next decode step builds ~145 ``mx.arange(...)`` index tensors with constant arguments (block ids,
slot ids, step ids, ...); each is a GPU kernel.  The arrays are immutable, so a small cache keyed by
(start, stop, step, dtype) returns the same evaluated array on every later step.  Values are identical, so
output is unchanged; arguments that grow with the context (key lengths) are not cached.
"""

from __future__ import annotations

import types
from collections import OrderedDict
from typing import Any

import mlx.core as mx

MAX_ENTRIES = 512
MAX_LEN = 8192
_STATE: dict[str, Any] = {"hits": 0, "misses": 0, "installed": []}


class _CachedMx(types.ModuleType):
    """``mlx.core`` whose ``arange`` memoizes small integer ranges."""

    def __init__(self) -> None:
        super().__init__("mlx.core")
        self._cache: OrderedDict = OrderedDict()

    def __getattr__(self, name: str) -> Any:
        return getattr(mx, name)

    def arange(self, *args, **kwargs):
        try:
            if kwargs.keys() - {"dtype"} or not all(type(a) is int for a in args):
                raise TypeError
            if (
                len(args) == 1
                and args[0] > MAX_LEN
                or len(args) >= 2
                and args[1] - args[0] > MAX_LEN
            ):
                raise TypeError
            key = (args, kwargs.get("dtype"))
            hash(key)
        except TypeError:
            return mx.arange(*args, **kwargs)
        hit = self._cache.get(key)
        if hit is not None:
            _STATE["hits"] += 1
            self._cache.move_to_end(key)
            return hit
        _STATE["misses"] += 1
        out = mx.arange(*args, **kwargs)
        self._cache[key] = out
        while len(self._cache) > MAX_ENTRIES:
            self._cache.popitem(last=False)
        return out


def install() -> bool:
    """Route ``mx`` inside the qwen4_exp language module through the memoizing view (idempotent)."""
    try:
        from mlx_vlm.models.qwen4_exp import language
    except Exception:  # noqa: BLE001 - mlx_vlm without qwen4_exp
        return False
    if isinstance(language.mx, _CachedMx):
        return True
    language.mx = _CachedMx()
    _STATE["installed"].append("qwen4_exp.language")
    return True


def uninstall() -> None:
    from mlx_vlm.models.qwen4_exp import language

    if isinstance(language.mx, _CachedMx):
        language.mx = mx


def stats() -> dict[str, Any]:
    return {"hits": _STATE["hits"], "misses": _STATE["misses"]}
