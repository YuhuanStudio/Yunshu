"""Backbone-agnostic serving-capability layer (Part 2/3).

Yunshu serves two model families through two engines:
- ``BatchedEngine`` drives mlx-lm language models (the L4 fast path + continuous
  batching + 4-tier KV hierarchy).
- ``VLMEngine`` drives mlx-vlm multimodal models (vision/audio + text).

The engines are NOT being merged — their decode drivers legitimately differ
(vision feature caches, mRoPE, mlx_vlm's ``model.language_model`` wrapper). But
the *serving decisions* that gate shared infrastructure — chiefly "can this
backbone reuse a cached KV prefix losslessly?" — depend only on a small set of
**backbone capabilities**, not on which engine you are.

This module is that pluggable middle layer: a pure, dependency-light way to
classify a loaded model's KV cache and derive its serving capabilities, so both
engines (and any future backend) make the SAME decision from the SAME logic
instead of hard-coding per-engine checks.

It encodes the three hard-won lossless-reuse disqualifiers (see
docs/VLM_TEXT_KV_PREFIX.md):
1. hybrid recurrent caches (ArraysCache) — not sliceable;
2. sliding-window caches (RotatingKVCache) — circular buffer loses linear history;
3. mRoPE — position tracked outside the cache, reset per request.
"""
from __future__ import annotations

import enum
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


class BackendKind(enum.StrEnum):
    """Which model family a backend serves."""
    LM = "lm"          # mlx-lm language model (BatchedEngine)
    VLM = "vlm"        # mlx-vlm multimodal model (VLMEngine)


@dataclass(frozen=True)
class CacheClass:
    """Classification of a model's per-layer KV cache, derived purely from the
    cache layer objects (no model forward needed)."""
    has_sliding_window: bool   # any RotatingKVCache / has max_size (window attn)
    is_hybrid: bool            # any non-sliceable recurrent layer (ArraysCache)
    layer_types: tuple[str, ...]

    @property
    def resumable(self) -> bool:
        """Whether a stored snapshot of this cache can be trimmed/resumed
        losslessly across requests. Sliding-window and hybrid caches cannot."""
        return not (self.has_sliding_window or self.is_hybrid)


def classify_cache(cache_layers: Iterable[Any]) -> CacheClass:
    """Classify KV cache layers (the output of ``make_prompt_cache``).

    Detection is by layer TYPE / attributes, NOT by ``can_trim_prompt_cache`` —
    that probe inspects an EMPTY cache and returns True for a rotating cache that
    has not rotated yet, which is exactly the trap that made gemma look reusable.
    """
    types: list[str] = []
    has_sw = False
    is_hybrid = False
    for c in cache_layers:
        tname = type(c).__name__
        types.append(tname)
        # Sliding-window / rotating attention: bounded circular KV buffer.
        if "Rotating" in tname or hasattr(c, "max_size"):
            has_sw = True
        # Recurrent / linear-attention state (GatedDeltaNet etc.): no sliceable
        # keys/values, or an explicit non-trimmable flag.
        has_kv = hasattr(c, "keys") and hasattr(c, "values")
        if not has_kv:
            is_hybrid = True
        else:
            f = getattr(c, "is_trimmable", None)
            if callable(f):
                try:
                    if not f():
                        is_hybrid = True
                except Exception:
                    is_hybrid = True
    return CacheClass(has_sliding_window=has_sw, is_hybrid=is_hybrid,
                      layer_types=tuple(types))


@dataclass(frozen=True)
class BackendCapabilities:
    """Serving-relevant capabilities of a loaded model, independent of engine.

    ``supports_kv_prefix_reuse`` is the single most important gate: it is True
    only when cross-request KV prefix reuse is byte-lossless for this backbone.
    """
    kind: BackendKind
    cache: CacheClass
    is_mrope: bool             # multimodal RoPE (position tracked outside cache)

    @property
    def supports_kv_prefix_reuse(self) -> bool:
        """Whether the KV CACHE can be reused losslessly across requests. Depends
        only on cache resumability. mRoPE does NOT disqualify — its position must
        be supplied explicitly by the caller (see ``requires_explicit_positions``);
        VLMEngine's text path does this. A caller that cannot supply explicit
        positions should gate on ``supports_kv_prefix_reuse and not
        requires_explicit_positions``."""
        return self.cache.resumable

    @property
    def requires_explicit_positions(self) -> bool:
        """mRoPE tracks token position OUTSIDE the KV cache and resets it per
        request, so a reused prefix's suffix would prefill at position 0 instead
        of ``matched``. The caller MUST pass explicit ``position_ids`` (sequential
        from the cache offset, for text-only) on every forward to reuse safely."""
        return self.is_mrope

    def bypass_reason(self) -> str | None:
        """Human-readable reason the cache cannot be reused, or None."""
        if self.cache.is_hybrid:
            return "hybrid/recurrent cache (ArraysCache) not sliceable"
        if self.cache.has_sliding_window:
            return "sliding-window (RotatingKVCache) loses linear history"
        return None


def derive_capabilities(
    kind: BackendKind,
    cache_layers: Iterable[Any],
    *,
    is_mrope: bool = False,
) -> BackendCapabilities:
    """Build the capability descriptor from a model's KV cache layers + the
    mRoPE flag. ``cache_layers`` is what ``make_prompt_cache(model)`` returns."""
    return BackendCapabilities(
        kind=kind,
        cache=classify_cache(cache_layers),
        is_mrope=bool(is_mrope),
    )


@runtime_checkable
class ModelBackend(Protocol):
    """The interface a serving backend exposes to shared L2–L5 infrastructure.

    Both ``BatchedEngine`` (LM) and ``VLMEngine`` (VLM) satisfy the lifecycle +
    generation surface already; this Protocol documents the contract and lets
    shared components type against a backend rather than a concrete engine. It is
    intentionally minimal — the goal is a pluggable middle layer for serving
    *decisions*, not to force the two decode drivers to merge.
    """

    model_name: str

    async def start(self) -> None: ...
    async def stop(self) -> None: ...
    def backend_capabilities(self) -> BackendCapabilities: ...
