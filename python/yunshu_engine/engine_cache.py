from __future__ import annotations

"""Engine cache extracted from batched_engine.

Runtime dependencies stay on the compatibility facade so existing patches apply.
"""


_PROGRESSIVE_QUANT_INTERVAL = 256


def _create_prompt_cache_with_quant(
    model, kv_quant_bits: int | None = None, kv_quant_group_size: int = 64
):
    """Create a prompt cache. KV quantization is applied LATER by the generation
    loop's ``_maybe_quantize_kv_cache`` once the cache offset reaches
    ``quantized_kv_start`` — this is the mlx-lm pattern (generate.py).

    This previously wrapped each layer as a QuantizedKVCache IMMEDIATELY
    at creation (from an EMPTY cache, ignoring ``quantized_kv_start``). 8-bit
    tolerated it, but 4-bit produced degenerate output ("a the a the a") even
    when the start threshold meant quantization should never engage — quantizing
    an empty cache and writing prefill into a 4-bit-quantized buffer corrupts the
    K/V. Deferring to the populated-cache ``to_quantized`` conversion (the
    upstream path) fixes 4-bit. ``kv_quant_bits`` is accepted for signature
    compatibility.
    """
    from mlx_lm.models.cache import make_prompt_cache

    return make_prompt_cache(model)


def _maybe_quantize_kv_cache(
    prompt_cache: list,
    quantized_kv_start: int,
    kv_group_size: int,
    kv_bits: int,
) -> None:
    """Quantize KV cache layers that have exceeded the start threshold.

    Follows mlx-lm's maybe_quantize_kv_cache pattern from generate.py:299.
    Uses MLX's native cache.to_quantized() for hardware-efficient 4/8-bit
    KV storage, reducing memory footprint by 2-4x for long sequences.
    """
    if kv_bits is None:
        return
    # RotatingKVCache (sliding-window layers in Gemma-3/Gemma-4/
    # gpt-oss/Cohere2/etc.) HAS a to_quantized method that just raises
    # NotImplementedError("RotatingKVCache Quantization NYI"). The old hasattr() guard
    # matched it → KV quant crashed 100% of requests for those models (a 500 on the
    # non-streaming path, a stream error on the streaming path). mlx-lm's own CLI never
    # quantizes sliding-window models; quantize only the global-attn KVCache layers.
    try:
        from mlx_lm.models.cache import RotatingKVCache
    except Exception:
        RotatingKVCache = ()  # type: ignore[assignment]
    for i, c in enumerate(prompt_cache):
        if isinstance(c, RotatingKVCache):
            continue
        if hasattr(c, "to_quantized") and hasattr(c, "offset"):
            if c.offset >= quantized_kv_start:
                prompt_cache[i] = c.to_quantized(group_size=kv_group_size, bits=kv_bits)


def _progressive_quantize_kv_cache(
    prompt_cache: list,
    quantized_kv_start: int,
    kv_group_size: int,
    kv_bits: int,
    current_token_count: int,
    interval: int = _PROGRESSIVE_QUANT_INTERVAL,
) -> None:
    """Progressively quantize KV cache during generation (C6 pattern).

    Called every N tokens during the generate loop to keep memory
    usage flat instead of peaking at full precision. Quantizes only
    newly eligible layers since last quantization pass.
    """
    if kv_bits is None or current_token_count % interval != 0:
        return
    _engine._maybe_quantize_kv_cache(
        prompt_cache, quantized_kv_start, kv_group_size, kv_bits
    )


def _prefix_cache_provenance(prefix_cache, prompt_cache_hit: bool, cached_tokens: int):
    """(tier, lookup_ms) of a fast-path request's cached prefix for ``x_yunshu.cache``."""
    if prompt_cache_hit:
        return "ram", None
    if not cached_tokens or prefix_cache is None:
        return "none", None
    lk = getattr(prefix_cache, "last_lookup", None) or {}
    tier = lk.get("tier", "hot")
    return ("ram" if tier == "hot" else tier), lk.get("ms")


# Resolve the facade after definitions to also support direct module imports.
from . import batched_engine as _engine
