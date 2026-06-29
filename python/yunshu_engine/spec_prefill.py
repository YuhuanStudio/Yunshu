from __future__ import annotations

"""Yunshu SpecPrefill — attention-based sparse prefill for long prompts.

Studied from oMLX's specprefill.py, adapted for Yunshu:
Reduces TTFT on long prompts by using a small draft model to identify
important tokens, then prefilling only those tokens on the target model.

Pipeline:
  1. score_tokens()  — draft model scores token importance via attention capture
  2. select_chunks() — chunk-based top-K% selection
  3. sparse_prefill() — target prefill with selected tokens at original positions
  4. cleanup_rope()  — restore original RoPE after generation

Key insight: RoPE is relative — Q_m @ K_p^T depends only on (m-p).
Selected keys stored contiguously in cache with correct RoPE angles
produce correct attention during decode.

References:
  - oMLX specprefill.py (omlx/patches/specprefill.py)
  - arxiv.org/abs/2502.02789 (Speculative Prefill)
"""

import logging
import math
from typing import Any

import mlx.core as mx

logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_KEEP_RATE = 0.20
DEFAULT_THRESHOLD = 8192
DEFAULT_CHUNK_SIZE = 32
DEFAULT_LOOKAHEAD = 8


def _avg_pool1d(x: mx.array, kernel_size: int) -> mx.array:
    """1D average pooling along last axis via prefix-sum."""
    if kernel_size <= 1:
        return x
    pad = kernel_size // 2
    padded = mx.pad(x, [(0, 0)] * (x.ndim - 1) + [(pad, pad)])
    zeros = mx.zeros(x.shape[:-1] + (1,), dtype=x.dtype)
    prefix = mx.concatenate([zeros, mx.cumsum(padded, axis=-1)], axis=-1)
    return (prefix[..., kernel_size:] - prefix[..., :-kernel_size]) / kernel_size


# ---------------------------------------------------------------------------
# Attention capture
# ---------------------------------------------------------------------------


class _AttentionCapture:
    """Wraps attention to capture post-RoPE query vectors during lookahead.

    Delegates to the original attention module while recording queries
    for importance scoring.
    """

    def __init__(self, original, buf_idx, query_buffer, query_extractor):
        self._original = original
        self._buf_idx = buf_idx
        self._query_buffer = query_buffer
        self._query_extractor = query_extractor

    def __call__(self, x, mask=None, cache=None, **kwargs):
        queries = self._query_extractor(self._original, x, cache)
        self._query_buffer[self._buf_idx].append(queries)
        return self._original(x, mask=mask, cache=cache, **kwargs)

    def __getattr__(self, name):
        return getattr(self._original, name)


# ---------------------------------------------------------------------------
# Architecture-specific query extractors
# ---------------------------------------------------------------------------


def _qwen35_extract_queries(attn, x, cache=None):
    """Qwen3.5: gate split + q_norm + RoPE."""
    B, L, D = x.shape
    q_out = attn.q_proj(x)
    queries, _gate = mx.split(
        q_out.reshape(B, L, attn.num_attention_heads, -1), 2, axis=-1
    )
    queries = attn.q_norm(queries).transpose(0, 2, 1, 3)
    if cache is not None:
        queries = attn.rope(queries, offset=cache.offset)
    else:
        queries = attn.rope(queries)
    return queries


def _llama_extract_queries(attn, x, cache=None):
    """Standard transformer: q_proj + reshape + RoPE."""
    B, L, D = x.shape
    n_heads = getattr(
        attn,
        "num_attention_heads",
        getattr(attn, "n_heads", getattr(attn, "num_heads", None)),
    )
    queries = attn.q_proj(x)
    queries = queries.reshape(B, L, n_heads, -1).transpose(0, 2, 1, 3)
    if cache is not None:
        queries = attn.rope(queries, offset=cache.offset)
    else:
        queries = attn.rope(queries)
    return queries


def _detect_query_extractor(attn):
    """Auto-detect the right query extractor for an attention module."""
    if hasattr(attn, "q_norm") and hasattr(attn, "num_attention_heads"):
        # Qwen3.5 pattern: gated attention with q_norm
        return _qwen35_extract_queries
    if hasattr(attn, "q_proj"):
        # Standard LLaMA pattern
        return _llama_extract_queries
    return None


def _find_attention_layers(model):
    """Find all attention layers in the model."""
    results = []
    layers = getattr(model, "layers", [])
    if not layers:
        inner = getattr(model, "model", None)
        if inner is not None:
            layers = getattr(inner, "layers", [])
    for idx, layer in enumerate(layers):
        if hasattr(layer, "self_attn"):
            results.append((idx, layer))
    return results


def _get_attn_module(layer):
    """Get attention module from a layer."""
    if hasattr(layer, "self_attn"):
        return layer.self_attn
    return None


def _patch_attention_capture(model, query_buffer):
    """Patch attention layers with _AttentionCapture to capture queries.

    Returns (originals_dict, attn_layers) for later unpatching.
    """
    attn_layers = _find_attention_layers(model)
    originals = {}

    for layer_idx, layer in attn_layers:
        attn = _get_attn_module(layer)
        if attn is None:
            continue
        extractor = _detect_query_extractor(attn)
        if extractor is None:
            continue
        originals[layer_idx] = attn
        layer.self_attn = _AttentionCapture(attn, layer_idx, query_buffer, extractor)

    return originals, attn_layers


def _unpatch_attention_capture(model, originals, attn_layers):
    """Restore original attention modules."""
    for layer_idx, layer in attn_layers:
        if layer_idx in originals:
            layer.self_attn = originals[layer_idx]


def score_tokens(
    draft_model: Any,
    tokens: list[int] | mx.array,
    n_lookahead: int = DEFAULT_LOOKAHEAD,
    pool_kernel: int = 13,
    temp: float = 0.6,
    top_p: float = 0.95,
) -> mx.array:
    """Score token importance using draft model attention capture.

    wrap attention modules to capture query vectors
    during lookahead decode, then compute real attention scores Q @ K^T / sqrt(d).

    Args:
        draft_model: Small draft model for scoring.
        tokens: Prompt token IDs.
        n_lookahead: Decode steps for query capture.
        pool_kernel: Smoothing kernel size.
        temp: Sampling temperature for lookahead.
        top_p: Top-p for lookahead.

    Returns:
        mx.array of shape (M,) with per-token importance scores.
    """
    from mlx_lm.models.cache import make_prompt_cache

    tokens_list = tokens.tolist() if isinstance(tokens, mx.array) else list(tokens)

    n_prompt = len(tokens_list)
    prompt = mx.array(tokens_list)

    # Create draft cache
    cache = make_prompt_cache(draft_model)

    # Prefill draft model
    n = n_prompt
    processed = 0
    step_size = 2048
    while n - processed > 1:
        chunk = min(step_size, n - processed - 1)
        draft_model(prompt[processed : processed + chunk][None], cache=cache)
        mx.eval([c.state for c in cache if hasattr(c, "state")])
        processed += chunk
        mx.clear_cache()

    logits = draft_model(prompt[processed:][None], cache=cache)
    mx.eval(logits)

    # Patch attention to capture queries during lookahead
    query_buffer: dict[int, list] = {}
    originals, attn_layers = _patch_attention_capture(draft_model, query_buffer)

    try:
        # Lookahead decode to capture query vectors
        from mlx_lm.sample_utils import make_sampler

        sampler = make_sampler(temp=temp, top_p=top_p)
        y = sampler(logits[:, -1, :])
        mx.eval(y)

        for _ in range(n_lookahead):
            logits = draft_model(y.reshape(1, -1), cache=cache)
            y = sampler(logits[:, -1, :])
            mx.eval(y)
    finally:
        _unpatch_attention_capture(draft_model, originals, attn_layers)

    # Compute attention-based importance scores: Q @ K^T / sqrt(d_k)
    # Build a mapping from cache index to layer index for correct query pairing.
    # Cache entries and attention layers are in 1:1 correspondence: cache[i]
    # corresponds to attn_layers[i].  We track which cache indices have valid
    # keys so we can look up the right captured queries.
    _cache_to_layer: dict[int, int] = {}
    _valid_cache_idx = 0
    for _ci, c in enumerate(cache):
        if not hasattr(c, "keys") or c.keys is None:
            continue
        if c.keys.shape[-2] < n_prompt:
            continue
        _cache_to_layer[_ci] = _valid_cache_idx
        _valid_cache_idx += 1

    importance_scores = []
    for cache_idx, c in enumerate(cache):
        if not hasattr(c, "keys") or c.keys is None:
            continue
        keys = c.keys
        if keys.shape[-2] < n_prompt:
            continue
        prompt_keys = keys[..., :n_prompt, :].astype(mx.float32)

        # Look up captured queries for this cache entry's layer
        layer_idx = _cache_to_layer.get(cache_idx)
        layer_queries = query_buffer.get(layer_idx, []) if layer_idx is not None else []

        if not layer_queries:
            scores = mx.mean(mx.abs(prompt_keys), axis=-1)
            scores = mx.mean(scores, axis=1).squeeze(0)
            importance_scores.append(scores)
            continue

        # Real attention scoring: average Q @ K^T / sqrt(d) across lookahead steps
        d_k = prompt_keys.shape[-1]
        all_attn = []
        for q in layer_queries:
            # q: (B, n_heads, 1, d_k), prompt_keys: (B, n_heads, M, d_k)
            attn_weights = (
                q[..., -1:, :].astype(mx.float32) @ prompt_keys.transpose(0, 1, 3, 2)
            ) / math.sqrt(d_k)
            attn_weights = mx.softmax(attn_weights, axis=-1)
            all_attn.append(attn_weights.squeeze(2).squeeze(0))  # (n_heads, M)

        if all_attn:
            combined_q = mx.stack(all_attn, axis=0)  # (n_steps, n_heads, M)
            layer_score = mx.mean(combined_q, axis=(0, 1))  # (M,)
            importance_scores.append(layer_score)

    if not importance_scores:
        return mx.ones(n_prompt) / n_prompt

    combined = mx.stack(importance_scores, axis=0)
    combined = mx.max(combined, axis=0)

    if pool_kernel > 1:
        combined = _avg_pool1d(combined[None, :], pool_kernel).squeeze(0)

    mx.eval(combined)
    return combined


def select_chunks(
    importance: mx.array,
    keep_pct: float = DEFAULT_KEEP_RATE,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> mx.array:
    """Select top-K% token chunks by average importance.

    Groups tokens into chunks, ranks by average importance, and returns
    the sorted indices of tokens in the top chunks.

    Args:
        importance: (M,) per-token importance scores.
        keep_pct: Fraction of chunks to keep (0.1–0.5).
        chunk_size: Tokens per chunk.

    Returns:
        Sorted mx.array of kept token indices.
    """
    M = importance.shape[0]
    if keep_pct >= 1.0:
        return mx.arange(M)

    n_chunks = math.ceil(M / chunk_size)
    keep_n = max(1, math.ceil(n_chunks * keep_pct))

    chunk_scores = []
    for i in range(n_chunks):
        start = i * chunk_size
        end = min(start + chunk_size, M)
        chunk_scores.append(float(mx.mean(importance[start:end]).item()))

    top_chunks = sorted(range(n_chunks), key=lambda i: chunk_scores[i], reverse=True)
    top_chunks = sorted(top_chunks[:keep_n])

    indices: list[int] = []
    for ci in top_chunks:
        start = ci * chunk_size
        end = min(start + chunk_size, M)
        indices.extend(range(start, end))

    return mx.array(indices)


def manual_rope(
    x: mx.array,
    positions: mx.array,
    dims: int,
    base: float = 10000.0,
    scale: float = 1.0,
) -> mx.array:
    """Apply RoPE at arbitrary (non-contiguous) positions.

    Args:
        x: (B, n_heads, L, head_dim)
        positions: (L,) position indices
        dims: Number of dimensions to rotate
        base: RoPE base frequency
        scale: Position scale divisor
    """
    half = dims // 2
    inv_freq = 1.0 / (base ** (mx.arange(0, dims, 2, dtype=mx.float32) / dims))
    scaled_pos = positions.astype(mx.float32) / scale
    angles = scaled_pos[:, None] * inv_freq[None, :]
    cos_a = mx.cos(angles)[None, None, :, :]
    sin_a = mx.sin(angles)[None, None, :, :]
    x_rot, x_pass = x[..., :dims], x[..., dims:]
    x1, x2 = x_rot[..., :half], x_rot[..., half:]
    rotated = mx.concatenate(
        [x1 * cos_a - x2 * sin_a, x1 * sin_a + x2 * cos_a], axis=-1
    )
    return mx.concatenate([rotated, x_pass], axis=-1)


class _PositionMappedRoPE:
    """Applies RoPE at non-contiguous positions during sparse prefill."""

    def __init__(self, original_rope, all_positions, cache_start=0):
        self._original = original_rope
        self._all_positions = all_positions
        self._cache_start = (
            int(cache_start) if isinstance(cache_start, mx.array) else int(cache_start)
        )
        self._dims = getattr(original_rope, "dims", getattr(original_rope, "dim", None))
        self._base = getattr(original_rope, "base", 10000.0)
        self._scale = getattr(original_rope, "scale", 1.0)

    def __call__(self, x, offset=0):
        L = x.shape[2]
        idx = int(offset) - self._cache_start
        positions = self._all_positions[idx : idx + L]
        return manual_rope(x, positions, self._dims, base=self._base, scale=self._scale)


class _OffsetAdjustedRoPE:
    """Adds constant offset to RoPE positions for decode after sparse prefill.

    After sparse prefill of N tokens from M total:
      cache.offset = N + i, desired position = M + i
      adjustment = M - N
    """

    def __init__(self, original_rope, adjustment: int):
        self._original = original_rope
        self._adjustment = adjustment

    def __call__(self, x, offset=0):
        return self._original(x, offset=offset + self._adjustment)


def sparse_prefill(
    model: Any,
    tokens: list[int] | mx.array,
    selected_indices: mx.array,
    cache: list,
    step_size: int = 2048,
) -> mx.array:
    """Prefill model cache with selected tokens at their original positions.

    Runs the model on only selected tokens while preserving positional
    encoding via manual RoPE. After this call, cache contains KV entries
    with correct RoPE positions.

    Args:
        model: Target model with .layers property.
        tokens: (M,) all prompt token IDs.
        selected_indices: (N,) sorted indices into tokens to keep.
        cache: list of KVCache from make_prompt_cache().
        step_size: Chunk size for processing.

    Returns:
        logits from the last selected token.

    Side effects:
        - Populates cache with KV for selected tokens
        - Installs _OffsetAdjustedRoPE on attention layers
        - Call cleanup_rope(model) after generation to restore
    """
    if not isinstance(tokens, mx.array):
        tokens = mx.array(tokens)

    M = tokens.shape[0]
    selected_positions = selected_indices.astype(mx.int32)
    selected_tokens = tokens[selected_indices]
    N = selected_tokens.shape[0]

    # Find attention layers and patch RoPE
    attn_layers = _find_attention_layers(model)
    original_ropes = {}

    for layer_idx, layer in attn_layers:
        attn = _get_attn_module(layer)
        if attn is not None and hasattr(attn, "rope"):
            original_ropes[layer_idx] = attn.rope
            cache_start = 0
            for c in cache:
                if hasattr(c, "offset"):
                    cache_start = c.offset
                    break
            attn.rope = _PositionMappedRoPE(attn.rope, selected_positions, cache_start)

    try:
        n = int(N)
        processed = 0

        while n - processed > 1:
            chunk = min(step_size, n - processed - 1)
            model(selected_tokens[processed : processed + chunk][None], cache=cache)
            mx.eval([c.state for c in cache if hasattr(c, "state")])
            processed += chunk
            mx.clear_cache()

        logits = model(selected_tokens[processed:][None], cache=cache)
        mx.eval(logits)

    finally:
        # Replace position-mapped RoPE with offset-adjusted RoPE for decode
        total_prompt_len = M
        final_cache_offset = N
        adjustment = total_prompt_len - final_cache_offset

        for layer_idx, layer in attn_layers:
            attn = _get_attn_module(layer)
            if (
                attn is not None
                and hasattr(attn, "rope")
                and layer_idx in original_ropes
            ):
                original = original_ropes[layer_idx]
                if adjustment > 0:
                    attn.rope = _OffsetAdjustedRoPE(original, adjustment)
                else:
                    attn.rope = original

    return logits


def cleanup_rope(model: Any) -> None:
    """Restore original RoPE on all attention layers.

    Call after generation to remove _OffsetAdjustedRoPE wrappers.
    """
    for _, layer in _find_attention_layers(model):
        attn = _get_attn_module(layer)
        if attn is None or not hasattr(attn, "rope"):
            continue
        rope = attn.rope
        if isinstance(rope, (_OffsetAdjustedRoPE, _PositionMappedRoPE)):
            attn.rope = rope._original
