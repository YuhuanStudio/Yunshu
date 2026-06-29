from __future__ import annotations

"""Yunshu BatchedEngine — user-facing continuous batching engine.

Written from scratch:
- Wraps AsyncEngineCore for clean separation of concerns
- Lazy model loading on first request
- Chat template application before generation
- GeneratorExit-safe cleanup for streaming
- OpenAI-compatible response types

Architecture:
  BatchedEngine (user-facing API)
    → AsyncEngineCore (orchestration)
      → Scheduler (BatchGenerator management)
      → RequestOutputCollector (per-request output buffer)

This is the engine that ModelManager and the gateway routers use.
"""

import asyncio
import logging
import os
import platform
import threading
import time
from collections.abc import AsyncIterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from typing import Any

from .text_utils import StopHoldbackBuffer

logger = logging.getLogger(__name__)


def _is_cancelled(event: Any) -> bool:
    """Thread-safe cancel check. Works from the MLX executor thread.

    asyncio.Event.is_set() reads ._value (GIL-protected bool), which is
    safe from any thread in CPython.  Using the explicit attribute avoids
    the thread-safety warning from calling asyncio APIs off the event loop.
    """
    if event is None:
        return False
    if isinstance(event, asyncio.Event):
        return event._value
    return event.is_set()


_REASONING_EFFORT_MAP = {"low": 2048, "medium": 8192, "high": 32768}
_MAX_STREAMING_TEXT_BUFFER = (
    1 * 1024 * 1024
)  # 1MB safety limit for streaming text buffer


def _resolve_think_token_ids(tokenizer):
    """Resolve the single-token ids for the <think> / </think> markers, or (None, None).

    The old call sites encoded "<think" / "</think" WITHOUT the closing '>',
    which for the canonical thinking models (Qwen3/Qwen3.5/DeepSeek-R1) tokenizes to TWO
    tokens (e.g. Qwen3.5 `</think` → [510, 26003]) while the model actually emits the
    SINGLE special token `</think>` (with bracket, e.g. 248069). So the `len == 1` guard
    failed, think_end_token became None, the streaming reasoning-state machine never
    engaged, and the ENTIRE chain-of-thought (plus the literal markup) leaked into
    delta.content with reasoning_tokens=0 — defeating the streaming fixes on the
    DEFAULT path for the most common reasoning models. Encode the BRACKETED form with
    add_special_tokens=False (so a BOS-prepending tokenizer doesn't inflate the length).
    """

    def _enc(s: str):
        try:
            return tokenizer.encode(s, add_special_tokens=False)
        except TypeError:
            # Some wrappers don't accept the kwarg — fall back, then strip a leading BOS.
            ids = tokenizer.encode(s)
            bos = getattr(tokenizer, "bos_token_id", None)
            if bos is not None and len(ids) > 1 and ids[0] == bos:
                ids = ids[1:]
            return ids
        except Exception:
            return None

    try:
        _ts = _enc("<think>")
        _te = _enc("</think>")
        if _ts and _te and len(_ts) == 1 and len(_te) == 1:
            return _ts[0], _te[0]
    except Exception:
        logger.debug("thinking token resolve failed", exc_info=True)
    return None, None


@contextmanager
def _wired_limit_ctx(model):
    """Raise wired memory limit during generation to prevent weight swapping.

    Follows mlx-lm's generate.py pattern: set limit to recommended max before
    generation, restore + synchronize after.
    """
    mx = None
    old_limit = None
    try:
        import mlx.core as mx

        max_rec = mx.metal.recommended_max_working_memory_size()
        model_bytes = sum(p.nbytes for p in model.parameters())
        old_limit = mx.set_wired_limit(max_rec) if model_bytes > max_rec * 0.5 else None
    except Exception:
        logger.debug("wired limit setup failed", exc_info=True)
        mx = None
        old_limit = None
    try:
        yield
    finally:
        if old_limit is not None and mx is not None:
            try:
                mx.synchronize()
                mx.set_wired_limit(old_limit)
            except Exception:
                logger.debug("wired limit restore failed", exc_info=True)


@dataclass
class GenerationOutput:
    """Output from generation."""

    text: str = ""
    new_text: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finished: bool = False
    finish_reason: str | None = None
    cached_tokens: int = 0
    logprobs: list[dict] | None = None
    ttft_ms: float = 0.0
    reasoning_tokens: int = 0
    current_state: str | None = None  # "reasoning" or "normal" — matches RequestOutput
    error: str | None = None  # Error message if generation failed
    prefill_progress: tuple[int, int] | None = (
        None  # (processed, total) during chunked prefill
    )
    # True ONLY when a user-supplied stop sequence fired (not natural EOS). finish_reason
    # is "stop" for both, so protocols that must distinguish (Anthropic stop_sequence vs
    # end_turn) read this flag rather than guessing from finish_reason.
    stopped_by_stop_sequence: bool = False
    # Per-prompt-token logprobs (eval/perplexity). vLLM shape — a list
    # aligned to the prompt tokens; element 0 is None (no preceding context).
    prompt_logprobs: list | None = None


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
    upstream path) both fixes 4-bit and makes ``YUNSHU_KV_QUANT_START`` actually
    take effect. ``kv_quant_bits`` is accepted for signature compatibility.
    """
    from mlx_lm.models.cache import make_prompt_cache

    return make_prompt_cache(model)


def _parse_quant_config_env(s: str) -> dict | None:
    """Parse YUNSHU_QUANT_CONFIG into an mlx-lm quantization dict.

    Accepts JSON ({"group_size":64,"bits":4}), a bare int (bits, group_size=64), or a
    compact "bits" / "bits,group_size" / "bits:group_size" string. Returns None if it
    can't be parsed (caller ignores it with a warning rather than crashing the load).
    This is passed to mlx_lm.utils.load via `model_config={"quantization": <dict>}` — NOT
    as a bogus `quantization=` kwarg (which load() doesn't accept → TypeError → dead load).
    """
    if not s:
        return None
    s = s.strip()
    try:
        import json as _json

        parsed = _json.loads(s)
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, bool):
            return None
        if isinstance(parsed, int):
            return {"bits": parsed, "group_size": 64}
    except Exception:
        pass
    parts = [p for p in s.replace(":", ",").split(",") if p.strip()]
    try:
        if len(parts) == 1:
            return {"bits": int(parts[0]), "group_size": 64}
        if len(parts) >= 2:
            return {"bits": int(parts[0]), "group_size": int(parts[1])}
    except ValueError:
        pass
    return None


def _prefill_step_size() -> int:
    """Chunk size for prompt prefill (tokens processed per forward pass).

    mlx-lm's generate_step chunks the prefill at this size so the activation peak
    stays bounded regardless of prompt length (a 200k-token prefill peaks ~12GB on
    a 36GB M3 Max). Default 2048 (mlx-lm's default). Lower it to reduce the prefill
    peak on very memory-constrained hardware; raise it to speed up long prefills
    when memory is ample. Tuned via YUNSHU_PREFILL_STEP_SIZE.
    """
    import os

    try:
        v = int(os.environ.get("YUNSHU_PREFILL_STEP_SIZE", "2048"))
    except (TypeError, ValueError):
        v = 2048
    return v if v > 0 else 2048


def _wrap_custom_logits_processor(proc):
    """SAMP-2: Wrap a user-provided logits processor to adapt its signature.

    User-provided processors follow the vLLM convention:
        (token_ids: list[int], logits: mx.array) -> mx.array

    But mlx-lm's generate_step passes (tokens: mx.array, logits: mx.array).
    This wrapper converts mx.array tokens → list[int] before calling the user processor.
    """

    def _wrapped(tokens_mx, logits):
        token_ids = [int(t) for t in tokens_mx]
        return proc(token_ids, logits)

    return _wrapped


def _apply_spec_bonus_penalties(
    logits: Any,
    all_token_ids: list[int],
    prompt_token_count: int,
    repetition_penalty: float = 1.0,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    logit_bias: dict[int, float] | None = None,
    recent_ctx: int = 20,
) -> Any:
    """Apply penalty and logit_bias processors to bonus token logits.

    For speculative decode paths, penalties are only applied to the BONUS
    token (first new token after acceptance) since draft tokens are already
    committed.

    Args:
        logits: 1-D or 2-D logits array (vocab_size,) or (1, vocab_size).
        all_token_ids: Full token history (prompt + generated).
        prompt_token_count: Number of prompt tokens (for freq/presence penalty).
        repetition_penalty: Multiplier for repeated tokens (>1 = penalize).
        frequency_penalty: Subtractive penalty proportional to token count.
        presence_penalty: Subtractive penalty for any token seen >0 times.
        logit_bias: Additive bias per token ID.
        recent_ctx: How many recent tokens to check for repetition penalty.

    Returns:
        Modified logits array (same shape).
    """
    import mlx.core as _mx

    # Generated tokens only (exclude prompt)
    gen_tokens = (
        all_token_ids[prompt_token_count:]
        if len(all_token_ids) > prompt_token_count
        else []
    )

    if repetition_penalty != 1.0 and len(gen_tokens) > 0:
        recent = gen_tokens[-recent_ctx:]
        unique_recent = list(dict.fromkeys(recent))
        sel = logits[..., unique_recent]
        sel = _mx.where(sel < 0, sel * repetition_penalty, sel / repetition_penalty)
        logits[..., _mx.array(unique_recent)] = sel

    if (frequency_penalty != 0.0 or presence_penalty != 0.0) and len(gen_tokens) > 0:
        counts: dict[int, int] = {}
        for t in gen_tokens:
            counts[t] = counts.get(t, 0) + 1
        for tid, cnt in counts.items():
            # OpenAI permits frequency/presence penalty in [-2.0, 2.0].
            # Negative values BOOST the token (encourage repetition); the
            # earlier code only applied when > 0 which silently dropped
            # negative penalties — confirmed via curl with pp=-2.0 producing
            # identical output to pp=0. Apply for any non-zero value.
            if frequency_penalty != 0.0:
                logits[..., tid] = logits[..., tid] - frequency_penalty * cnt
            if presence_penalty != 0.0 and cnt > 0:
                logits[..., tid] = logits[..., tid] - presence_penalty

    if logit_bias:
        # Guard against out-of-range / negative token ids (see _logit_bias_proc):
        # an invalid user-supplied id would index out of bounds and crash.
        vocab = logits.shape[-1]
        for tid, bias in logit_bias.items():
            if 0 <= tid < vocab:
                logits[..., tid] = logits[..., tid] + bias

    return logits


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
    _maybe_quantize_kv_cache(prompt_cache, quantized_kv_start, kv_group_size, kv_bits)


def _store_thinking_segment(
    ids, thinking_tokens: list[int], thinking_store, kv_cache=None
) -> None:
    """Store a thinking segment KV for future reuse."""
    try:
        import hashlib as _hl

        conv_id = _hl.sha256(str([int(t) for t in ids[:16]]).encode()).hexdigest()[:16]
        # Snapshot to avoid sharing mutable reference with prefix_cache
        _kv_snapshot = [c for c in kv_cache] if kv_cache else None
        thinking_store.store(
            conversation_id=conv_id,
            thinking_tokens=thinking_tokens,
            context_tokens=[int(t) for t in ids],
            kv_data=_kv_snapshot,
        )
    except Exception:
        logger.debug("thinking segment store failed", exc_info=True)


def _build_noncached_sampler_text(
    temperature: float,
    top_p: float,
    top_k: int,
    min_p: float,
    seed: int | None,
    xtc_probability: float = 0.0,
    xtc_threshold: float = 0.0,
):
    """Numpy-backed sampler that bypasses mlx-lm's @mx.compile cache.

    mlx-lm's `categorical_sampling` is wrapped with
    `@mx.compile(inputs=mx.random.state, outputs=mx.random.state)`. The
    compile cache traps the first call's PRNG state, so subsequent
    requests with temperature>0 produce identical token streams even
    after `mx.random.seed()` between them — the cause of the repeated-output
    bug for VLM and the n>1 collapse seen on /v1/chat/completions.

    For temperature>0 use `numpy.random.Generator` (independent RNG per
    request). Always returns an mx.array compatible with mlx-lm's
    generate_step.
    """
    import time as _t

    import numpy as _np

    base = (
        int(seed) & ((1 << 63) - 1)
        if seed is not None
        else _t.time_ns() & ((1 << 63) - 1)
    )
    rng = _np.random.default_rng(base)
    _t_ = float(temperature)
    _tp = float(top_p) if top_p else 1.0
    _tk = int(top_k) if top_k and top_k > 0 else 0
    _mp = float(min_p) if min_p else 0.0
    # XTC (eXclude Top Choices) was silently dropped on this
    # default non-streaming fast path — the request was accepted but produced non-XTC
    # output (the streaming path applied it, so behavior diverged by the `stream` flag).
    _xtc_p = float(xtc_probability) if xtc_probability else 0.0
    _xtc_thr = float(xtc_threshold) if xtc_threshold else 0.0

    def _sampler(logits):
        import mlx.core as _mx

        arr = _np.asarray(logits.astype(_mx.float32))
        flat = arr.reshape(-1, arr.shape[-1])
        out = _np.empty(flat.shape[0], dtype=_np.int64)
        for i in range(flat.shape[0]):
            # Build the surviving-token set on the UN-tempered softmax,
            # then apply temperature LAST (only when forming the final categorical
            # distribution). mlx-lm's make_sampler runs apply_top_p/min_p/xtc/top_k
            # on the raw logprobs and applies temp inside categorical_sampling
            # (`logprobs * (1/temp)`) — temp LAST. The old code divided by temp
            # FIRST, so for temp!=1 + any filter the nucleus/min_p/XTC threshold
            # selected a DIFFERENT token set than the streaming path → same request
            # sampled from a materially different distribution depending on the
            # `stream` flag. An earlier fix only corrected the order AMONG the filters.
            raw = flat[i].astype(_np.float64)
            l = raw - _np.max(raw)
            p = _np.exp(l)
            p = p / p.sum()
            # Apply filters in mlx-lm make_sampler's ORDER — top_p → min_p →
            # XTC → top_k — so this non-streaming temp>0 sampler produces the SAME
            # distribution as the streaming path (which uses make_sampler) for the same
            # params. The old order (top_k first) gave a different surviving token set
            # when both top_k and top_p were set → stream/non-stream diverged.
            if 0 < _tp < 1:
                order = _np.argsort(-p)
                cum = _np.cumsum(p[order])
                # Nucleus = smallest set whose cumulative prob >= top_p, INCLUDING
                # the token that crosses the threshold (matches mlx-lm apply_top_p).
                # `order[cum <= _tp]` dropped the crossing token → nucleus too narrow.
                k = int(_np.searchsorted(cum, _tp, side="left")) + 1
                keep = order[: max(1, min(k, len(order)))]
                m = _np.zeros_like(p)
                m[keep] = 1.0
                p = p * m
                p = p / p.sum()
            if _mp > 0:
                pmax = p.max()
                m = (p >= _mp * pmax).astype(_np.float64)
                p = p * m
                p = p / p.sum()
            # XTC: with probability _xtc_p, remove every token whose prob is above the
            # threshold EXCEPT the least-probable one among them (matches mlx-lm
            # apply_xtc: mask = probs > min(probs where probs > threshold)). Keeps the
            # low-confidence tail, dropping the high-prob cluster → more diverse output.
            if _xtc_p > 0.0 and rng.random() < _xtc_p:
                above = p > _xtc_thr
                if above.any():
                    min_above = p[above].min()
                    remove = p > min_above
                    if remove.any() and not remove.all():
                        p = p * (~remove).astype(_np.float64)
                        _s = p.sum()
                        if _s > 0:
                            p = p / _s
            if _tk and _tk < len(p):
                idx = _np.argpartition(p, -_tk)[-_tk:]
                m = _np.zeros_like(p)
                m[idx] = 1.0
                p = p * m
                p = p / p.sum()
            # Apply temperature LAST, restricted to the survivor set (p>0), exactly
            # like mlx-lm's categorical_sampling(masked_logprobs, temp). Sampling
            # the un-tempered survivor `p` directly would ignore temperature; sample
            # from softmax(raw_logits/temp) over the kept tokens instead.
            keep = p > 0
            lf = _np.where(keep, raw / _t_, -_np.inf)
            lf = lf - _np.max(lf)
            pf = _np.exp(lf)
            _sf = pf.sum()
            pf = pf / _sf if _sf > 0 else p
            out[i] = int(rng.choice(len(pf), p=pf))
        return _mx.array(out.reshape(arr.shape[:-1]).astype(_np.int64))

    return _sampler


def _build_gpu_sampler_text(
    temperature: float,
    top_p: float,
    top_k: int,
    min_p: float,
    seed: int | None,
    xtc_probability: float = 0.0,
    xtc_threshold: float = 0.0,
):
    """On-GPU temp>0 sampler (opt-in via YUNSHU_GPU_SAMPLER=1).

    The default numpy sampler (_build_noncached_sampler_text) does
    `np.asarray(logits)` INSIDE mlx-lm's _step → a forced GPU→CPU sync EVERY
    token, which breaks generate_step's async pipeline (it can no longer overlap
    the next forward with token consumption) and runs a full-vocab numpy softmax
    on the CPU (~1100µs/token on a 152k vocab). This sampler stays on the GPU:

    - Uses mlx-lm's OWN filter functions (apply_top_p/min_p/xtc/top_k) — the exact
      chain the streaming make_sampler uses — so the distribution is identical to
      both the streaming path and the numpy path.
    - Replaces mlx-lm's categorical_sampling (the @mx.compile(inputs=mx.random.state)
      one whose compile-cache traps the PRNG → the n>1 collapse the numpy sampler
      exists to avoid) with Gumbel-max keyed by an EXPLICIT per-request mx.random.key
      that is split per step. Gumbel-max(x) is distributionally identical to
      categorical(softmax(x)), and the explicit key gives per-request
      reproducibility WITHOUT the compile-cache trap.

    Returns an mx.array (no .item()/np sync), preserving the async decode pipeline.
    """
    import time as _t

    import mlx.core as mx
    from mlx_lm.sample_utils import (
        apply_min_p,
        apply_top_k,
        apply_top_p,
    )

    _t_ = float(temperature)
    methods = []
    if top_p and 0 < float(top_p) < 1.0:
        _tp = float(top_p)
        methods.append(lambda x: apply_top_p(x, _tp))
    if min_p and float(min_p) != 0.0:
        _mp = float(min_p)
        methods.append(lambda x: apply_min_p(x, _mp))
    # top_k must be applied AFTER XTC (mlx-lm order: top_p→min_p→XTC→top_k,
    # matching the default numpy sampler). Keeping it in `methods` applied it BEFORE the
    # XTC block below, so XTC computed its cutoff over a top_k-truncated distribution —
    # masking a different token set than mlx-lm and breaking this sampler's stated
    # "identical to the numpy path" invariant. Apply it last instead.
    _top_k_n = int(top_k) if (top_k and int(top_k) > 0) else 0

    # XTC handled INLINE (not via mlx-lm's apply_xtc) because that function is
    # @mx.compile(inputs=mx.random.state) → its compile-cache traps the global
    # PRNG exactly like categorical_sampling does (the n>1-collapse this sampler
    # exists to avoid). We replicate its math but draw the coin from our explicit
    # per-request split key so XTC stays reproducible, on-GPU, and trap-free.
    _xtc_on = bool(xtc_probability and float(xtc_probability) > 0.0)
    _xp, _xt = float(xtc_probability), float(xtc_threshold)
    if _xtc_on and not (0 <= _xt <= 0.5):
        raise ValueError(f"xtc_threshold must be in [0, 0.5], got {_xt}")

    base = (
        int(seed) & ((1 << 63) - 1)
        if seed is not None
        else _t.time_ns() & ((1 << 63) - 1)
    )
    state = {"key": mx.random.key(base)}

    def _sampler(logprobs):
        lp = logprobs
        for m in methods:
            lp = m(lp)
        if _xtc_on:
            # Inline XTC: drop every token whose prob exceeds the smallest prob
            # above the threshold, gated by a per-step coin from the explicit key.
            probs = mx.softmax(lp, axis=-1)
            cutoff = mx.where(probs > _xt, probs, mx.inf).min(axis=-1, keepdims=True)
            mask = probs > cutoff
            state["key"], coin_key = mx.random.split(state["key"])
            coin = mx.random.uniform(0, 1, key=coin_key)
            lp = mx.where(coin > _xp, lp, mx.where(mask, -mx.inf, lp))
        # top_k LAST (after XTC), matching mlx-lm / the numpy sampler. Guard
        # top_k >= vocab — mlx-lm's apply_top_k raises there (numpy no-ops; keep parity).
        if _top_k_n and _top_k_n < lp.shape[-1]:
            lp = apply_top_k(lp, _top_k_n)
        # Temperature LAST (matches categorical_sampling: logprobs * 1/temp), then
        # Gumbel-max with a freshly-split explicit key (no global PRNG state).
        state["key"], sub = mx.random.split(state["key"])
        u = mx.random.uniform(shape=lp.shape, key=sub)
        g = -mx.log(-mx.log(u))
        return mx.argmax(lp * (1.0 / _t_) + g, axis=-1)

    return _sampler


def _build_temp_sampler(
    temperature, top_p, top_k, min_p, seed, xtc_probability=0.0, xtc_threshold=0.0
):
    """Pick the temp>0 sampler. Default = numpy (proven, avoids the
    @mx.compile PRNG trap). YUNSHU_GPU_SAMPLER=1 = on-GPU Gumbel-max (no per-token
    GPU→CPU sync, preserves mlx-lm's async pipeline; same distribution)."""
    import os as _os

    if _os.environ.get("YUNSHU_GPU_SAMPLER", "").strip().lower() in (
        "1",
        "true",
        "yes",
    ):
        return _build_gpu_sampler_text(
            temperature, top_p, top_k, min_p, seed, xtc_probability, xtc_threshold
        )
    return _build_noncached_sampler_text(
        temperature, top_p, top_k, min_p, seed, xtc_probability, xtc_threshold
    )


def _build_constrained_sampler(sampler, json_schema, tokenizer):
    """Build a constrained sampler from a grammar specification.

    Handles:
    - JSON schema dict → JsonSchemaConstraint
    - "json_object" string → generic JSON constraint
    - {"type": "regex", "pattern": "..."} → RegexConstraint
    - {"type": "choice", "choices": [...]} → ChoiceConstraint
    - {"type": "cfg", "grammar": "..."} → LarkGrammarConstraint

    When YUNSHU_GRAMMAR_BITMASK=1 is set, uses the bitmask engine instead
    of the allowlist-based ConstrainedSampler (xgrammar-style approach).
    """
    # Determine grammar type and payload
    grammar_type = None
    grammar = None

    if isinstance(json_schema, dict) and json_schema.get("type") in (
        "regex",
        "choice",
        "cfg",
    ):
        grammar_type = json_schema["type"]
        if grammar_type == "regex":
            grammar = json_schema.get("pattern", "")
        elif grammar_type == "choice":
            grammar = json_schema.get("choices", [])
        elif grammar_type == "cfg":
            grammar = json_schema.get("grammar", "")
        else:
            return sampler
    else:
        grammar_type = "json_schema"
        grammar = json_schema

    # ── Bitmask path (YUNSHU_GRAMMAR_BITMASK=1) ──
    try:
        from .grammar_bitmask import (
            BitmaskConstrainedSampler,
            build_bitmask_engine,
            is_bitmask_enabled,
        )

        if is_bitmask_enabled():
            try:
                engine = build_bitmask_engine(grammar_type, grammar)
                return BitmaskConstrainedSampler(sampler, engine, tokenizer)
            except Exception:
                logger.debug(
                    "bitmask engine setup failed, falling back to allowlist",
                    exc_info=True,
                )
    except ImportError:
        logger.debug(
            "grammar_bitmask module not available, using allowlist path", exc_info=True
        )

    # ── Standard allowlist path ──
    if grammar_type in ("regex", "choice", "cfg"):
        from .grammar_constraint import ConstraintFactory

        try:
            constraint = ConstraintFactory.create(grammar_type, grammar, tokenizer)
            from .json_schema import ConstrainedSampler

            return ConstrainedSampler(sampler, constraint, tokenizer)
        except Exception:
            logger.debug(
                "grammar constraint setup failed, returning unconstrained sampler",
                exc_info=True,
            )
            return sampler

    # Standard JSON schema path
    from .json_schema import ConstrainedSampler, JsonSchemaConstraint

    if isinstance(json_schema, str):
        if json_schema == "json_object":
            # Generic JSON object mode — no specific schema
            constraint = JsonSchemaConstraint(None)
        else:
            import json as _json

            schema = _json.loads(json_schema)
            constraint = JsonSchemaConstraint(schema)
    else:
        constraint = JsonSchemaConstraint(json_schema)
    return ConstrainedSampler(sampler, constraint, tokenizer)


def _build_grammar_constraint(json_schema, tokenizer):
    """Build the RAW constraint object (not a sampler wrapper) from a grammar spec.

    Routes {"type": "regex"|"choice"|"cfg"} to ConstraintFactory and everything else
    (a JSON-schema dict, the "json_object" string, or a JSON-schema string) to
    JsonSchemaConstraint. The speculative decoder drives the constraint directly
    (advance / rollback / get_allowed_tokens), so it needs the bare constraint, not the
    ConstrainedSampler wrapper that _build_constrained_sampler returns.

    The cross-model spec paths previously wrapped EVERY grammar dict in
    JsonSchemaConstraint, so a {"type":"regex"/"choice"/"cfg"} spec was silently turned
    into a JSON-object constraint (_get_type_from_schema → "regex" falls through to the
    default object branch) — the model was forced to emit JSON and the user's regex /
    choice / cfg constraint was dropped with no error. This mirrors the correct routing
    in _build_constrained_sampler.
    """
    if isinstance(json_schema, dict) and json_schema.get("type") in (
        "regex",
        "choice",
        "cfg",
    ):
        gtype = json_schema["type"]
        if gtype == "regex":
            grammar = json_schema.get("pattern", "")
        elif gtype == "choice":
            grammar = json_schema.get("choices", [])
        else:  # cfg
            grammar = json_schema.get("grammar", "")
        from .grammar_constraint import ConstraintFactory

        return ConstraintFactory.create(gtype, grammar, tokenizer)

    from .json_schema import JsonSchemaConstraint

    if isinstance(json_schema, str) and json_schema != "json_object":
        import json as _json

        try:
            return JsonSchemaConstraint(_json.loads(json_schema))
        except Exception:
            return JsonSchemaConstraint(json_schema)
    if json_schema == "json_object":
        return JsonSchemaConstraint(None)
    return JsonSchemaConstraint(json_schema)


def _resolve_model_max_ctx(model) -> int:
    """Resolve a model's context window. mlx-lm Model objects store
    config in `.args` (Qwen/Llama use max_position_embeddings), NOT `.config` /
    a bare max_seq_len — so the old single-attr lookups were dead for most
    models. Check the model, its .config, and its .args for the usual keys.
    Returns 0 when undeterminable (caller treats that as 'no clamp')."""
    for src in (model, getattr(model, "config", None), getattr(model, "args", None)):
        if src is None:
            continue
        for attr in ("max_position_embeddings", "max_seq_len", "n_positions"):
            v = getattr(src, attr, None)
            if isinstance(v, int) and v > 0:
                return v
    return 0


class BatchedEngine:
    """User-facing continuous batching engine.

    Wraps EngineCore with:
    - Lazy model loading
    - Chat template preprocessing
    - SamplingParams construction
    - GeneratorExit-safe cleanup
    - Special token cleaning
    - Speculative decoding (Phase 4: single-request EAGLE-3 path)
    """

    def __init__(
        self,
        model_name: str = "",
        stream_interval: int = 1,
        enable_thinking: bool | None = None,
    ) -> None:
        self.model_name = model_name
        self.stream_interval = stream_interval
        self.enable_thinking = enable_thinking
        # LoRA concurrency keystone: this engine acquires/applies + releases/
        # restores the LoRA adapter ITSELF, inside its executor closures (serialized with
        # generation). The gateway must therefore NOT apply on the event loop (that was the
        # cross-thread race). VLM/legacy engines lack this flag → gateway applies for them.
        self._self_manages_lora = True
        # Friendly identifier used for Prometheus labels, audit, and any
        # outward-facing display. ModelManager passes the full filesystem
        # path as `model_name` so it can be fed to mlx_lm.load_model;
        # using that raw path as a label leaks absolute paths into
        # Prometheus.
        # Derive a short basename for label use without disturbing
        # callers that still rely on `model_name` for the loader.
        import os as _os

        self.model_label = (
            (_os.path.basename(model_name.rstrip("/")) if model_name else "")
            or model_name
            or "default"
        )

        self._model = None
        self._tokenizer = None
        self._engine_core = None
        self._loaded = False
        self._starting = False  # Guard against concurrent start() calls
        self._kv_manager = None  # Set from EngineCore._kv_manager after start

        # Fast-path active request tracking (prevents model eviction mid-generation)
        self._active_fast_path_count = 0
        self._fast_path_lock = threading.Lock()

        # ── Wired production modules ──
        # Model preprocessor registry (auto-detects model family for multimodal input)
        from .model_preprocessor import PreprocessorRegistry

        self._preprocessor_registry = PreprocessorRegistry()

        # Speculative decoding state (Phase 4)
        self._spec_decoder = None  # SpeculativeDecoder instance
        self._spec_enabled = False

        # Gemma-4 dual-load assistant drafter (EAGLE-style external drafter that
        # shares the target's KV; validated 2.08×). Gated by YUNSHU_GEMMA4_ASSISTANT.
        self._gemma4_assistant_proposer = None  # Gemma4AssistantProposer

        # MTP speculative decoding (built-in multi-token prediction heads)
        self._mtp_decoder = None  # MTPDecoder instance
        self._mtp_strategy = None  # MTPStrategy wrapper

        # N-gram proposer for model-free speculative decoding
        self._ngram_proposer = None  # NgramProposer, created on demand
        # Route greedy requests through the (lossless) n-gram spec path by default,
        # not only when spec_decode=true. Set in _init_spec_decode from env.
        self._ngram_greedy_default = False
        self._ngram_stats = {"proposals": 0, "accepted": 0, "total_draft": 0}

        # Response cache hit/miss counters (YUNSHU_RESPONSE_CACHE=1)
        self._response_cache_hits = 0
        self._response_cache_misses = 0

        # GPU-accelerated rejection sampling (opt-in via YUNSHU_GPU_REJECTION=1)
        from .gpu_rejection import GPURejectionSampler, should_enable_gpu_rejection

        self._gpu_rejection_sampler = GPURejectionSampler()
        self._gpu_rejection_enabled = should_enable_gpu_rejection()

        # Spec draft verifier: production-grade draft verification with
        # KV cache trimming and bonus token emission
        from .spec_draft_verifier import SpecDraftVerifier

        self._spec_draft_verifier = SpecDraftVerifier(track_stats=True)

        # Adaptive speculative decode controller (opt-in via YUNSHU_ADAPTIVE_SPEC=1)
        self._adaptive_spec = None

        # Lookahead reasoning: boosts spec decode during <think/> blocks
        from .speculative_decoder import LookaheadReasoning

        self._lookahead_reasoning = LookaheadReasoning()

        # Medusa speculative decoding (multi-head prediction on hidden state)
        self._medusa_proposer = None  # MedusaProposer instance
        self._medusa_strategy = None  # MedusaStrategy wrapper

        # SpecPrefill config (opt-in via YUNSHU_SPEC_PREFILL env var)
        self._spec_prefill_enabled = False
        self._spec_prefill_threshold = 8192
        self._spec_prefill_keep_rate = 0.20
        self._spec_prefill_draft_model = None

        # KV prefix cache for multi-turn speedup
        # Enable the WARM tier — 128 cached prefixes (32 full-
        # precision HOT + up to 96 4-bit-quantized WARM) fit in roughly the
        # memory of the old 64 full entries on UMA. Tunable via env for bench.
        import os as _os

        from .kv_prefix_cache import KVPrefixCache

        _pc_max = int(_os.environ.get("YUNSHU_PREFIX_MAX_ENTRIES", "128"))
        _pc_hot = int(_os.environ.get("YUNSHU_PREFIX_HOT_LIMIT", "32"))
        self._kv_prefix_cache = KVPrefixCache(
            max_entries=_pc_max, hot_limit=_pc_hot, min_prefix_length=32
        )
        # HYBRID-model prefix reuse on the fast path. Hybrid
        # models (Qwen3.5: KVCache + ArraysCache) normally bypass the prefix
        # cache because trimming the recurrent ArraysCache state corrupts it.
        # With this on, the fast path instead chunk-prefills and stores trim=0
        # block-boundary snapshots, which reuse losslessly (verified). Opt-in
        # while it proves out; the bypass remains the fallback on any error.
        # Default ON: only affects HYBRID models — standard models are
        # trimmable and never enter this path. block=128 is the knee of the
        # reuse/cold-penalty curve (≈3.1x reuse for ≈+18% cold prefill);
        # YUNSHU_HYBRID_PREFIX_BLOCK=64 buys more reuse for a steeper cold cost,
        # 256 the reverse. Set YUNSHU_HYBRID_PREFIX=0 to fall back to bypass.
        self._hybrid_prefix_enabled = _os.environ.get(
            "YUNSHU_HYBRID_PREFIX", "1"
        ).strip() in ("1", "true", "yes")
        self._hybrid_prefix_block = int(
            _os.environ.get("YUNSHU_HYBRID_PREFIX_BLOCK", "128")
        )
        # Opt-in MTP (YUNSHU_MTP=1). mlx-vlm is the supported MTP
        # implementation (its speculative path is the only correct one — see mlxvlm_mtp.py).
        # HONESTY: the often-quoted "1.82x lossless" is from a STANDALONE PROOF
        # script (bench_mtp_vlm_27b.py, self-marked "INTEGRATION TODO") — it is NOT a served,
        # regression-gated number, and the wired path here is single-backend + honors only
        # temperature (drops top_p/json_schema/penalties, see _warn_mtp_dropped_params) +
        # non-streaming. Treat it as EXPERIMENTAL, not a shipped prod win; the real spec win
        # is the gemma-4 assistant drafter. When set and the model has native MTP weights,
        # the engine serves via that backend (single-backend swap, no dual-load): greedy →
        # MTP; sampling → plain gen on the same model. (YUNSHU_MLXVLM_MTP kept as an alias.)
        self._mlxvlm_mtp_enabled = _os.environ.get("YUNSHU_MTP", "").strip() in (
            "1",
            "true",
            "yes",
        ) or _os.environ.get("YUNSHU_MLXVLM_MTP", "0").strip() in ("1", "true", "yes")
        self._mlxvlm_mtp = None

        # Warm prompt prefill stats (tracked across _warm_prompt_prefill calls)
        self._warm_prompt_stats: dict = {
            "prompts_loaded": 0,
            "prompts_prefilled": 0,
            "prompts_skipped_cached": 0,
            "prompts_failed": 0,
            "total_tokens_prefilled": 0,
            "prefill_time_s": 0.0,
            "source": "none",
        }

        # Prompt cache for exact-match KV state reuse (supplements KVPrefixCache)
        # When the same prompt text is submitted multiple times, the prompt cache
        # returns the full KV state directly — zero prefill compute.
        from .prompt_cache import PromptCacheManager

        self._prompt_cache = PromptCacheManager(
            max_entries=256,
            max_memory_mb=512.0,
            ttl_seconds=3600.0,
        )

        # SSD KV cache persistence (opt-in via YUNSHU_SSD_CACHE=1)
        import os

        if os.environ.get("YUNSHU_SSD_CACHE", "").strip() in ("1", "true", "yes"):
            ssd_dir = os.environ.get("YUNSHU_SSD_CACHE_DIR", "~/.cache/yunshu/kv-ssd")
            _ssd_raw = os.environ.get("YUNSHU_SSD_CACHE_MAX_GB", "10")
            try:
                _ssd_float = float(_ssd_raw)
            except (TypeError, ValueError):
                logger.warning(
                    "YUNSHU_SSD_CACHE_MAX_GB invalid value %r, using default 10",
                    _ssd_raw,
                )
                _ssd_float = 10.0
            ssd_max_gb = int(_ssd_float)
            if _ssd_float != ssd_max_gb:
                logger.warning(
                    "YUNSHU_SSD_CACHE_MAX_GB=%r truncated to %d GB (fractional GB not supported)",
                    _ssd_raw,
                    ssd_max_gb,
                )
            self._kv_prefix_cache.enable_ssd_cache(
                cache_dir=ssd_dir,
                max_size_bytes=ssd_max_gb * 1024**3,
                model_name=model_name,
            )

        # KV cache quantization config (mlx-lm pattern: to_quantized)
        # Enable via YUNSHU_KV_QUANT_BITS=4 or 8 (MLX only supports these values)
        _qbits = os.environ.get("YUNSHU_KV_QUANT_BITS")
        if _qbits:
            _qbits_int = int(_qbits)
            if _qbits_int not in (2, 3, 4, 8):
                logger.warning(
                    "CONFIG: YUNSHU_KV_QUANT_BITS=%s is not a supported value "
                    "(MLX supports 2, 3, 4, 8). Ignoring.",
                    _qbits,
                )
                self._kv_quant_bits: int | None = None
            else:
                self._kv_quant_bits = _qbits_int
        else:
            self._kv_quant_bits = None
        self._kv_quant_group_size: int = int(
            os.environ.get("YUNSHU_KV_QUANT_GROUP_SIZE", "64")
        )
        self._kv_quant_start: int = int(os.environ.get("YUNSHU_KV_QUANT_START", "0"))

        # Memory pressure eviction config (vllm-mlx pattern)
        # Stored as a percentage (0-100) for consistency with
        # KVPrefixCache.evict_under_pressure(). Converted to a 0-1
        # fraction when calling KVCacheManager.memory_pressure_evict().
        self._mem_pressure_threshold = float(
            os.environ.get("YUNSHU_MEM_PRESSURE_THRESHOLD", "85.0")
        )
        # If the value looks like a fraction (<=1.0), convert to percentage
        if 0 < self._mem_pressure_threshold <= 1.0:
            self._mem_pressure_threshold *= 100.0

        # DeltaNet state inversion for KV cache eviction recovery
        # Enable via YUNSHU_DELTANET_INVERSION=1 — when KV blocks are evicted
        # from the prefix cache, analytically invert the SSM recurrence so
        # the evicted context can be partially recovered (75x less overhead
        # than checkpoint/restore). Works with GatedDeltaNet models (Qwen3.5).
        self._deltanet_inverter = None
        self._deltanet_inversion_enabled = os.environ.get(
            "YUNSHU_DELTANET_INVERSION", ""
        ).strip() in ("1", "true", "yes")
        self._deltanet_inversion_stats = {
            "evictions_captured": 0,
            "inversions_attempted": 0,
            "inversions_succeeded": 0,
            "states_stored": 0,
        }

        # Per-model settings (loaded from model_settings.json + env vars)
        self._settings = None

        # Thinking Segment KV Substore — reasoning token KV cache reuse
        # Enable via YUNSHU_THINKING_CACHE=1
        self._thinking_store = None
        self._total_reasoning_tokens = 0
        if os.environ.get("YUNSHU_THINKING_CACHE", "").strip() in ("1", "true", "yes"):
            from yunshu_kv.thinking_segment import (
                ThinkingSegmentConfig,
                ThinkingSegmentSubstore,
            )

            _thinking_cfg = ThinkingSegmentConfig(
                max_segments_per_conversation=int(
                    os.environ.get("YUNSHU_THINKING_MAX_PER_CONV", "10")
                ),
                max_total_segments=int(
                    os.environ.get("YUNSHU_THINKING_MAX_TOTAL", "1000")
                ),
                min_tokens_to_cache=int(
                    os.environ.get("YUNSHU_THINKING_MIN_TOKENS", "32")
                ),
                ttl_seconds=float(os.environ.get("YUNSHU_THINKING_TTL", "3600")),
                enable_ssd=os.environ.get("YUNSHU_THINKING_SSD", "").strip()
                in ("1", "true", "yes"),
                ssd_cache_dir=os.environ.get("YUNSHU_THINKING_SSD_DIR", ""),
                enable_compression=os.environ.get(
                    "YUNSHU_THINKING_COMPRESS", ""
                ).strip()
                in ("1", "true", "yes"),
            )
            self._thinking_store = ThinkingSegmentSubstore(_thinking_cfg)

        # LoRA adapter manager
        self._lora_manager = None

        # mx.compile() for Metal kernel caching
        self._compiled = False
        _mx_compile_env = os.environ.get("YUNSHU_MX_COMPILE", "").strip().lower()
        if _mx_compile_env in ("1", "true", "yes"):
            self._use_compile = True
        elif _mx_compile_env in ("0", "false", "no"):
            self._use_compile = False
        else:
            # Auto-enable mx.compile on Apple Silicon — it's always safe
            self._use_compile = platform.system() == "Darwin"

        # The hand-written Metal kernels were REMOVED. They were never
        # invoked in the served path and, benchmarked head-to-head on this M3 Max,
        # were uniformly SLOWER than Apple's MLX ops (fp16 gemv 1.07-1.19x, q4 gemv
        # 1.41-2.23x, sdpa was just mx.einsum) — mx.fast.*/mx.matmul/
        # mx.quantized_matmul are already hand-tuned for Apple Silicon. Production
        # attention/matmul stays on mx.fast.
        self._metal_kernel_manager = None
        self._metal_kernels_enabled = False

        # Engine loop default (continuous batching mode)
        # Enable via YUNSHU_ENGINE_LOOP=1 for multi-user concurrent serving
        self._engine_loop_default = os.environ.get(
            "YUNSHU_ENGINE_LOOP", ""
        ).strip() in ("1", "true", "yes")

        # Streaming optimizer pipeline components (C18 pattern)
        # Enable via YUNSHU_STREAMING_PIPELINE=1 for pipelined GPU/CPU overlap
        self._streaming_pipeline_enabled = os.environ.get(
            "YUNSHU_STREAMING_PIPELINE", ""
        ).strip() in ("1", "true", "yes")

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def config(self):
        """Engine config — delegates to EngineCoreConfig when available.

        This property exists so that the admin ``/config/engine`` endpoint
        (which reads ``engine.config``) works with both the legacy ``Engine``
        class (which stores config directly) and ``BatchedEngine`` (which
        keeps the config inside ``EngineCore``).
        """
        if self._engine_core is not None:
            return self._engine_core.config
        return None

    @property
    def is_running(self) -> bool:
        """True when the engine core loop is active (continuous batching).

        Used by ``_ensure_engine_started`` in the gateway engine module
        to decide whether ``start()`` needs to be called.  The legacy
        Engine class has the same property; BatchedEngine was missing it,
        which caused an ``AttributeError`` in single-engine fallback mode.
        """
        return self._loaded and (
            self._engine_core is not None and self._engine_core.is_running
        )

    async def start(self) -> None:
        """Load model and start EngineCore."""
        if self._loaded:
            return

        # Guard against concurrent start() calls from multiple coroutines.
        # Without this, two concurrent generate() calls that both see
        # _loaded=False can race and both load the model simultaneously,
        # doubling memory usage and causing model ref leaks.
        if getattr(self, "_starting", False):
            # Another coroutine is already starting — wait for it
            import asyncio

            while getattr(self, "_starting", False):
                await asyncio.sleep(0.05)
            if self._loaded:
                return
            # If the other starter failed, we need to start ourselves
        self._starting = True

        try:
            await self._do_start()
        finally:
            self._starting = False

    async def _do_start(self) -> None:
        """Internal start implementation (called under _starting guard)."""
        from .mlx_executor import get_mlx_executor

        executor = get_mlx_executor()
        loop = asyncio.get_running_loop()

        # mlx-vlm native MTP single-backend mode. When enabled
        # and the checkpoint has native MTP weights, load the mlx-vlm target +
        # MTP drafter on the executor thread and skip the mlx-lm fast-path setup
        # entirely (no dual-load). chat() delegates to this backend.
        if self._mlxvlm_mtp_enabled:
            try:
                from .mlxvlm_mtp import MLXVLMMtp, is_mtp_capable

                if is_mtp_capable(self.model_name):
                    backend = MLXVLMMtp(self.model_name)
                    await loop.run_in_executor(executor, backend.load)
                    self._mlxvlm_mtp = backend
                    self._tokenizer = backend.tokenizer
                    _cache_tokenizer_vocab(self._tokenizer)  # missed this twin
                    self._loaded = True
                    logger.info("mlx-vlm MTP backend active for %s", self.model_name)
                    return
                logger.info(
                    "YUNSHU_MTP set but %s is not MTP-capable; using standard path",
                    self.model_name,
                )
            except Exception:
                logger.warning(
                    "mlx-vlm MTP backend load failed; falling back to standard path",
                    exc_info=True,
                )

        # Load model on MLX executor thread (non-blocking)
        def _load():
            from mlx_lm.utils import load as load_model

            kwargs = {}
            # Check for quantization override from env or settings.
            # The old code did kwargs["quantization"] = <str> and passed it to
            # mlx_lm.utils.load — which has NO `quantization` parameter (its params are
            # path/tokenizer_config/model_config/adapter_path/lazy/return_config/revision),
            # so setting YUNSHU_QUANT_CONFIG raised TypeError and FAILED the whole model
            # load (the except below only caught ValueError). The mlx-lm contract is to
            # override the model's config.json quantization via `model_config`. Parse the
            # env as JSON ({"group_size":64,"bits":4}) or a compact "bits" / "bits,group".
            qconfig = os.environ.get("YUNSHU_QUANT_CONFIG")
            if qconfig:
                _qd = _parse_quant_config_env(qconfig)
                if _qd:
                    kwargs["model_config"] = {
                        **kwargs.get("model_config", {}),
                        "quantization": _qd,
                    }
                else:
                    logger.warning(
                        "YUNSHU_QUANT_CONFIG=%r not parseable as JSON or bits[,group_size]; ignored",
                        qconfig,
                    )
            try:
                return load_model(self.model_name, **kwargs)
            except TypeError as e:
                # An override kwarg the installed mlx-lm doesn't accept must NOT
                # fail the whole engine — retry a clean load without overrides.
                logger.warning(
                    "load() rejected an override kwarg (%s); retrying without overrides",
                    str(e)[:120],
                )
                return load_model(self.model_name)
            except ValueError as e:
                # Some checkpoints ship weights the model class intentionally
                # omits — e.g. Gemma-4's KV-shared layers store vestigial
                # k/v/k_norm the architecture never uses → "Received N
                # parameters not in model". These EXTRA weights are safe to
                # drop. Retry strict=False ONLY for the extra-params case;
                # a "missing parameters" error must still hard-fail.
                if "not in model" not in str(e):
                    raise
                logger.warning(
                    "Checkpoint has extra params not in the model class; "
                    "retrying load with strict=False (%s)",
                    str(e)[:100],
                )
                from pathlib import Path as _Path

                from mlx_lm.utils import (
                    load_model as _load_strict_false,
                )
                from mlx_lm.utils import (
                    load_tokenizer as _load_tok,
                )

                mp = _Path(self.model_name)
                if not mp.exists():
                    from mlx_lm.utils import hf_repo_to_path

                    mp = hf_repo_to_path(self.model_name)
                ret = _load_strict_false(mp, strict=False)
                model = ret[0] if isinstance(ret, tuple) else ret
                return model, _load_tok(mp)

        self._model, self._tokenizer = await loop.run_in_executor(executor, _load)
        # On-the-fly weight quantization at load (YUNSHU_QUANT_MODE =
        # mxfp4 / nvfp4 / mxfp8 / affine). Lets a bf16/fp16 checkpoint be quantized
        # IN-MEMORY at load — smaller + faster decode on bandwidth-bound hardware
        # (4-bit weights ≈ 2.9x over bf16; mxfp4's per-block microscaling can beat
        # affine-4bit on quality-per-bit). Runs on the MLX executor. Already-quantized
        # layers are skipped by the predicate (no to_quantized), so it's safe to set
        # even on a pre-quantized model (no-op there).
        _qmode = os.environ.get("YUNSHU_QUANT_MODE", "").strip().lower()
        if _qmode in ("mxfp4", "nvfp4", "mxfp8", "affine"):
            await loop.run_in_executor(executor, self._quantize_on_load, _qmode)
        _cache_tokenizer_vocab(self._tokenizer)

        try:
            await self._finish_start(loop, executor)
            self._loaded = True
        except Exception:
            # Partial init: clean up model that was loaded but subsystems failed
            logger.error(
                "BatchedEngine start failed after model load, cleaning up",
                exc_info=True,
            )
            await self.stop()
            raise

    async def _finish_start(self, loop, executor) -> None:
        """Complete engine initialization after model loading.

        Split from start() so that if this phase fails, the already-loaded
        model can be properly cleaned up via stop().
        """
        # Apply model-specific patches (DeepSeek MLA, Qwen 3.5 YARN, Gemma softcap)
        try:
            from .model_patches import apply_model_patches

            patches = apply_model_patches(self._model, self._tokenizer, self.model_name)
            if patches:
                logger.info(f"Model patches applied: {patches}")
        except Exception:
            logger.warning("Model patches skipped", exc_info=True)

        # Detect model architecture optimizations (RoPE scaling, attention type, MoE)
        try:
            from .model_optimizations import (
                AttentionOptimizer,
                MoEEfficiencyOptimizer,
                RoPEScalingOptimizer,
            )

            rope_opt = RoPEScalingOptimizer()
            target_ctx = getattr(self._model, "max_seq_len", None)
            if target_ctx is None:
                config = getattr(self._model, "config", None) or getattr(
                    self._model, "args", None
                )
                target_ctx = (
                    getattr(config, "max_position_embeddings", 4096) if config else 4096
                )
            rope_opt.configure(self._model, target_context_length=target_ctx)
            attn_opt = AttentionOptimizer()
            attn_opt.detect_attention_type(self._model)
            moe_opt = MoEEfficiencyOptimizer()
            # Auto-detect MoE config from model
            moe_num_experts = 0
            moe_top_k = 0
            if hasattr(self._model, "config"):
                cfg = self._model.config
                moe_num_experts = getattr(
                    cfg, "num_experts", getattr(cfg, "num_local_experts", 0)
                )
                moe_top_k = getattr(
                    cfg, "num_experts_per_tok", getattr(cfg, "num_selected_experts", 0)
                )
            if moe_num_experts > 0 and moe_top_k > 0:
                moe_opt.configure(self._model, moe_num_experts, moe_top_k)
            logger.info(
                f"Model optimizations detected: RoPE={rope_opt.get_scaling_config().scaling_type}, "
                f"Attention={attn_opt.get_stats().get('attention_type', 'unknown')}, "
                f"MoE={moe_opt.get_stats().get('num_experts', 0)} experts"
            )
        except Exception:
            logger.warning("Model optimization detection skipped", exc_info=True)

        # The home-grown Qwen3.5 MTP (n_confirmed_patch +
        # mtp_patch + mtp_decoder) is DEPRECATED. It reimplemented what mlx-vlm
        # already does correctly, but lacked mlx-vlm's GatedDeltaNet
        # intermediate-state capture, so it produced garbage on 27B and only ~0.9x
        # on 9B. The supported MTP path is now mlx-vlm (YUNSHU_MTP=1, see
        # mlxvlm_mtp.py — ~1.82x in a proof script only, not served/gated; EXPERIMENTAL).
        # The legacy patches are only applied under YUNSHU_LEGACY_MTP=1 (escape hatch).
        if os.environ.get("YUNSHU_LEGACY_MTP", "0").strip() in ("1", "true", "yes"):
            try:
                from .n_confirmed_patch import apply_n_confirmed_patch

                if apply_n_confirmed_patch():
                    logger.info("[legacy] n_confirmed patch applied")
            except Exception:
                logger.debug("n_confirmed patch skipped", exc_info=True)
            try:
                from .mtp_patch import apply_mtp_patch

                if apply_mtp_patch():
                    logger.info("[legacy] MTP patch applied")
            except Exception:
                logger.warning("MTP patch skipped", exc_info=True)

        # Load per-model settings from model_settings.json + env overrides
        self._load_model_settings()

        # Detect model's cache types for type-aware KV management
        try:
            from mlx_lm.models.cache import make_prompt_cache

            from yunshu_kv.model_cache_config import ModelCacheConfig

            test_cache = make_prompt_cache(self._model)
            self._cache_config = ModelCacheConfig.build_from_cache(test_cache)
            logger.info(
                f"Cache config: {self._cache_config.num_layers} layers, "
                f"{self._cache_config.sliceable_count} sliceable"
            )
        except Exception as e:
            logger.debug(f"Cache type detection skipped: {e}")
            self._cache_config = None

        # Initialize DeltaNet inversion for KV eviction recovery
        # Registers capture hooks on SSM layers and wires the pre-eviction
        # callback into the KV prefix cache so evicted states are inverted.
        if self._deltanet_inversion_enabled and self._model is not None:
            self._init_deltanet_inversion()

        # Warmup: ModelWarmupManager handles compile caching + KV prefill
        # The basic generate_step warmup is wrapped inside _model_warmup()
        def _warmup():
            import mlx.core as mx
            from mlx_lm.generate import generate_step
            from mlx_lm.sample_utils import make_sampler

            ids = mx.array(self._tokenizer.encode("Hi"))
            sampler = make_sampler(temp=0.0)
            for _ in generate_step(ids, self._model, max_tokens=1, sampler=sampler):
                break
            mx.synchronize()
            mx.clear_cache()

        def _model_warmup():
            """Full warmup using ModelWarmupManager (compile + KV cache prefill)."""
            from .model_optimizations import ModelWarmupManager

            mgr = ModelWarmupManager()
            model_type = "generic"
            name_lower = self.model_name.lower()
            for family in ("qwen", "llama", "deepseek", "gemma"):
                if family in name_lower:
                    model_type = family
                    break
            use_compile = self._use_compile and not self._compiled
            result = mgr.warmup(self._model, model_type=model_type, compile=use_compile)
            if not result.output_valid:
                raise RuntimeError(
                    f"Model warmup produced invalid outputs for {self.model_name} "
                    f"— model weights or config may be corrupted"
                )
            logger.info(
                f"Model warmup: {result.warmup_time_s:.3f}s, "
                f"compile={result.compile_cached}, prompts={result.prompts_warmed}"
            )
            # Also do basic warmup to ensure MX compile cache is populated
            _warmup()

        await loop.run_in_executor(executor, _model_warmup)

        # The previous block called `mx.compile(self._model)`
        # and DISCARDED the return value. mx.compile returns a NEW compiled
        # callable; it does not mutate the model in place — so this compiled
        # nothing (the model was still invoked via its original __call__) while
        # logging a misleading "Model compiled — Metal kernels cached". There is
        # no CUDA-graph equivalent on Apple Silicon, and mlx-lm already applies
        # @mx.compile internally to the hot sampling/RoPE paths; whole-model
        # compile also recompiles on every prefill shape change (ragged batching),
        # which can hurt. Removed the dead attempt — no engine-level whole-model
        # compile is performed. (ModelWarmupManager's own compile path above is
        # unchanged; the _use_compile/_compiled flags remain for back-compat.)

        # Initialize speculative decoding if model supports it (Phase 4)
        self._init_spec_decode()

        # Initialize LoRA adapter manager
        self._init_lora()

        # Warm prompt prefill: pre-populate KV cache with common system prompts
        await self._warm_prompt_prefill()

        # Auto-start EngineCore for continuous batching (default production path)
        # Previously was lazy-loaded only when use_engine_loop=True.
        # Now always starts so the scheduler is ready for concurrent requests.
        use_fast_only = os.environ.get("YUNSHU_FAST_PATH_ONLY", "").strip() in (
            "1",
            "true",
            "yes",
        )
        if not use_fast_only:
            try:
                await self._ensure_engine_core()
                logger.info("EngineCore auto-started — continuous batching ready")
            except Exception as e:
                logger.warning(
                    f"EngineCore auto-start failed ({e}), falling back to fast-path only"
                )
                self._engine_core = None

        # KV-5: Apply auto-tuner KV quantization recommendation if available
        self._apply_auto_tuner_kv_quant()

        # Metal kernel init removed (kernels deleted — slower than mx.fast).

    def _init_deltanet_inversion(self) -> None:
        """Initialize DeltaNet state inversion for KV cache eviction recovery.

        Creates a DeltaNetInverter, registers capture hooks on the model's
        SSM layers, and wires a pre-eviction callback into KVPrefixCache.
        When KV blocks are evicted under memory pressure, the callback
        analytically inverts the SSM recurrence to recover the pre-eviction
        state — 75x less overhead than checkpoint/restore.

        Only activates for models with SSM layers (GatedDeltaNet, e.g. Qwen3.5).
        Controlled via YUNSHU_DELTANET_INVERSION=1 env var.
        """
        try:
            from .deltanet_inversion import DeltaNetInverter

            self._deltanet_inverter = DeltaNetInverter()

            # Register capture hooks on any SSM layers the model has
            has_ssm = any(hasattr(m, "state") for _, m in self._model.named_modules())
            if has_ssm:
                self._deltanet_inverter.register_hooks(self._model)
                logger.info(
                    "DeltaNet inversion hooks registered — SSM eviction recovery enabled"
                )
            else:
                logger.info(
                    "DeltaNet inversion enabled but no SSM layers found — "
                    "inversion will run only on explicit invert_evicted_state() calls"
                )

            # Wire pre-eviction callback into KV prefix cache
            # Use a weak reference to avoid a reference cycle:
            # engine -> _kv_prefix_cache -> _pre_evict_callback -> engine
            if self._kv_prefix_cache is not None:
                import weakref

                _weak_self = weakref.ref(self)

                def _on_evict(prompt_tokens, cache):
                    strong = _weak_self()
                    if strong is not None:
                        strong._on_prefix_cache_eviction(prompt_tokens, cache)

                self._kv_prefix_cache._pre_evict_callback = _on_evict
                logger.info("DeltaNet eviction callback wired into KV prefix cache")

        except Exception as e:
            logger.warning(
                f"DeltaNet inversion init failed ({e}), continuing without SSM recovery"
            )
            self._deltanet_inverter = None

    def _on_prefix_cache_eviction(self, _prompt_tokens, cache) -> None:
        """Pre-eviction callback: capture DeltaNet state before KV cache is dropped.

        Called by KVPrefixCache._remove_entry() when an entry is evicted.
        Scans the cache layers for SSM state tensors and captures their
        inverted form so the evicted context can be partially recovered.
        """
        if self._deltanet_inverter is None:
            return
        self._deltanet_inversion_stats["evictions_captured"] += 1

        # Attempt to extract and invert SSM states from cache layers.
        # Standard KVCache layers are skipped — only SSM (DeltaNet) layers
        # with a .state attribute are candidates for inversion.
        if not isinstance(cache, list):
            return

        for layer in cache:
            state = getattr(layer, "state", None)
            if state is None:
                continue
            # If the inverter has captured entries (from model hooks),
            # invert them to recover pre-step state.
            try:
                self._deltanet_inversion_stats["inversions_attempted"] += 1
                self._deltanet_inverter.start_capture()
                recovered = self._deltanet_inverter.invert_all()
                if recovered:
                    self._deltanet_inversion_stats["inversions_succeeded"] += 1
                    self._deltanet_inversion_stats["states_stored"] += len(recovered)
                    logger.debug(
                        f"DeltaNet inversion: recovered {len(recovered)} layer states "
                        f"from evicted cache"
                    )
            except Exception:
                logger.debug(
                    "DeltaNet inversion failed for evicted layer", exc_info=True
                )

    def invert_evicted_state(self, _prompt_tokens: list[int] | None = None) -> list:
        """Trigger DeltaNet state inversion for evicted or current SSM states.

        Manually triggers inversion of captured SSM intermediates. Useful for
        testing or for recovering state after programmatic eviction.

        Args:
            prompt_tokens: Optional prompt tokens to identify which cache entry
                to invert. If None, inverts the last captured state.

        Returns:
            List of recovered mx.array states (one per SSM layer), or empty
            list if inversion is disabled or no state is available.
        """
        if self._deltanet_inverter is None:
            logger.debug(
                "DeltaNet inversion not enabled — invert_evicted_state() is no-op"
            )
            return []

        try:
            self._deltanet_inversion_stats["inversions_attempted"] += 1
            results = self._deltanet_inverter.invert_all()
            if results:
                self._deltanet_inversion_stats["inversions_succeeded"] += 1
                self._deltanet_inversion_stats["states_stored"] += len(results)
            return results
        except Exception:
            logger.debug("DeltaNet invert_evicted_state failed", exc_info=True)
            return []

    def _load_model_settings(self):
        """Load per-model settings from model directory and apply to engine."""
        from .model_settings import load_model_settings

        model_path = ""
        if self._model is not None:
            config = getattr(self._model, "config", None)
            if config is not None:
                if isinstance(config, dict):
                    model_path = config.get("_name_or_path", self.model_name)
                else:
                    model_path = getattr(config, "_name_or_path", self.model_name)
        self._settings = load_model_settings(
            model_path or self.model_name, self.model_name
        )
        self._apply_settings()

    def _apply_settings(self):
        """Apply loaded ModelSettings to engine config."""
        if self._settings is None:
            return
        s = self._settings
        if s.kv_cache_quant_bits is not None:
            # Validate the bits BEFORE assigning, mirroring the
            # YUNSHU_KV_QUANT_BITS env path above. model_settings.json (and the
            # YUNSHU_MODEL_{ID}_KV_CACHE_QUANT_BITS override) flowed through here
            # WITHOUT validation, so an invalid value (e.g. 5) reached
            # to_quantized(bits=5) and crashed every request that hit the quant
            # threshold. MLX only supports 2/3/4/8.
            if s.kv_cache_quant_bits in (2, 3, 4, 8):
                self._kv_quant_bits = s.kv_cache_quant_bits
            else:
                logger.warning(
                    "CONFIG: model_settings kv_cache_quant_bits=%s is not a "
                    "supported value (MLX supports 2, 3, 4, 8). Ignoring.",
                    s.kv_cache_quant_bits,
                )
        if s.kv_cache_quant_group_size != 64:
            self._kv_quant_group_size = s.kv_cache_quant_group_size
        # NOTE: kv_cache_quant_start_layer is a LAYER INDEX — turbo_quant
        # uses it for per-layer fp16/int8/int4 tiering (turbo_quant.py:178-188). It
        # must NOT be assigned to self._kv_quant_start, which is mlx-lm's
        # quantized_kv_start: a TOKEN-POSITION threshold compared against cache.offset.
        # Conflating them made "quantize from layer N" silently mean "quantize once the
        # sequence reaches N tokens". _kv_quant_start stays env-driven
        # (YUNSHU_KV_QUANT_START); the layer field is consumed by turbo_quant only.
        if not s.prefix_cache_enabled:
            self._kv_prefix_cache = None
        if s.spec_decode_enabled:
            self._spec_enabled = True
        if s.spec_prefill_enabled:
            self._spec_prefill_enabled = True
            self._spec_prefill_threshold = s.spec_prefill_threshold
            self._spec_prefill_keep_rate = s.spec_prefill_keep_rate
        if s.ssd_cache_enabled and self._kv_prefix_cache is not None:
            self._kv_prefix_cache.enable_ssd_cache(
                cache_dir=s.ssd_cache_dir,
                max_size_bytes=s.ssd_cache_max_gb * 1024**3,
                model_name=self.model_name,
            )
        if s.enable_thinking is not None:
            self.enable_thinking = s.enable_thinking
        if s.moe_top_k > 0 and self._model is not None:
            from .moe_optimization import apply_moe_top_k

            result = apply_moe_top_k(self._model, s.moe_top_k)
            if result["patched_layers"] > 0:
                logger.info(f"MoE top-k applied: {result}")

    def get_settings(self):
        return self._settings

    def _jump_forward_generate_sync(
        self, input_ids, constraint, max_tokens, eos_ids, stop=None
    ):
        """Jump-forward decoding (opt-in YUNSHU_JUMP_FORWARD=1) for
        JSON-schema/grammar-constrained generation. Runs on the MLX executor.

        Custom constrained decode loop (mlx-lm's generate_step is a black box that
        can't skip forwards). Each iteration runs ONE model forward over
        [sampled_token] + [the structural tokens the FSM FORCES next] — so the
        forced JSON structure (':', ',', '}', forced key names, 'true'/'false'/
        'null') is emitted WITHOUT a per-token forward. Forward count = number of
        BRANCH points, not total tokens → a big decode-latency cut for structured
        output on bandwidth-bound Apple Silicon. Greedy (argmax of the
        constraint-masked logits) for deterministic, schema-valid output.

        Returns (text, generated_token_ids, n_forwards, stop_hit). Constraint
        guarantees the TEXT is schema-valid; the forced run's exact token boundaries
        may differ from natural tokenization (no backward token-healing), which only
        affects the model's free-value continuation, never the forced structure.
        stop_hit is True iff generation terminated because a `stop` substring appeared
        (so the caller can report finish_reason="stop" and an accurate token count).
        """
        import mlx.core as mx
        from mlx_lm.models.cache import make_prompt_cache

        from .json_schema import apply_json_constraint

        model, tok = self._model, self._tokenizer
        eos = set(eos_ids or [])
        stops = [s for s in (stop or []) if s]

        def _first_stop(text: str) -> int:
            # earliest index at which any stop substring begins, or -1
            best = -1
            for s in stops:
                p = text.find(s)
                if p != -1 and (best == -1 or p < best):
                    best = p
            return best

        cache = make_prompt_cache(model)
        logits = model(mx.array([input_ids]), cache=cache)[:, -1, :]
        gen_ids: list[int] = []
        pieces: list[str] = []
        acc = ""  # running detokenized text, for incremental stop scanning
        stop_hit = False
        n_fwd = 1
        max_tokens = int(max_tokens)
        while len(gen_ids) < max_tokens:
            allowed = constraint.get_allowed_tokens(tok, gen_ids)
            if not allowed:
                # FSM is in a terminal/DONE state with no further legal tokens. If
                # eos is empty (e.g. ignore_eos) there is nothing left to sample —
                # stop instead of looping forever on an all-masked distribution.
                if not eos:
                    break
            masked = apply_json_constraint(logits, allowed if allowed else list(eos))
            tid = int(mx.argmax(masked, axis=-1).item())
            if tid in eos:
                break
            gen_ids.append(tid)
            ttext = tok.decode([tid]) if tok else ""
            pieces.append(ttext)
            acc += ttext
            # Detect the stop substring INCREMENTALLY (like _generate_fast),
            # so we break at the token that completes it instead of running to
            # max_tokens and only truncating `text` post-hoc — which left gen_ids
            # over-counted (wrong completion_tokens) and finish_reason="length".
            if stops and _first_stop(acc) != -1:
                stop_hit = True
                break
            try:
                constraint.advance(ttext)
            except Exception:
                logger.debug(
                    "jump-forward: constraint.advance(token) failed", exc_info=True
                )
                break
            batch = [tid]
            # Jump-forward: emit the grammar-forced continuation in this same forward.
            try:
                fc = constraint.forced_continuation()
            except Exception:
                fc = ""
            if fc and len(gen_ids) < max_tokens:
                fids = tok.encode(fc, add_special_tokens=False) if tok else []
                # Don't overshoot max_tokens: a long forced continuation could push
                # gen_ids past the budget in one jump. Truncate to the remaining room.
                room = max_tokens - len(gen_ids)
                truncated = len(fids) > room
                if truncated:
                    fids = fids[:room]
                if fids:
                    batch = [tid] + fids
                    gen_ids.extend(fids)
                    # On truncation the emitted text must reflect the tokens we
                    # actually fed, not the full forced string (which won't be
                    # finished). Decode the kept tokens.
                    femit = fc if not truncated else (tok.decode(fids) if tok else "")
                    pieces.append(femit)
                    acc += femit
                    try:
                        # Advance the FSM only by the text we
                        # actually emitted — advancing by the full `fc` on truncation
                        # desynced the constraint from gen_ids.
                        constraint.advance(femit)
                    except Exception:
                        logger.debug(
                            "jump-forward: advance(forced) failed", exc_info=True
                        )
                    if stops and _first_stop(acc) != -1:
                        stop_hit = True
                        break
                    if truncated:
                        # gen_ids is now at max_tokens; the loop is about to exit.
                        # Skip the otherwise-wasted final forward (its logits would
                        # be discarded).
                        break
            out = model(mx.array([batch]), cache=cache)
            logits = out[:, -1, :]
            mx.eval(logits)
            n_fwd += 1
        text = "".join(pieces)
        if stops:
            p = _first_stop(text)
            if p != -1:
                text = text[:p]
                stop_hit = True
        return text, gen_ids, n_fwd, stop_hit

    def _quantize_on_load(self, mode: str) -> None:
        """Quantize the loaded model's weights in-memory (mxfp4/nvfp4/
        mxfp8/affine) via mlx-lm's native quantization. Runs on the MLX executor.
        Skips layers without to_quantized or whose width isn't group-aligned (the
        same predicate mlx-lm's quantize_model uses) — so already-quantized models
        are a safe no-op."""
        try:
            import mlx.nn as nn
        except Exception:
            return
        defaults = {
            "affine": (64, 4),
            "mxfp4": (32, 4),
            "nvfp4": (16, 4),
            "mxfp8": (32, 8),
        }
        group_size, bits = defaults.get(mode, (64, 4))

        def _pred(path, module):
            if not hasattr(module, "to_quantized"):
                return False
            w = getattr(module, "weight", None)
            return w is not None and w.shape[-1] % group_size == 0

        try:
            nn.quantize(self._model, group_size, bits, mode=mode, class_predicate=_pred)
            logger.info(
                "On-the-fly quantization applied: mode=%s group_size=%d bits=%d (%s)",
                mode,
                group_size,
                bits,
                self.model_name,
            )
        except Exception as e:
            logger.error(
                "On-the-fly quantization (mode=%s) failed (%s); serving the model "
                "as loaded.",
                mode,
                e,
                exc_info=True,
            )

    def _kv_bytes_per_token(self) -> int:
        """Estimate the model's bf16 KV-cache bytes per token (K+V across all
        layers): 2 (K,V) * num_layers * num_kv_heads * head_dim * 2 bytes.
        Returns 0 if the dims can't be read."""
        model = getattr(self, "_model", None)
        if model is None:
            return 0
        cfg = getattr(model, "config", None) or getattr(model, "args", None)
        if cfg is None:
            return 0
        # Some models nest text config (VLM/omni).
        text_cfg = getattr(cfg, "text_config", None)

        def _g(names):
            for obj in (text_cfg, cfg):
                if obj is None:
                    continue
                for n in names:
                    v = getattr(obj, n, None)
                    if v:
                        return int(v)
            return 0

        n_layers = _g(("num_hidden_layers", "n_layers", "num_layers"))
        n_kv = _g(("num_key_value_heads", "n_kv_heads", "num_kv_heads")) or _g(
            ("num_attention_heads", "n_heads", "num_heads")
        )
        hidden = _g(("hidden_size", "d_model", "n_embd"))
        n_heads = _g(("num_attention_heads", "n_heads", "num_heads")) or 1
        head_dim = _g(("head_dim",)) or (hidden // n_heads if hidden else 0)
        if not (n_layers and n_kv and head_dim):
            return 0
        return 2 * n_layers * n_kv * head_dim * 2  # K+V, bf16

    def _effective_kv_quant_bits(self, total_tokens: int) -> int | None:
        """KV-quant bits for THIS request.

        Explicit config (env YUNSHU_KV_QUANT_BITS / per-model settings) always
        wins. Otherwise apply a SIZE-gated default: 8-bit (near-lossless) once the
        estimated KV cache is large enough that it dominates decode bandwidth.

        An earlier version gated on TOKEN COUNT (>=8192), but a real-model benchmark
        (scripts/bench/bench_kv_quant_longctx.py) showed that's WRONG for small
        models: Qwen2.5-0.5B at 9001 tokens has only ~108MB of KV (~12KB/token)
        vs ~300MB of 4-bit weights, so KV doesn't dominate and quantization is
        NET-NEGATIVE (0.98x — the dequant overhead exceeds the saved KV bandwidth).
        8-bit KV only wins when KV bytes are a large fraction of the per-step read,
        which depends on layers*kv_heads*head_dim*tokens, NOT tokens alone. So gate
        on estimated total KV bytes (default >=2GB, big-model/very-long-context
        regime). Opt out with YUNSHU_KV_QUANT_AUTO=0; tune via
        YUNSHU_KV_QUANT_AUTO_MIN_BYTES. Falls back to the old token threshold only
        when model dims can't be read."""
        if self._kv_quant_bits is not None:
            return self._kv_quant_bits
        if os.environ.get("YUNSHU_KV_QUANT_AUTO", "1").strip().lower() in (
            "0",
            "false",
            "no",
        ):
            return None
        per_tok = self._kv_bytes_per_token()
        if per_tok > 0:
            try:
                min_bytes = int(
                    os.environ.get("YUNSHU_KV_QUANT_AUTO_MIN_BYTES", str(2 * 1024**3))
                )
            except ValueError:
                min_bytes = 2 * 1024**3
            return 8 if (per_tok * total_tokens) >= min_bytes else None
        # Dims unreadable — fall back to the conservative token threshold.
        try:
            threshold = int(os.environ.get("YUNSHU_KV_QUANT_AUTO_THRESHOLD", "16384"))
        except ValueError:
            threshold = 16384
        return 8 if total_tokens >= threshold else None

    def _apply_auto_tuner_kv_quant(self) -> None:
        """KV-5: Apply auto-tuner's kv_quantization_bits recommendation.

        Checks the AutoTuner's current params for KV quantization recommendation
        and updates the engine's _kv_quant_bits if the auto-tuner suggests
        quantization that isn't already configured. This enables adaptive KV
        quantization based on observed memory pressure.

        Priority: env var > model settings > auto_tuner recommendation.
        """
        # Only apply if KV quantization isn't already explicitly configured
        if self._kv_quant_bits is not None:
            return  # Already configured via env var or model settings

        # Check EngineCore's auto-tuner if available
        if self._engine_core is not None:
            auto_tuner = getattr(self._engine_core, "_auto_tuner", None)
            if auto_tuner is not None:
                recommended_bits = auto_tuner.params.kv_quantization_bits
                if recommended_bits < 16:  # 16 means no quantization
                    self._kv_quant_bits = recommended_bits
                    logger.info(
                        f"KV-5: Auto-tuner recommends {recommended_bits}-bit KV quantization"
                    )

    def _init_lora(self):
        """Initialize LoRA adapter manager after model load."""
        max_loras = int(os.environ.get("YUNSHU_MAX_LORAS", "4"))
        from .lora_manager import LoRAAdapterManager, set_lora_manager

        self._lora_manager = LoRAAdapterManager(max_loras=max_loras)
        self._lora_manager.set_base_model(self._model)
        # Wire LoRA memory tracking into model manager budget. The singleton
        # accessor lives in the gateway layer (not yunshu_engine.model_manager);
        # this is a best-effort runtime hook, so the lazy import here is fine and
        # fails gracefully when the engine runs standalone (no gateway).
        try:
            from yunshu_gateway.engine import get_model_manager

            _mm = get_model_manager()
            if _mm is not None:
                self._lora_manager.set_memory_callback(_mm.track_lora_memory)
        except Exception as e:
            # WARNING (not DEBUG) so silent budget-tracking failures are
            # surfaced — LoRA memory accounting won't update against the
            # model-manager budget if this wiring is skipped.
            logger.warning(
                "LoRA memory callback wiring failed (model-manager budget "
                "will not track LoRA memory): %s",
                e,
                exc_info=True,
            )
        set_lora_manager(self._lora_manager, engine_id=self.model_name or "default")

        # Auto-discover adapters in model directory
        model_path = ""
        config = getattr(self._model, "config", None)
        if config is not None:
            if isinstance(config, dict):
                model_path = config.get("_name_or_path", "")
            else:
                model_path = getattr(config, "_name_or_path", "")
        if model_path:
            discovered = self._lora_manager.discover_adapters(model_path)
            if discovered:
                logger.info(f"Discovered {len(discovered)} LoRA adapters: {discovered}")

    def get_lora_manager(self):
        return self._lora_manager

    async def _ensure_engine_core(self):
        """Lazy-create EngineCore only when continuous batching is needed."""
        if self._engine_core is not None:
            return
        from .engine_core import EngineCore, EngineCoreConfig
        from .mlx_executor import get_mlx_executor

        executor = get_mlx_executor()
        arch_kwargs = self._extract_model_arch(self._model)

        # Sarathi-style hybrid chunked prefill (opt-in via YUNSHU_HYBRID_PREFILL=1)
        hybrid_prefill = os.environ.get("YUNSHU_HYBRID_PREFILL", "").strip() in (
            "1",
            "true",
            "yes",
        )
        hybrid_chunk = int(os.environ.get("YUNSHU_HYBRID_CHUNK_SIZE", "512"))

        prefill_chunk_size = int(os.environ.get("YUNSHU_PREFILL_CHUNK_SIZE", "2048"))

        self._engine_core = EngineCore(
            model=self._model,
            tokenizer=self._tokenizer,
            config=EngineCoreConfig(
                stream_interval=self.stream_interval,
                enable_hybrid_prefill=hybrid_prefill,
                hybrid_chunk_size=hybrid_chunk,
                prefill_chunk_size=prefill_chunk_size,
                # Decode-batch width is env-tunable, but the
                # default of 32 is near-optimal on Apple-Silicon UMA and should
                # rarely be raised. Decode is memory-bandwidth-bound, so
                # aggregate throughput SATURATES around batch ~32; measured on
                # Qwen3.5-2B, raising it to 64 made N=96/128 WORSE (130 then a
                # crash) because a wider decode batch only adds KV memory
                # pressure without more throughput. The N>=32 throughput plateau
                # is the hardware ceiling, not a tunable software limit.
                completion_batch_size=int(
                    __import__("os").environ.get("YUNSHU_COMPLETION_BATCH_SIZE", "32")
                ),
                # Scheduling policy is now plumbed through (was hardwired
                # FCFS → PRIORITY/FAIR preemption + aging were unreachable). Opt in
                # with YUNSHU_SCHEDULER_POLICY=priority|fair (engine-loop only).
                scheduler_policy=__import__("os")
                .environ.get("YUNSHU_SCHEDULER_POLICY", "fcfs")
                .lower(),
                aging_weight=float(
                    __import__("os").environ.get("YUNSHU_SCHEDULER_AGING_WEIGHT", "0.1")
                ),
                **arch_kwargs,
            ),
            executor=executor,
        )
        self._engine_core.scheduler.config.model_name = self.model_name

        # Propagate paged KV manager for memory pressure eviction in fast paths
        self._kv_manager = self._engine_core._kv_manager

        # Setup memory guard with model dimensions
        try:
            model_cfg = (
                getattr(self._model, "config", self._model) if self._model else None
            )
            if model_cfg is not None:
                num_layers = getattr(model_cfg, "num_hidden_layers", 0)
                num_kv_heads = getattr(model_cfg, "num_key_value_heads", 0)
                head_dim = getattr(model_cfg, "hidden_size", 0) // max(
                    getattr(model_cfg, "num_attention_heads", 1), 1
                )
                num_attn_heads = getattr(model_cfg, "num_attention_heads", None)
                if num_layers and num_kv_heads and head_dim:
                    self._engine_core.setup_memory_guard(
                        num_layers=num_layers,
                        num_kv_heads=num_kv_heads,
                        head_dim=head_dim,
                        num_attention_heads=num_attn_heads,
                    )
        except Exception:
            logger.warning("MemoryGuard setup skipped", exc_info=True)

        # Setup TurboQuant per-layer mixed-precision KV quantization
        try:
            model_cfg = (
                getattr(self._model, "config", self._model) if self._model else None
            )
            if model_cfg is not None:
                num_layers = getattr(model_cfg, "num_hidden_layers", 0)
                if num_layers > 0:
                    quant_bits = getattr(model_cfg, "kv_cache_quant_bits", None)
                    quant_start = int(
                        os.environ.get(
                            "YUNSHU_TURBOQUANT_START_LAYER",
                            getattr(model_cfg, "kv_cache_quant_start_layer", 0),
                        )
                    )
                    quant_group = int(
                        os.environ.get(
                            "YUNSHU_TURBOQUANT_GROUP_SIZE",
                            getattr(model_cfg, "kv_cache_quant_group_size", 64),
                        )
                    )
                    # Enable if model has quant settings or env opt-in
                    env_enable = os.environ.get("YUNSHU_TURBOQUANT", "").strip() in (
                        "1",
                        "true",
                        "yes",
                    )
                    if quant_bits is not None or env_enable:
                        self._engine_core.setup_turbo_quant(
                            total_layers=num_layers,
                            kv_quant_bits=quant_bits,
                            kv_quant_start_layer=quant_start,
                            kv_quant_group_size=quant_group,
                        )
        except Exception:
            logger.debug("TurboQuant setup skipped", exc_info=True)

        # Setup HybridKVCache layer type registration for Mamba/hybrid models
        try:
            model_cfg = (
                getattr(self._model, "config", self._model) if self._model else None
            )
            if model_cfg is not None and self._model is not None:
                num_layers = getattr(model_cfg, "num_hidden_layers", 0)
                if num_layers > 0:
                    self._engine_core.setup_hybrid_kv(
                        model=self._model,
                        total_layers=num_layers,
                    )
        except Exception:
            logger.debug("HybridKVCache setup skipped", exc_info=True)

        # Wire prefix cache into scheduler for batch-path insert_segments (C16)
        self._engine_core.set_prefix_cache(self._kv_prefix_cache)
        # Wire HybridKVCache into scheduler for layer-type-aware KV routing
        if self._engine_core._hybrid_kv is not None:
            self._engine_core.scheduler.set_hybrid_kv_cache(
                self._engine_core._hybrid_kv
            )
        # Wire speculative decoding decoders into the scheduler
        self._wire_spec_decoders_to_scheduler()
        await self._engine_core.start()

        logger.info(f"BatchedEngine started: {self.model_name}")

    def _wire_spec_decoders_to_scheduler(self) -> None:
        """Wire speculative decoding decoders into the scheduler's batch path.

        Called after _ensure_engine_core() creates the scheduler and after
        _init_spec_decode() has detected spec heads and created decoders.

        Three paths:
          1. Cross-model: SpeculativeDecoder → scheduler.set_spec_decoder()
          2. MTP: MTPDecoder → scheduler.set_mtp_decoder()
          3. N-gram: already handled by SchedulerConfig.ngram_spec_enabled

        Also propagates N-gram proposer settings to the scheduler config
        if BatchedEngine has one but the scheduler doesn't.
        """
        if self._engine_core is None:
            return
        scheduler = self._engine_core.scheduler

        # Path 1: Cross-model speculative decoder (EAGLE-3 / external draft)
        if self._spec_decoder is not None:
            scheduler.set_spec_decoder(self._spec_decoder)
            logger.info("Wired cross-model spec decoder into scheduler batch path")

        # Path 2: MTP decoder (built-in multi-token prediction heads)
        if self._mtp_decoder is not None:
            scheduler.set_mtp_decoder(self._mtp_decoder)
            logger.info("Wired MTP decoder into scheduler batch path")

        # Path 3: Propagate N-gram proposer to scheduler if needed
        # (BatchedEngine creates its own N-gram proposer, but the scheduler
        # may not have one if ngram_spec_enabled was False in EngineCoreConfig)
        if self._ngram_proposer is not None and scheduler._ngram_proposer is None:
            scheduler.enable_ngram_spec(
                min_n=self._ngram_proposer.config.min_n,
                max_n=self._ngram_proposer.config.max_n,
                k=self._ngram_proposer.config.k,
                mode=self._ngram_proposer.config.mode,
            )
            logger.info("Wired N-gram proposer into scheduler batch path")

    async def stop(self) -> None:
        """Stop engine and release resources.

        Idempotent: safe to call multiple times or before start().
        Order of shutdown:
        1. EngineCore (stops engine loop, drains in-flight requests)
        2. Subsystem cleanup (KV caches, spec decoder, LoRA, Metal kernels)
        3. Model/tokenizer release + GC + MLX cache clear
        """
        if not self._loaded and self._engine_core is None:
            return  # Never started or already stopped

        # 1. Stop EngineCore (continuous batching loop, drain in-flight)
        if self._engine_core is not None:
            await self._engine_core.stop()
            self._engine_core = None

        # 2. Release KV prefix cache (holds MLX array refs) and flush SSD tier
        if self._kv_prefix_cache is not None:
            self._kv_prefix_cache.close()
            self._kv_prefix_cache.clear()
            self._kv_prefix_cache = None

        # Prompt cache: clear on shutdown (in-memory only; for cross-restart
        # persistence use SSD cache via YUNSHU_SSD_CACHE=1)
        if self._prompt_cache is not None:
            self._prompt_cache.invalidate_all()
            self._prompt_cache = None

        # Speculative decoding subsystems
        self._spec_decoder = None
        self._gemma4_assistant_proposer = None
        self._ngram_proposer = None
        self._adaptive_spec = None
        self._mtp_decoder = None
        self._mtp_strategy = None
        self._medusa_proposer = None
        self._medusa_strategy = None
        self._warm_prompts = None
        self._thinking_store = None
        self._metal_kernel_manager = None

        # Unregister DeltaNet inversion hooks to restore original class methods
        if self._deltanet_inverter is not None:
            self._deltanet_inverter.unregister_hooks()
            self._deltanet_inverter = None

        # Stop LoRA adapter manager (unload all adapters, release weights)
        if self._lora_manager is not None:
            try:
                self._lora_manager.shutdown()
            except Exception:
                logger.debug("LoRA manager cleanup failed", exc_info=True)
            self._lora_manager = None
            from .lora_manager import set_lora_manager

            set_lora_manager(None, engine_id=self.model_name or "default")

        # 3. Release model + tokenizer refs, then GC + clear MLX cache
        self._model = None
        self._tokenizer = None
        self._loaded = False
        self._compiled = False

        import gc

        gc.collect()

        # Per-engine cleanup: synchronize GPU work and clear the MLX cache.
        # Do NOT call shutdown_mlx_executor() — it destroys the GLOBAL
        # singleton ThreadPoolExecutor shared by all engines (BatchedEngine,
        # VLM, Video, OCR, etc.). The global executor should only be shut
        # down at process teardown, not per-engine.
        from .mlx_executor import get_mlx_executor

        loop = asyncio.get_running_loop()
        import mlx.core as mx

        def _cleanup():
            mx.synchronize()
            mx.clear_cache()

        try:
            await loop.run_in_executor(get_mlx_executor(), _cleanup)
        except RuntimeError:
            logger.debug("MLX executor cleanup skipped (executor already shut down)")
        logger.info(f"BatchedEngine stopped: {self.model_name}")

    def _should_use_engine_loop(self, use_engine_loop: bool | None) -> bool:
        """Determine whether to route through EngineCore continuous batching.

        Auto-detects: if EngineCore has active requests, prefer the batch
        path for better throughput under concurrency. Single-request fast
        path is preferred for latency when no other requests are pending.
        """
        if use_engine_loop is not None:
            return use_engine_loop
        # If a fast-path request is already in-flight, do NOT route to the
        # engine loop — the BatchGenerator shares the model + generation_stream
        # with the fast path on the single max_workers=1 MLX executor, and
        # overlapping insert()/next() with a fast-path generate_step corrupts
        # BatchGenerator._currently_processing vs _prompt_batch (IndexError in
        # mlx_lm.generate._next). This guard must run BEFORE the forced-mode
        # check so YUNSHU_ENGINE_LOOP=1 cannot bypass it.
        if getattr(self, "_active_fast_path_count", 0) > 0:
            return False
        # Forced continuous-batching mode (YUNSHU_ENGINE_LOOP=1): route to the
        # engine loop whenever no fast-path request occupies the executor.
        if getattr(self, "_engine_loop_default", False):
            return True
        # Auto-detect: switch to batch path when concurrency is detected
        return bool(
            self._engine_core is not None and self._engine_core.has_active_requests
        )

    # ── Non-generative tasks: embeddings, pooling ────────────────────────────

    def _resolve_embedding_pooling(self) -> str:
        """The model's intended sentence-embedding pooling: MEAN | CLS | LAST.

        embed() hardcoded MEAN, but CLS-pooled models (the BGE
        family — BAAI/bge-*) are trained to use the CLS token; mean-pooling them yields
        vectors in the wrong pooling space (degraded retrieval). Detect via the
        sentence-transformers `1_Pooling/config.json` shipped in the model dir. Defaults
        to MEAN whenever the dir/config is absent or ambiguous → ZERO regression for the
        mean-pooled models (E5/Nomic/GTE) already served. Cached per engine.
        """
        cached = getattr(self, "_embed_pooling", None)
        if cached is not None:
            return cached
        pooling = "MEAN"
        try:
            import json as _json
            import os as _os

            cand = self.model_name if isinstance(self.model_name, str) else ""
            # Resolve a HF repo id (e.g. "BAAI/bge-base-en-v1.5") to its local
            # snapshot before looking for 1_Pooling/config.json. The old code joined the RAW
            # name → "BAAI/bge-base-en-v1.5/1_Pooling/config.json", a relative path that never
            # exists on disk → isfile False → every repo-id-served model fell through to MEAN.
            # That silently MEAN-pooled CLS-trained BGE models (the exact bug this fix
            # targets), only working if the operator passed an explicit local dir. Mirror the
            # engine's own load-path resolution (line 1203-1206).
            if cand and not _os.path.isdir(cand):
                try:
                    from mlx_lm.utils import hf_repo_to_path

                    cand = str(hf_repo_to_path(self.model_name))
                except Exception:
                    pass
            pcfg = _os.path.join(cand, "1_Pooling", "config.json")
            if cand and _os.path.isfile(pcfg):
                with open(pcfg) as _f:
                    d = _json.load(_f)
                if d.get("pooling_mode_cls_token"):
                    pooling = "CLS"
                elif d.get("pooling_mode_lasttoken"):
                    pooling = "LAST"
        except Exception:
            pooling = "MEAN"
        self._embed_pooling = pooling
        return pooling

    def embed(self, texts: list[str], normalize: bool = True) -> list[list[float]]:
        """Generate embeddings for the given texts.

        Uses the model's transformer backbone (before LM head) to extract hidden states,
        applies the model's intended pooling (MEAN/CLS/LAST — see
        _resolve_embedding_pooling), and optionally L2-normalizes the result (default:
        True, matching the OpenAI embeddings API contract).

        Runs synchronously — callers in async contexts should wrap with
        ``await loop.run_in_executor(get_mlx_executor(), engine.embed, texts)``.
        """
        if not self._loaded or self._model is None or self._tokenizer is None:
            raise RuntimeError("Engine not loaded — call start() first")

        import mlx.core as mx

        # Resolve the transformer backbone (before LM head projection).
        backbone = self._get_backbone()
        _pooling = self._resolve_embedding_pooling()

        embeddings: list[list[float]] = []
        for text in texts:
            tokens = self._tokenizer.encode(text)
            if not tokens:
                hidden_size = self._get_hidden_size()
                embeddings.append([0.0] * hidden_size)
                continue

            input_ids = mx.array([tokens])

            if backbone is not None:
                hidden = backbone(input_ids)
            else:
                output = self._model(input_ids)
                hidden = self._extract_hidden_states(output)

            # Pool over the sequence per the model's intended convention.
            if _pooling == "CLS":
                pooled = hidden[:, 0, :].squeeze(0)
            elif _pooling == "LAST":
                pooled = hidden[:, -1, :].squeeze(0)
            else:
                pooled = mx.mean(hidden, axis=1).squeeze(0)

            if normalize:
                norm = mx.sqrt(mx.sum(pooled * pooled) + 1e-12)
                pooled = pooled / norm

            embeddings.append(pooled.tolist())

        return embeddings

    def _compute_prompt_logprobs_sync(
        self, input_ids: list[int], top_k: int = 0
    ) -> list:
        """Compute per-prompt-token logprobs (eval/perplexity).

        Runs ONE forward over the prompt (separate from generation — does not
        touch the decode hot path) and returns a vLLM-shaped list aligned to the
        prompt tokens: element i is the logprob the model assigned to the actual
        token at position i given positions <i. Element 0 is None (no context).
        Each non-null element: {"token_id", "logprob", optional "top_logprobs":
        [{"token_id","logprob"}...]}. top_k=0 → only the realized token.

        Caller runs this on the MLX executor. Length-capped by the caller.
        """
        import mlx.core as mx

        if not input_ids or len(input_ids) < 2:
            return [None] * len(input_ids)
        ids = mx.array([input_ids])
        out = self._model(ids)
        logits = out[0] if not isinstance(out, mx.array) else out
        # logits: [1, seq, vocab] → drop batch dim
        logits = logits[0] if logits.ndim == 3 else logits
        logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        result: list = [None]  # position 0 has no preceding context
        n = len(input_ids)
        k = int(top_k) if top_k and top_k > 0 else 0
        # position i predicts token at i+1: use row i for the realized token i+1.
        for i in range(n - 1):
            tgt = int(input_ids[i + 1])
            row = logp[i]
            entry: dict = {"token_id": tgt, "logprob": float(row[tgt].item())}
            if k > 0:
                topk_idx = mx.argpartition(-row, kth=min(k, row.shape[-1] - 1))[:k]
                topk = sorted(
                    ((int(t), float(row[int(t)].item())) for t in topk_idx.tolist()),
                    key=lambda x: -x[1],
                )
                entry["top_logprobs"] = [
                    {"token_id": t, "logprob": lp} for t, lp in topk
                ]
            result.append(entry)
        return result

    async def _compute_prompt_logprobs_for(
        self,
        prompt,
        enable_thinking,
        top_k: int,
    ) -> list | None:
        """Async wrapper: derive input_ids from the prompt (templating chat
        messages) and compute prompt logprobs on the MLX executor. Length-gated
        to bound the [seq, vocab] logits memory."""
        import os

        cap = int(os.environ.get("YUNSHU_PROMPT_LOGPROBS_MAX_TOKENS", "8192"))
        if isinstance(prompt, list):
            text = self._apply_chat_template(prompt, enable_thinking=enable_thinking)
        else:
            text = prompt
        input_ids = self._encode_prompt(self._tokenizer, text)
        if not input_ids:
            return None
        if len(input_ids) > cap:
            logger.warning(
                "prompt_logprobs skipped: prompt %d tokens exceeds cap %d "
                "(set YUNSHU_PROMPT_LOGPROBS_MAX_TOKENS to raise)",
                len(input_ids),
                cap,
            )
            return None
        import asyncio as _asyncio

        from .mlx_executor import get_mlx_executor

        loop = _asyncio.get_running_loop()
        return await loop.run_in_executor(
            get_mlx_executor(),
            lambda: self._compute_prompt_logprobs_sync(input_ids, top_k),
        )

    def pool(self, texts: list[str], pooling_type: str = "MEAN") -> list[list[float]]:
        """Extract pooled hidden states for the given texts.

        Args:
            texts: List of input strings.
            pooling_type: One of "MEAN", "CLS", "LAST".

        Returns:
            List of pooled hidden-state vectors (NOT normalized).
        """
        if not self._loaded or self._model is None or self._tokenizer is None:
            raise RuntimeError("Engine not loaded — call start() first")

        import mlx.core as mx

        # Pool over the transformer's HIDDEN states (before the LM head), matching
        # embed() and the OpenAI/vLLM pooling contract. Using self._model() here
        # would pool the vocab-space LOGITS instead — a different (huge) dim that
        # can't be compared with embed()'s output. See _get_backbone().
        backbone = self._get_backbone()

        results: list[list[float]] = []
        for text in texts:
            tokens = self._tokenizer.encode(text)
            if not tokens:
                hidden_size = self._get_hidden_size()
                results.append([0.0] * hidden_size)
                continue

            input_ids = mx.array([tokens])
            if backbone is not None:
                hidden = backbone(input_ids)
            else:
                output = self._model(input_ids)
                hidden = self._extract_hidden_states(output)

            if pooling_type.upper() == "CLS":
                pooled = hidden[0, 0, :]  # batch=0, first token
            elif pooling_type.upper() == "LAST":
                pooled = hidden[0, -1, :]  # batch=0, last token
            else:  # MEAN
                pooled = mx.mean(hidden, axis=1).squeeze(
                    0
                )  # [1, seq, d] -> [1, d] -> [d]

            results.append(pooled.tolist())

        return results

    def _extract_hidden_states(self, output) -> Any:
        """Extract hidden states from various model output formats.

        MLX models return either:
        - A plain mx.array (the logits or hidden states)
        - A tuple/list where the first element is hidden states
        - An object with .last_hidden_state attribute
        """
        import mlx.core as mx

        if isinstance(output, mx.array):
            hidden = output  # [batch, seq_len, hidden_size] — OR vocab-space logits
        elif isinstance(output, (tuple, list)):
            hidden = output[0]
        elif hasattr(output, "last_hidden_state"):
            return output.last_hidden_state
        else:
            # Fallback: try subscript access, otherwise return as-is
            try:
                hidden = output[0]
            except (TypeError, IndexError):
                return output
        # This is only reached when _get_backbone() returned None, so we ran
        # the FULL model and `hidden` may be vocab-space LOGITS, not hidden states.
        # Pooling those gives a wrong-dimensioned, meaningless embedding. Warn loudly
        # (once) when the last dim doesn't match the model's hidden_size — the router
        # fallback already has this awareness; the engine path was silent.
        try:
            _hs = self._get_hidden_size()
            if (
                getattr(hidden, "ndim", 0) >= 1
                and _hs
                and hidden.shape[-1] != _hs
                and not getattr(self, "_warned_logits_pool", False)
            ):
                logger.warning(
                    "embed/pool: no transformer backbone resolved for %s — pooling raw "
                    "model output of dim %d (expected hidden_size %d). This is likely "
                    "vocab-space LOGITS → embeddings will be wrong-dimensioned and "
                    "semantically meaningless. Use a model with a recognizable backbone.",
                    self.model_name,
                    hidden.shape[-1],
                    _hs,
                )
                self._warned_logits_pool = True
        except Exception:
            pass
        return hidden

    def _get_hidden_size(self) -> int:
        """Get the model's hidden dimension size."""
        if hasattr(self._model, "config"):
            cfg = self._model.config
            for attr in ("hidden_size", "d_model", "n_embd", "embed_dim"):
                val = getattr(cfg, attr, None)
                if val is not None:
                    return val
        # Check args (MLX models store config in .args)
        args = getattr(self._model, "args", None)
        if args is not None:
            # Text config may be nested
            text_config = getattr(args, "text_config", None)
            for obj in (text_config, args):
                if obj is not None:
                    for attr in ("hidden_size", "d_model", "n_embd", "embed_dim"):
                        val = getattr(obj, attr, None)
                        if val is not None:
                            return val
        # Heuristic: check model layers
        if hasattr(self._model, "layers") and len(self._model.layers) > 0:
            layer = self._model.layers[0]
            if hasattr(layer, "mlp") and hasattr(layer.mlp, "fc1"):
                return layer.mlp.fc1.weight.shape[1]
        # Default fallback for common models
        return 768

    def _get_backbone(self):
        """Resolve the transformer backbone (before LM head projection).

        MLX models have varying structures:
        - Qwen3.5: model.language_model.model (inner Qwen3_5TextModel)
        - Standard HF: model.model
        - Some use model.transformer

        The backbone returns hidden states of shape [batch, seq, hidden_size]
        without projecting through the vocabulary head.
        """
        m = self._model
        # Qwen3.5 and similar: model.language_model.model
        lm = getattr(m, "language_model", None)
        if lm is not None:
            inner = getattr(lm, "model", None)
            if inner is not None and callable(inner):
                return inner
        # Standard: model.model
        inner = getattr(m, "model", None)
        if inner is not None and callable(inner):
            return inner
        # Some architectures: model.transformer
        inner = getattr(m, "transformer", None)
        if inner is not None and callable(inner):
            return inner
        return None

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 1.0,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        logit_bias: dict[int, float] | None = None,
        stop: list[str] | None = None,
        stop_token_ids: list[int] | None = None,
        seed: int | None = None,
        json_schema: dict | str | None = None,
        spec_decode: bool = False,
        use_engine_loop: bool | None = None,
        enable_thinking: bool | None = None,
        logprobs: bool = False,
        top_logprobs: int | None = None,
        thinking_budget: int | None = None,
        reasoning_effort: str | None = None,
        xtc_probability: float = 0.0,
        xtc_threshold: float = 0.0,
        cancel_event: asyncio.Event | None = None,
        priority: int = 0,
        logits_processors: list | None = None,
        timeout_seconds: float | None = None,
        images: list | None = None,
        lora_adapter: str | None = None,
        kv_cache_breakpoints: list[int] | None = None,
        min_tokens: int = 0,
        ignore_eos: bool = False,
        suppress_tokens: list[int] | None = None,
        prompt_logprobs: int | None = None,
    ) -> GenerationOutput:
        """Non-streaming text generation.

        Args:
            spec_decode: If True, use speculative decoding path when available.
                         Falls back to standard generation if spec decode is
                         not configured or the model lacks spec heads.
            use_engine_loop: If True, route through EngineCore's continuous
                             batching loop. If False, use fast path.
                             If None (default), uses YUNSHU_ENGINE_LOOP env var
                             (defaults to False if not set).
            logprobs: If True, return log probabilities for each generated token.
            top_logprobs: Number of top logprobs to return per token (max 20).
            timeout_seconds: Per-request generation timeout. If None, uses the
                             engine's default (300s). Overrides the engine default
                             for this request only.
        """
        if not self._loaded:
            await self.start()

        # Early return: max_tokens <= 0 produces no output
        if max_tokens <= 0:
            _pt = 0
            if self._tokenizer is not None:
                try:
                    if (
                        isinstance(prompt, list)
                        and prompt
                        and isinstance(prompt[0], dict)
                    ):
                        _text = self._apply_chat_template(prompt, enable_thinking)
                    else:
                        _text = prompt if isinstance(prompt, str) else str(prompt)
                    _pt = len(self._tokenizer.encode(_text))
                except Exception:
                    pass
            return GenerationOutput(
                text="",
                new_text="",
                prompt_tokens=_pt,
                completion_tokens=0,
                finished=True,
                finish_reason="length",
            )

        _use_engine_loop = self._should_use_engine_loop(use_engine_loop)

        # spec_decode is re-enabled ONLY for the one route proven
        # lossless AND beneficial — the Gemma-4 dual-load external-drafter
        # primitive (verified 2.08x, greedy-exact; YUNSHU_GEMMA4_ASSISTANT). The
        # other fast-path spec routes stay OFF because they are NOT lossless:
        # - cross-model / MTP on hybrid-recurrent models (Qwen3.5 ArraysCache)
        # can't trim/rollback their recurrent state → EMPTY output;
        # - the n-gram draft-verifier duplicates accepted tokens → wrong output;
        # - and on bandwidth-bound Apple Silicon those routes are also slower.
        # _gemma4_spec_eligible only returns True for greedy/pure-temperature
        # requests with the assistant drafter loaded, so routing them changes no
        # output. Everything else falls back to the verified fast path.
        if spec_decode:
            if (not _use_engine_loop) and self._gemma4_spec_eligible(
                logprobs=logprobs,
                json_schema=json_schema,
                logits_processors=logits_processors,
                logit_bias=logit_bias,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                repetition_penalty=repetition_penalty,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                xtc_probability=xtc_probability,
                lora_adapter=lora_adapter,
            ):
                logger.debug(
                    "spec_decode: routing to verified Gemma-4 assistant "
                    "primitive (lossless, ~2x)"
                )
            else:
                logger.debug(
                    "spec_decode requested — using standard generation "
                    "(only the Gemma-4 assistant primitive is proven "
                    "lossless here; MTP/cross-model/n-gram are not)"
                )
                spec_decode = False

        # Memory guard preflight check
        guard_rejection = self._check_memory_guard(prompt, max_tokens)
        if guard_rejection is not None:
            return guard_rejection

        # Resolve reasoning_effort → thinking_budget if not explicitly set
        if thinking_budget is None and reasoning_effort is not None:
            thinking_budget = _REASONING_EFFORT_MAP.get(reasoning_effort, 8192)
            if enable_thinking is None:
                enable_thinking = True

        # Gemma-4 default: its chat template enables thinking by default but
        # emits inline `thought` tokens that don't auto-stop, producing output
        # like "4thought\nThinking Process: …" for simple prompts. Default
        # enable_thinking=False for gemma-4 unless caller passed reasoning_effort.
        if enable_thinking is None and reasoning_effort is None:
            _mn = getattr(self, "model_name", None)
            if isinstance(_mn, str) and "gemma-4" in _mn.lower():
                enable_thinking = False

        # Response cache lookup (YUNSHU_RESPONSE_CACHE=1)
        _rc_hash = None
        # Only cache DETERMINISTIC requests. The HTTP middleware skips
        # temperature>0 && seed is None (it would freeze one random sample and replay it for
        # every identical request); the engine cache had NO such guard, so a sampled/creative
        # request got the same frozen completion every time. Mirror the middleware.
        _rc_deterministic = not (
            temperature is not None and temperature > 0 and seed is None
        )
        if not spec_decode and _rc_deterministic:
            try:
                from .gateway_optimizer import get_response_cache

                _rc = get_response_cache()
                if _rc.enabled:
                    _rc_hash = _rc.hash_request(
                        self.model_name,
                        prompt if isinstance(prompt, str) else str(prompt),
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        min_p=min_p,
                        repetition_penalty=repetition_penalty,
                        frequency_penalty=frequency_penalty,
                        presence_penalty=presence_penalty,
                        stop=str(stop),
                        seed=seed,
                        enable_thinking=enable_thinking,
                        thinking_budget=thinking_budget,
                        json_schema=str(json_schema),
                        # Output-affecting params must be in the key or a
                        # logprobs / biased / xtc request can collide with a
                        # plain one and serve a wrong cached response.
                        logprobs=logprobs,
                        top_logprobs=top_logprobs,
                        logit_bias=str(logit_bias),
                        xtc_probability=xtc_probability,
                        xtc_threshold=xtc_threshold,
                        # These ALSO change the output and were missing from the key —
                        # lora_adapter is the worst (LoRA is the product feature: a request for
                        # adapter Y collided with a cached adapter X response → wrong fine-tuned
                        # weights served as 200 OK).
                        lora_adapter=str(lora_adapter),
                        stop_token_ids=str(stop_token_ids),
                        min_tokens=min_tokens,
                        ignore_eos=ignore_eos,
                        suppress_tokens=str(suppress_tokens),
                    )
                    _rc_hit = await _rc.get(_rc_hash)
                    if _rc_hit is not None:
                        self._response_cache_hits += 1
                        return _rc_hit
                    self._response_cache_misses += 1
            except Exception:
                logger.debug("response cache lookup failed", exc_info=True)

        # ── Context window truncation for long prompts ──
        if isinstance(prompt, list) and prompt and isinstance(prompt[0], dict):
            try:
                if self._tokenizer and hasattr(self._tokenizer, "encode"):
                    text = self._apply_chat_template(prompt, enable_thinking)
                    token_count = len(self._tokenizer.encode(text))
                    max_ctx = getattr(self._model, "max_seq_len", None)
                    if max_ctx is None:
                        max_ctx = getattr(
                            getattr(self._model, "config", None), "max_seq_len", None
                        ) or getattr(
                            getattr(self._model, "args", None), "max_seq_len", None
                        )
                    # Thinking tokens also consume context window positions —
                    # subtract them from the available prompt budget.
                    _thinking_overhead = (
                        thinking_budget if (thinking_budget and enable_thinking) else 0
                    )
                    _generation_budget = max_tokens + _thinking_overhead
                    if max_ctx and token_count + _generation_budget > max_ctx:
                        from .context_window import ContextWindowManager

                        ctx_mgr = ContextWindowManager(
                            token_counter=lambda text: len(
                                self._tokenizer.encode(text)
                            ),
                        )
                        result = ctx_mgr.compute_truncation(
                            messages=prompt,
                            max_tokens=max_ctx - _generation_budget,
                            strategy="importance_aware",
                        )
                        prompt = result.messages
                        logger.debug(
                            f"Context window truncated: {token_count} → "
                            f"{result.truncated_token_count} tokens (saved {result.tokens_saved})"
                        )
            except Exception:
                logger.warning("context window truncation skipped", exc_info=True)

        # Gemma-4 assistant-drafter spec decode (#175): n=1 serving via the
        # validated dual-load primitive (~2.5x, lossless). Eligibility-gated so
        # the output is identical to normal generation; only active when the
        # drafter was loaded (YUNSHU_GEMMA4_ASSISTANT).
        if (
            spec_decode
            and not _use_engine_loop
            and self._gemma4_spec_eligible(
                logprobs=logprobs,
                json_schema=json_schema,
                logits_processors=logits_processors,
                logit_bias=logit_bias,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                repetition_penalty=repetition_penalty,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                xtc_probability=xtc_probability,
                lora_adapter=lora_adapter,
            )
        ):
            try:
                return await self._generate_gemma4_assistant_spec(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    seed=seed,
                    enable_thinking=enable_thinking,
                    stop_token_ids=stop_token_ids,
                )
            except Exception:
                logger.warning(
                    "Gemma-4 assistant spec decode failed; falling back", exc_info=True
                )

        # Speculative decoding path (Phase 4: single-request EAGLE-3).
        # NB: all spec paths below are gated on `not logprobs` — none of them
        # populate per-token logprobs, so when logprobs are requested correctness
        # wins and we fall through to normal generation (which does).
        # The cross-model SpeculativeDecoder omits residual-
        # distribution resampling and tests acceptance at temperature 1.0
        # regardless of request temperature, so it is NOT lossless for
        # temperature>0 (output distribution is biased toward the greedy
        # sequence). Restrict it to GREEDY requests, where longest-exact-argmax
        # acceptance IS lossless; temp>0 falls through to correct normal decode.
        if (
            spec_decode
            and not logprobs
            and not _use_engine_loop
            and self._spec_enabled
            and self._spec_decoder is not None
            and temperature <= 0.0
        ):
            return await self._generate_speculative(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                repetition_penalty=repetition_penalty,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                logit_bias=logit_bias,
                stop=stop,
                stop_token_ids=stop_token_ids,
                seed=seed,
                enable_thinking=enable_thinking,
                logprobs=logprobs,
                top_logprobs=top_logprobs,
                thinking_budget=thinking_budget,
                xtc_probability=xtc_probability,
                xtc_threshold=xtc_threshold,
                json_schema=json_schema,
                cancel_event=cancel_event,
                logits_processors=logits_processors,
                timeout_seconds=timeout_seconds or 300.0,
                lora_adapter=lora_adapter,
            )

        # MTP speculative decoding (built-in multi-token prediction heads)
        if (
            spec_decode
            and not logprobs
            and self._mtp_decoder is not None
            and not _use_engine_loop
        ):
            return await self._generate_mtp(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                repetition_penalty=repetition_penalty,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                logit_bias=logit_bias,
                stop=stop,
                stop_token_ids=stop_token_ids,
                seed=seed,
                enable_thinking=enable_thinking,
                logprobs=logprobs,
                top_logprobs=top_logprobs,
                thinking_budget=thinking_budget,
                cancel_event=cancel_event,
                json_schema=json_schema,
                xtc_probability=xtc_probability,
                xtc_threshold=xtc_threshold,
                logits_processors=logits_processors,
                timeout_seconds=timeout_seconds or 300.0,
                lora_adapter=lora_adapter,
            )

        # N-gram speculative decoding (model-free, CPU-based proposal).
        # N-gram verify accepts a draft iff it equals the
        # target's ARGMAX but samples the bonus at the request temperature, so for
        # temperature>0 accepted tokens are forced to the greedy sequence (biased,
        # not lossless). Restrict to GREEDY requests; temp>0 → normal decode.
        #
        # Default-on for greedy: measured byte-identical to the plain fast path
        # across diverse prompts (it's lossless at temp<=0) and never slower
        # (1.00–1.07x on Qwen2.5-3B; larger wins on copy-heavy / agentic output),
        # so a greedy request takes this path WITHOUT needing spec_decode=true.
        # The proposer is free when idle and the adaptive controller backs off,
        # so there's no penalty when acceptance is low. Opt out with
        # YUNSHU_NGRAM_DEFAULT=0 (or YUNSHU_NGRAM_SPEC=0 to drop the proposer).
        if (
            self._ngram_proposer is not None
            and (spec_decode or self._ngram_greedy_default)
            and not logprobs
            and not _use_engine_loop
            and temperature <= 0.0
        ):
            return await self._generate_ngram_spec(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                repetition_penalty=repetition_penalty,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                logit_bias=logit_bias,
                stop=stop,
                stop_token_ids=stop_token_ids,
                seed=seed,
                logprobs=logprobs,
                top_logprobs=top_logprobs,
                json_schema=json_schema,
                xtc_probability=xtc_probability,
                xtc_threshold=xtc_threshold,
                enable_thinking=enable_thinking,
                thinking_budget=thinking_budget,
                cancel_event=cancel_event,
                logits_processors=logits_processors,
                timeout_seconds=timeout_seconds or 300.0,
                lora_adapter=lora_adapter,
            )

        # Fast path: direct generate_step on executor thread for full GPU utilization
        if not _use_engine_loop:
            result = await self._generate_fast(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                repetition_penalty=repetition_penalty,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                logit_bias=logit_bias,
                stop=stop,
                stop_token_ids=stop_token_ids,
                seed=seed,
                enable_thinking=enable_thinking,
                logprobs=logprobs,
                top_logprobs=top_logprobs,
                thinking_budget=thinking_budget,
                xtc_probability=xtc_probability,
                xtc_threshold=xtc_threshold,
                json_schema=json_schema,
                cancel_event=cancel_event,
                logits_processors=logits_processors,
                priority=priority,
                timeout_seconds=timeout_seconds or 300.0,
                lora_adapter=lora_adapter,
                kv_cache_breakpoints=kv_cache_breakpoints,
                min_tokens=min_tokens,
                ignore_eos=ignore_eos,
                suppress_tokens=suppress_tokens,
            )
            if _rc_hash is not None and result.finish_reason != "error":
                try:
                    from .gateway_optimizer import get_response_cache

                    await get_response_cache().put(_rc_hash, result)
                except Exception:
                    logger.debug("response cache store failed", exc_info=True)
            # prompt_logprobs — one extra prompt forward (eval/perplexity).
            # Opt-in, length-gated (full [seq,vocab] logits are large), and run on
            # the MLX executor so it doesn't block the event loop. Never fails the
            # request — best-effort attach.
            if prompt_logprobs is not None and result.finish_reason != "error":
                try:
                    _pl = await self._compute_prompt_logprobs_for(
                        prompt,
                        enable_thinking,
                        int(prompt_logprobs),
                    )
                    result.prompt_logprobs = _pl
                except Exception:
                    logger.debug("prompt_logprobs computation failed", exc_info=True)
            return result

        # Engine loop path: continuous batching with scheduler overhead
        result = await self._engine_core.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            logit_bias=logit_bias,
            stop=stop,
            stop_token_ids=stop_token_ids,
            seed=seed,
            json_schema=json_schema,
            enable_thinking=enable_thinking,
            thinking_budget=thinking_budget,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            priority=priority,
            xtc_probability=xtc_probability,
            xtc_threshold=xtc_threshold,
            reasoning_effort=reasoning_effort,
            logits_processors=logits_processors,
            cancel_event=cancel_event,
            lora_adapter=lora_adapter,
            kv_cache_breakpoints=kv_cache_breakpoints,
            # The engine-loop branch silently dropped min_tokens / ignore_eos /
            # suppress_tokens — the scheduler's _make_sampler honors them but they
            # were never threaded here, so an engine-loop request ignored all three (e.g.
            # min_tokens floor, ignore_eos throughput runs, suppress_tokens bans). The fast
            # path passes them (above); make the opt-in loop match.
            min_tokens=min_tokens,
            ignore_eos=ignore_eos,
            suppress_tokens=suppress_tokens,
        )

        if result is None:
            return GenerationOutput(
                finished=True, finish_reason="error", error="engine_core returned None"
            )

        # Map engine_core finish_reason to OpenAI-compatible finish_reason
        finish_reason = result.finish_reason
        if finish_reason == "memory_exceeded":
            finish_reason = "memory_limit"
        # Guard: finish_reason must never be None when finished=True
        if finish_reason is None:
            finish_reason = "stop"

        # Apply output parser to extract reasoning/tool_calls from raw text
        output_text = _clean_special_tokens(result.output_text)
        _reasoning_tok = 0
        try:
            from .output_parser import parse_output

            parsed = parse_output(output_text, self.model_name)
            if parsed.finish_reason:
                finish_reason = parsed.finish_reason
            # Use cleaned content (reasoning stripped) as the main text
            if parsed.reasoning and parsed.content != output_text:
                output_text = parsed.content
        except Exception:
            logger.debug("output_parser failed", exc_info=True)

        # Reasoning parser: model-specific reasoning extraction with token count
        # Supplements output_parser with per-model reasoning token counting.
        # Fall back to scheduler-computed reasoning_tokens when parser returns 0.
        _reasoning_tok = getattr(result, "reasoning_tokens", 0) or 0
        try:
            from .reasoning_parser import get_reasoning_parser

            rp = get_reasoning_parser(self.model_name)
            rp_out = rp.parse(output_text)
            if rp_out.reasoning and rp_out.reasoning_tokens > 0:
                _reasoning_tok = rp_out.reasoning_tokens
        except Exception:
            logger.debug("reasoning_parser failed", exc_info=True)

        # TTFT from engine_core (computed before request cleanup)
        _ttft_ms = getattr(result, "ttft_ms", 0.0)

        # Record TTFT in Prometheus (consistency with fast path)
        if _ttft_ms > 0:
            try:
                from yunshu_gateway.middleware.prometheus_exporter import (
                    get_prometheus_metrics,
                )

                pm = get_prometheus_metrics()
                pm.observe_histogram(
                    "ttft_seconds",
                    _ttft_ms / 1000.0,
                    labels={"model_id": self.model_label},
                )
            except Exception:
                logger.debug(
                    "engine loop TTFT prometheus recording failed", exc_info=True
                )

        engine_loop_result = GenerationOutput(
            text=output_text,
            new_text=output_text,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            finished=True,
            finish_reason=finish_reason,
            reasoning_tokens=_reasoning_tok,
            cached_tokens=getattr(result, "cached_tokens", 0),
            logprobs=getattr(result, "logprobs", None),
            ttft_ms=_ttft_ms,
            error=getattr(result, "error", None),
        )
        if _rc_hash is not None and engine_loop_result.finish_reason != "error":
            try:
                from .gateway_optimizer import get_response_cache

                await get_response_cache().put(_rc_hash, engine_loop_result)
            except Exception:
                logger.debug("response cache store failed", exc_info=True)
        return engine_loop_result

    def _capture_hybrid_prefix(
        self, model, full_ids, cache, prefix_cache, start_offset, block
    ):
        """Advance `cache` through full_ids[start_offset:-1] in block-sized
        chunks, storing a trim=0 block-boundary snapshot into `prefix_cache` at
        each boundary. The last token is intentionally NOT consumed — it is left
        for generate_step to prefill and produce the first decode logits.

        This is how the fast path reuses prefixes for HYBRID
        models (Qwen3.5). A boundary snapshot holds the EXACT recurrent state
        after N tokens, so a later request sharing that N-token prefix reuses it
        with zero trim — never slicing the non-trimmable ArraysCache layers.
        Returns the absolute number of prompt tokens now resident in `cache`.
        """
        n = len(full_ids)
        if n <= 1:
            return start_offset
        p = int(start_offset)
        end = n - 1  # leave the final token for generate_step
        while p < end:
            chunk = full_ids[p : min(p + block, end)]
            cn = int(chunk.shape[0])
            if cn == 0:
                break
            _ = model(chunk[None], cache=cache)
            p += cn
            # NB: no per-chunk mx.eval. MLX arrays are immutable — each model()
            # call produces NEW key/value arrays — so a detached snapshot taken
            # now stays valid even though the prefill graph is still lazy. Forcing
            # eval here would serialize the prefill (a large cold-latency penalty);
            # leaving it lazy keeps prefill pipelined and materializes once at the
            # first decode token.
            if p % block == 0 and p >= prefix_cache._min_prefix:
                try:
                    prefix_cache.add(full_ids[:p], cache)
                except Exception:
                    logger.debug("hybrid boundary snapshot add failed", exc_info=True)
        return p

    async def _generate_fast(
        self,
        prompt: str | list[dict],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 1.0,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        logit_bias: dict[int, float] | None = None,
        stop: list[str] | None = None,
        stop_token_ids: list[int] | None = None,
        seed: int | None = None,
        enable_thinking: bool | None = None,
        logprobs: bool = False,
        top_logprobs: int | None = None,
        thinking_budget: int | None = None,
        timeout_seconds: float = 300.0,
        xtc_probability: float = 0.0,
        xtc_threshold: float = 0.0,
        json_schema: dict | str | None = None,
        cancel_event: asyncio.Event | None = None,
        logits_processors: list | None = None,
        priority: int = 0,
        lora_adapter: str | None = None,
        kv_cache_breakpoints: list[int] | None = None,
        min_tokens: int = 0,
        ignore_eos: bool = False,
        suppress_tokens: list[int] | None = None,
    ) -> GenerationOutput:
        """Fast path: run generate_step directly on executor thread.

        Bypasses EngineCore's continuous batching loop for single requests.
        Runs the entire generation in one tight GPU loop on the MLX executor,
        eliminating per-token async round-trip overhead for full GPU utilization.
        Note: priority is accepted for API consistency but has no effect in the
        single-request fast path (no scheduler contention).
        """
        from mlx_lm.generate import generate_step
        from mlx_lm.sample_utils import make_sampler

        tokenizer = self._tokenizer
        model = self._model

        # ── Model preprocessor for multimodal input ──
        if isinstance(prompt, list) and prompt and isinstance(prompt[0], dict):
            # Check for multimodal content (images, audio, video)
            has_multimodal = any(
                isinstance(c.get("content"), list)
                for c in prompt
                if isinstance(c.get("content"), list)
            )
            if has_multimodal and self._preprocessor_registry is not None:
                try:
                    model_config = {"model_type": self.model_name or ""}
                    if hasattr(model, "config") and hasattr(model.config, "model_type"):
                        model_config["model_type"] = model.config.model_type
                    preprocessor = self._preprocessor_registry.detect(model_config)
                    if preprocessor is not None:
                        processed = preprocessor.preprocess(prompt, tokenizer=tokenizer)
                        if processed.token_ids:
                            prompt = processed.token_ids
                except Exception:
                    logger.debug(
                        "model preprocessor failed, using raw prompt", exc_info=True
                    )

        # ── Pre-encoding context window truncation (message-level) ──
        # When prompt is a list of message dicts, use ContextWindowManager to
        # truncate at the message level BEFORE applying the chat template.
        # This preserves system prompts — the engine-loop path does the same.
        if isinstance(prompt, list) and prompt and isinstance(prompt[0], dict):
            try:
                _max_ctx_pre = getattr(model, "max_seq_len", None)
                if _max_ctx_pre is None:
                    _max_ctx_pre = getattr(
                        getattr(model, "config", None), "max_seq_len", None
                    ) or getattr(getattr(model, "args", None), "max_seq_len", None)
                if _max_ctx_pre and _max_ctx_pre > 0:
                    _thinking_overhead = (
                        thinking_budget if (thinking_budget and enable_thinking) else 0
                    )
                    _generation_budget = max_tokens + _thinking_overhead
                    _est_tokens = sum(
                        len(str(m.get("content", ""))) // 4 + 4 for m in prompt
                    )
                    if _est_tokens + _generation_budget > _max_ctx_pre:
                        from .context_window import ContextWindowManager

                        ctx_mgr = ContextWindowManager(
                            token_counter=lambda text: len(tokenizer.encode(text)),
                        )
                        result = ctx_mgr.compute_truncation(
                            messages=prompt,
                            max_tokens=_max_ctx_pre - _generation_budget,
                            strategy="importance_aware",
                        )
                        prompt = result.messages
                        logger.debug(
                            "Fast path pre-encode truncation: estimated %d → %d tokens",
                            _est_tokens,
                            result.truncated_token_count,
                        )
            except Exception:
                logger.debug(
                    "context window truncation skipped in fast path", exc_info=True
                )

        # Encode prompt
        if isinstance(prompt, str):
            text = prompt
        elif isinstance(prompt, list) and prompt and isinstance(prompt[0], dict):
            # Route through _apply_chat_template (NOT raw tokenizer.apply_chat_template)
            # so the fast path — the DEFAULT serving path — also gets: tool_calls.
            # arguments str→dict normalization (multi-turn tool history otherwise
            # double-encodes or raises in Jinja → silent degraded fallback), the
            # family message adapter (Gemma4/Mistral/Harmony role rules), dangling
            # <think> closing, and the enable_thinking-unsupported retry.
            text = self._apply_chat_template(prompt, enable_thinking=enable_thinking)
        else:
            text = str(prompt)

        input_ids = self._encode_prompt(tokenizer, text)
        prompt_tokens = len(input_ids)

        if not input_ids:
            bos_id = getattr(tokenizer, "bos_token_id", None)
            if bos_id is not None:
                input_ids = [bos_id]
            else:
                eos_id = getattr(tokenizer, "eos_token_id", 1)
                input_ids = [eos_id]
            prompt_tokens = len(input_ids)

        # ── Context window: clamp generation + truncate over-long prompt ──
        # Resolve the REAL context window. The old code read ONLY the
        # `max_seq_len` attribute, which mlx-lm models don't expose (Qwen/Llama use
        # `max_position_embeddings`) → this safety net was DEAD for most models
        # (same root cause as the gateway-guard bug). And it only handled
        # prompt-alone-too-long; a prompt that FITS but whose prompt+max_tokens
        # exceeds the window decoded past max_position_embeddings into
        # RoPE-extrapolated garbage. Resolve correctly, then clamp max_tokens.
        _max_ctx = _resolve_model_max_ctx(model)
        if _max_ctx and _max_ctx > 0:
            # Deeper fallback: prompt alone exceeds the window → left-truncate.
            # (The gateway context guard normally rejects this with a 400 first.)
            if prompt_tokens >= _max_ctx:
                _orig = prompt_tokens
                input_ids = input_ids[-max(1, _max_ctx - 1) :]
                prompt_tokens = len(input_ids)
                logger.warning(
                    "Fast path prompt truncated to fit context: %d → %d tokens (ctx=%d)",
                    _orig,
                    prompt_tokens,
                    _max_ctx,
                )
            # Clamp max_tokens so prompt + generation stays within the window
            # (prevents RoPE-extrapolated garbage past max_position_embeddings).
            _room = _max_ctx - prompt_tokens
            if _room >= 1 and max_tokens > _room:
                logger.info(
                    "Fast path clamped max_tokens %d → %d to fit context "
                    "(prompt=%d, ctx=%d)",
                    max_tokens,
                    _room,
                    prompt_tokens,
                    _max_ctx,
                )
                max_tokens = _room

        stop_ids = set()
        # eos_token_id / eos_token_ids may be a single int OR an
        # iterable depending on the tokenizer (Qwen3.6-27B exposes eos_token_ids
        # as a bare int, which crashed `stop_ids.update(...)` with "'int' object
        # is not iterable"). Accept both shapes.
        # Collect EOS ids SEPARATELY so ignore_eos can suppress only the
        # model EOS (not user stop_token_ids), and min_tokens can mask them.
        _eos_ids: set[int] = set()
        _eid = getattr(tokenizer, "eos_token_id", None)
        if _eid is not None:
            _eos_ids.update(_eid if isinstance(_eid, (list, tuple, set)) else (_eid,))
        _eids = getattr(tokenizer, "eos_token_ids", None)
        if _eids is not None:
            _eos_ids.update(
                _eids if isinstance(_eids, (list, tuple, set)) else (_eids,)
            )
        if not ignore_eos:
            stop_ids.update(_eos_ids)

        # Pre-encode stop strings for suffix matching.
        # A user stop string must ALWAYS be string-matched.
        # The previous code routed single-token stops to stop_ids via
        # tokenizer.encode(s), but the encoded id of a BARE stop ("C", "3")
        # rarely equals the SPACE-PREFIXED token the model actually emits (" C"),
        # so token-id matching silently missed most single-token stops (the stop
        # text leaked with finish_reason="stop" coming from natural EOS). We keep
        # the token-id as a cheap early-stop hint but ALSO always add the string
        # to stop_suffixes so the string matcher + final truncation enforce it.
        stop_suffixes = []
        if stop:
            for s in stop:
                if not s:
                    continue
                ids = tokenizer.encode(s)
                if len(ids) == 1:
                    stop_ids.add(ids[0])
                stop_suffixes.append(s)
        if stop_token_ids:
            stop_ids.update(stop_token_ids)

        # For temp>0, bypass mlx-lm's @mx.compile-wrapped categorical_sampling
        # which traps the FIRST call's PRNG state in its compile cache (so
        # re-seeding via mx.random.seed() between requests is a no-op). Use
        # a numpy.random.Generator backed sampler instead — same shape as
        # the vlm_engine fix. Greedy (temp=0) keeps mlx-lm's
        # compiled argmax (deterministic anyway, faster).
        if temperature is not None and temperature > 1e-6:
            sampler = _build_temp_sampler(
                temperature=temperature,
                top_p=top_p,
                top_k=top_k if top_k and top_k > 0 else 0,
                min_p=min_p if min_p else 0.0,
                seed=seed,
                xtc_probability=xtc_probability,
                xtc_threshold=xtc_threshold,
            )
        else:
            sampler = make_sampler(
                temp=temperature,
                top_p=top_p,
                top_k=top_k if top_k > 0 else 0,
                min_p=min_p,
                xtc_probability=xtc_probability,
                xtc_threshold=xtc_threshold,
            )

        # JSON Schema / grammar constraint: wrap sampler with ConstrainedSampler
        if json_schema is not None:
            try:
                sampler = _build_constrained_sampler(sampler, json_schema, tokenizer)
            except Exception:
                logger.warning(
                    "Grammar constraint setup failed, falling back to unconstrained",
                    exc_info=True,
                )

        # Jump-forward decoding (opt-in YUNSHU_JUMP_FORWARD=1) for
        # JSON-schema-constrained GREEDY generation — emits the grammar-forced
        # structure with one forward per BRANCH instead of per token. Bypasses the
        # normal generate_step loop; gated + greedy + JSON-schema only (not
        # regex/choice/cfg, not logprobs) so the default path is untouched.
        # NOTE: the custom loop honors only greedy JSON-schema decoding — it does
        # NOT apply lora_adapter, cancel_event, logit_bias, suppress_tokens, or
        # min_tokens. Gate OFF whenever any of those are set so we never silently
        # ignore a request constraint; those requests fall through to the normal
        # constrained decode below.
        if (
            os.environ.get("YUNSHU_JUMP_FORWARD", "").strip().lower()
            in ("1", "true", "yes")
            and json_schema is not None
            and not logprobs
            and (temperature is None or temperature <= 1e-6)
            and lora_adapter is None
            and cancel_event is None
            and not logit_bias
            and not suppress_tokens
            and not min_tokens
            and not (
                isinstance(json_schema, dict)
                and json_schema.get("type") in ("regex", "choice", "cfg")
            )
        ):
            try:
                import json as _json
                import time as _jf_time

                from .json_schema import JsonSchemaConstraint
                from .mlx_executor import get_mlx_executor

                if json_schema == "json_object":
                    _jf_schema = None
                elif isinstance(json_schema, str):
                    _jf_schema = _json.loads(json_schema)
                else:
                    _jf_schema = json_schema
                _jf_constraint = JsonSchemaConstraint(_jf_schema)
                _jf_t0 = _jf_time.perf_counter()
                _jf_loop = asyncio.get_running_loop()
                # Count this against the fast-path concurrency gauge like the normal
                # path (it occupies the same single MLX executor thread).
                self._active_fast_path_count += 1
                try:
                    (
                        _jf_text,
                        _jf_ids,
                        _jf_nfwd,
                        _jf_stop,
                    ) = await _jf_loop.run_in_executor(
                        get_mlx_executor(),
                        lambda: self._jump_forward_generate_sync(
                            input_ids, _jf_constraint, max_tokens, stop_ids, stop
                        ),
                    )
                finally:
                    self._active_fast_path_count -= 1
                logger.debug(
                    "jump-forward: %d tokens in %d forwards (%.1fx)",
                    len(_jf_ids),
                    _jf_nfwd,
                    len(_jf_ids) / max(_jf_nfwd, 1),
                )
                # A stop-substring termination is finish_reason="stop"
                # (and _jf_ids is now truncated at the stop, so completion_tokens is
                # accurate) — only report "length" when we genuinely hit the budget
                # without a stop.
                _jf_fr = (
                    "length"
                    if (not _jf_stop and len(_jf_ids) >= max_tokens)
                    else "stop"
                )
                return GenerationOutput(
                    text=_jf_text,
                    new_text=_jf_text,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=len(_jf_ids),
                    finished=True,
                    finish_reason=_jf_fr,
                    ttft_ms=(_jf_time.perf_counter() - _jf_t0) * 1000.0,
                )
            except Exception:
                logger.warning(
                    "jump-forward path failed; falling back to normal "
                    "constrained decode",
                    exc_info=True,
                )

        # SAMP-2: Save user-provided custom logits processors before building internal list
        # Length-gated KV-quant bits for THIS request (8-bit default at
        # long context; explicit config wins; None below the threshold).
        _req_kv_bits = self._effective_kv_quant_bits(prompt_tokens + max_tokens)
        _custom_logits_processors = logits_processors or []
        logits_processors = []
        if repetition_penalty != 1.0:

            def _rep_penalty(tokens, logits, rp=repetition_penalty, ctx=20):
                if len(tokens) > 0:
                    recent = tokens[-ctx:]
                    import mlx.core as _mx

                    sel = logits[..., recent]
                    sel = _mx.where(sel < 0, sel * rp, sel / rp)
                    logits[..., _mx.array(recent)] = sel
                return logits

            logits_processors.append(_rep_penalty)
        if frequency_penalty != 0.0 or presence_penalty != 0.0:
            # Maintain incremental counts dict instead of rebuilding
            # from full tokens list every step (O(T²) → O(T) total). The
            # closure captures a mutable state dict so successive calls only
            # observe the newest token.
            _fp_state: dict[str, object] = {"counts": {}, "last_len": -1}

            # The prompt offset inside the processor is 1, NOT the full
            # prompt length. mlx-lm prefills all-but-the-last prompt token OUTSIDE
            # _step, so the `tokens` accumulator handed to logits_processors is
            # [last_prompt_token, gen1, gen2, ...]. With n_prompt=prompt_tokens the
            # guard `cur_len <= n_prompt` made frequency/presence penalty INERT for
            # the first ~prompt_len generated tokens (short completions never
            # penalized at all) and then tokens[n_prompt:] under-counted repeats.
            def _freq_pres_penalty(
                tokens,
                logits,
                fp=frequency_penalty,
                pp=presence_penalty,
                n_prompt=1,
                _st=_fp_state,
            ):
                counts: dict[int, int] = _st["counts"]  # type: ignore[assignment]
                last_len = int(_st["last_len"])  # type: ignore[arg-type]
                cur_len = len(tokens)
                if cur_len <= n_prompt:
                    _st["last_len"] = cur_len
                    return logits
                # Reset/rebuild if tokens shrank (spec-decode rollback) or
                # we have not yet started incremental accounting.
                if cur_len < last_len or last_len < n_prompt:
                    counts = {}
                    for t in tokens[n_prompt:]:
                        counts[int(t)] = counts.get(int(t), 0) + 1
                    _st["counts"] = counts
                else:
                    start = max(last_len, n_prompt)
                    for t in tokens[start:]:
                        counts[int(t)] = counts.get(int(t), 0) + 1
                _st["last_len"] = cur_len
                for tid, cnt in counts.items():
                    # OpenAI permits frequency/presence penalty in [-2.0, 2.0];
                    # negative values BOOST the token (encourage repetition).
                    # Guarding with `> 0` silently dropped negative penalties
                    # (verified: fp=-2.0 produced output identical to fp=0).
                    if fp != 0.0:
                        logits[..., tid] = logits[..., tid] - fp * cnt
                    if pp != 0.0 and cnt > 0:
                        logits[..., tid] = logits[..., tid] - pp
                return logits

            logits_processors.append(_freq_pres_penalty)
        if logit_bias:

            def _logit_bias_proc(_tokens, logits, biases=logit_bias):
                # Skip token ids outside [0, vocab) — a user-supplied out-of-range
                # or negative id would otherwise index out of bounds and crash the
                # request (MLX: "Cannot squeeze axis 1 with size 0"). OpenAI ignores
                # invalid logit_bias token ids rather than erroring.
                vocab = logits.shape[-1]
                for tid, bias in biases.items():
                    if 0 <= tid < vocab:
                        logits[..., tid] = logits[..., tid] + bias
                return logits

            logits_processors.append(_logit_bias_proc)

        # suppress_tokens — hard-ban specific token ids (logits → -inf).
        if suppress_tokens:
            _sup = [int(t) for t in suppress_tokens]

            def _suppress_proc(_tokens, logits, sup=_sup):
                vocab = logits.shape[-1]
                for tid in sup:
                    if 0 <= tid < vocab:
                        logits[..., tid] = -float("inf")
                return logits

            logits_processors.append(_suppress_proc)

        # min_tokens — mask EOS + single-token stops to -inf until at
        # least min_tokens have been GENERATED, so generation can't end early.
        # mlx-lm's tokens accumulator is [last_prompt_token, gen1, ...] so the
        # generated count is len(tokens) - 1 (the prompt-offset insight).
        # SKIP when a JSON/grammar constraint is active. The constraint
        # only permits EOS once the document is structurally COMPLETE (so it
        # already enforces a minimum), and masking EOS at the DONE state leaves the
        # constrained sampler with an all-(-inf) allowed set → its argmax fallback
        # emits an INVALID non-EOS token after a complete JSON value. Valid JSON
        # must win over min_tokens (which can't lengthen structured output anyway).
        if min_tokens and min_tokens > 0 and stop_ids and json_schema is None:
            _mask_ids = list(stop_ids)

            def _min_tokens_proc(tokens, logits, ids=_mask_ids, floor=int(min_tokens)):
                if (len(tokens) - 1) < floor:
                    vocab = logits.shape[-1]
                    for tid in ids:
                        if 0 <= tid < vocab:
                            logits[..., tid] = -float("inf")
                return logits

            logits_processors.append(_min_tokens_proc)

        # SAMP-2: Wrap user-provided custom logits processors to adapt signature.
        # User processors take (token_ids: list[int], logits: mx.array) -> mx.array
        # but generate_step passes (tokens: mx.array, logits: mx.array).
        if _custom_logits_processors:
            logits_processors.extend(
                _wrap_custom_logits_processor(p) for p in _custom_logits_processors
            )

        # Convert KV cache breakpoint char offsets to token positions.
        _kv_breakpoint_token_positions: list[int] = []
        if kv_cache_breakpoints and isinstance(text, str) and len(text) > 0:
            for char_off in kv_cache_breakpoints:
                if char_off <= 0 or char_off > len(text):
                    continue
                try:
                    prefix_ids = tokenizer.encode(text[:char_off])
                    _kv_breakpoint_token_positions.append(len(prefix_ids))
                except Exception:
                    logger.debug(
                        "KV breakpoint char->token conversion failed at offset %d",
                        char_off,
                        exc_info=True,
                    )

        # Generate inflight request ID outside the closure so it is
        # accessible in the outer exception handlers below. Previously
        # this was defined inside _run() which caused a NameError when
        # an exception fired before the executor ran the closure.
        _inflight_req_id = f"fp-{id(generate_step)}-{int(time.monotonic() * 1e6)}"

        # LoRA concurrency keystone: acquire+apply and release+restore are NO
        # LONGER done here on the event loop. They run inside `_run_with_lora` on the
        # max_workers=1 MLX executor (below), serialized with generate_step — so a
        # release-triggered _restore_base can never mutate the shared model while another
        # request's generation reads it (the cross-thread race the concurrency audit found).

        def _run():
            import mlx.core as mx

            if seed is not None:
                mx.random.seed(seed)
            ids = mx.array(input_ids)
            tokens = []
            token_logprobs = []
            ttft_s = 0.0
            cached_tokens = 0
            detokenizer = tokenizer.detokenizer
            detokenizer.reset()
            if self._lookahead_reasoning is not None:
                self._lookahead_reasoning._in_thinking = False
                self._lookahead_reasoning._thinking_tokens = []
                self._lookahead_reasoning._recent_accepts = []
            thinking_tokens_used = 0
            think_end_token = None
            think_start_token = None
            _in_thinking = False
            _thinking_tokens: list[int] = []
            _stopped_by_suffix = False
            _stopped_by_stop_id = False

            # Prefill progress tracking for the fast path
            _prefill_req_id = f"fp-{id(_run)}-{int(time.monotonic() * 1e6)}"
            _prefill_tracker = None
            try:
                from .prefill_progress import get_prefill_tracker

                _prefill_tracker = get_prefill_tracker()
                _prefill_tracker.update(
                    _prefill_req_id, 0, prompt_tokens, self.model_name or "default"
                )
            except Exception:
                logger.debug("prefill tracker setup failed", exc_info=True)
                _prefill_tracker = None

            if thinking_budget is not None or enable_thinking:
                # Resolve via the bracketed-form helper (the bare "</think"
                # encoded to 2 tokens for Qwen3/DeepSeek-R1 → guard failed → state machine
                # never engaged). Multi-token markers still resolve to (None, None). The
                # helper swallows its own errors, so no surrounding try/except is needed.
                think_start_token, think_end_token = _resolve_think_token_ids(tokenizer)

            # Seed reproducibility: when caller pins a seed for sampling
            # determinism, bypass KV prefix / prompt caches. Cached KV blocks
            # came from an earlier prefill whose intermediate accumulation
            # order is not bit-exact to a fresh prefill (bf16 / async eval),
            # so the first-decode logits drift by a few ULPs and a seeded
            # categorical sample lands on a different token. Disabling both
            # caches when seed is set restores cold==warm reproducibility at
            # the cost of TTFT on repeat prompts (acceptable for a seeded
            # request).
            _bypass_cache_for_seed = seed is not None and temperature > 0

            # Hybrid-model guard: prompt/prefix KV reuse both rely on trimming
            # the cached KV (snapshot trim at store, trim=1 + re-feed at lookup,
            # arbitrary-prefix trim for the prefix cache). Hybrid models mix
            # full-attention KVCache layers (trimmable) with linear/recurrent
            # ArraysCache layers (NOT trimmable — their recurrent state encodes
            # every token seen and cannot be sliced back). Reusing such a cache
            # silently corrupts the recurrent layers and produces grossly wrong
            # logits (verified: cold 'Thinking' -> warm garbage, logit Δ≈13).
            # mlx_lm.can_trim_prompt_cache reports False for these caches, so
            # disable ALL KV caching for them — correctness over TTFT.
            _cache_trimmable = self._cache_supports_trim(model)
            # Hybrid models are normally bypassed (not
            # trimmable). When YUNSHU_HYBRID_PREFIX is on, route them through the
            # boundary-snapshot path instead: keep the cache enabled, force
            # trim=0-only reuse (no_trim mode), and chunk-prefill with snapshots.
            _hybrid_mode = (
                (not _cache_trimmable)
                and self._hybrid_prefix_enabled
                and not _bypass_cache_for_seed
                and self._kv_prefix_cache is not None
            )
            if _hybrid_mode:
                self._kv_prefix_cache._no_trim_mode = True
            _bypass_cache = _bypass_cache_for_seed or (
                (not _cache_trimmable) and not _hybrid_mode
            )
            # The KV prefix cache AND the exact-match prompt cache are
            # keyed on token ids (+ model name) but NOT on the active LoRA adapter. The
            # cached KV is computed with whatever adapter was applied when it was stored,
            # so a later request with the SAME prompt prefix but a DIFFERENT adapter (or
            # base) would get a prefix/exact hit and decode on the WRONG adapter's KV —
            # silently serving a model the caller didn't ask for (the "LoRA-fail must
            # not silently serve base" class, reopened through the cache). Until the caches
            # are adapter-keyed, bypass both whenever an adapter is active. (Adapter-keyed
            # reuse is a future enhancement; correctness first.)
            if lora_adapter is not None:
                _bypass_cache = True

            # Prompt cache: try exact-match KV lookup by messages hash.
            # Skip for hybrid — the prompt cache reuses full KV
            # with a trim=1 refeed (and stores post-generation state), which
            # corrupts non-trimmable recurrent layers. Hybrid uses boundary
            # snapshots only.
            _pc_hit = False
            if (
                not _bypass_cache
                and not _hybrid_mode
                and hasattr(self, "_prompt_cache")
                and self._prompt_cache is not None
            ):
                try:
                    from .prompt_cache import compute_messages_hash

                    _pc_hash = compute_messages_hash(
                        [{"role": "user", "content": text}],
                        model=self.model_name,
                    )
                    _pc_entry = self._prompt_cache.lookup(_pc_hash)
                    if _pc_entry is not None and _pc_entry.kv_state is not None:
                        # CRITICAL: snapshot the stored cache before using
                        # it for decode. Decoding mutates KV layers in place
                        # (c.keys = mx.concat(c.keys, new_keys) replaces the
                        # attribute), so a shared reference would corrupt
                        # the stored entry — next lookup gets a cache with
                        # offset advanced past prompt length, ids_to_prefill
                        # becomes empty, and the model continues the prior
                        # response (BUG-A cross-request leakage).
                        # FULL-HIT RE-FEED: the prompt cache stores exactly
                        # prompt-length KV (token_count == len(ids)), so a hit
                        # leaves ids_to_prefill empty and the engine re-feeds
                        # the last prompt token to start decoding. If the cache
                        # still holds that token (trim=0), re-feeding DUPLICATES
                        # it (offset -> len+1, wrong RoPE positions) and the
                        # model echoes the prompt / emits garbage. Trim the last
                        # token from the snapshot so the re-fed ids[-1:] lands at
                        # the correct position. (BUG-A2 full-hit duplication.)
                        _pc_count = int(_pc_entry.token_count)
                        _refeed = 1 if (_pc_count >= len(ids) and len(ids) > 0) else 0
                        if self._kv_prefix_cache is not None:
                            try:
                                cache = self._kv_prefix_cache._snapshot_cache(
                                    _pc_entry.kv_state,
                                    trim=_refeed,
                                )
                            except Exception:
                                logger.debug(
                                    "prompt cache snapshot failed; falling back to direct ref",
                                    exc_info=True,
                                )
                                cache = _pc_entry.kv_state
                                _refeed = 0
                        else:
                            cache = _pc_entry.kv_state
                            _refeed = 0
                        _pc_hit = True
                        cached_tokens = _pc_count - _refeed
                        logger.debug(
                            f"Prompt cache hit: hash={_pc_hash[:12]}, "
                            f"tokens={cached_tokens} (refeed={_refeed})"
                        )
                except Exception:
                    logger.debug("prompt cache lookup failed", exc_info=True)

            # Try KV prefix cache hit (skip when prompt cache already hit —
            # prompt cache provides full KV state which is always better,
            # and skip when seed-bypass is active for reproducibility).
            prefix_cache = None if _bypass_cache else self._kv_prefix_cache
            # Proactive memory pressure eviction (vllm-mlx pattern)
            if prefix_cache is not None and self._mem_pressure_threshold > 0:
                prefix_cache.evict_under_pressure(self._mem_pressure_threshold)
                # Also evict from paged KV manager when enabled
                if self._kv_manager is not None:
                    try:
                        self._kv_manager.memory_pressure_evict(
                            self._mem_pressure_threshold / 100.0
                        )
                    except Exception:
                        logger.debug("paged KV pressure eviction failed", exc_info=True)
            if not _pc_hit:
                try:
                    cached_kv, _, matched = (
                        prefix_cache.get(ids)
                        if prefix_cache is not None
                        else (None, None, 0)
                    )
                except Exception:
                    logger.warning(
                        "KV prefix cache get failed — falling back to full prefill",
                        exc_info=True,
                    )
                    cached_kv, _, matched = None, None, 0
                cache = (
                    cached_kv
                    if cached_kv is not None
                    else _create_prompt_cache_with_quant(
                        model, _req_kv_bits, self._kv_quant_group_size
                    )
                )
                if cached_kv is not None:
                    cached_tokens = matched
                    ids_to_prefill = ids[matched:]
                else:
                    ids_to_prefill = ids
                # Full prefix match: generate_step needs at least one token
                # to start decoding. Re-feed the last cached token.
                if len(ids_to_prefill) == 0 and len(ids) > 0:
                    ids_to_prefill = ids[-1:]
            else:
                # Prompt cache provided full KV — skip prefix cache lookup.
                cached_kv = None
                ids_to_prefill = ids[cached_tokens:]
                if len(ids_to_prefill) == 0 and len(ids) > 0:
                    ids_to_prefill = ids[-1:]

            # Inflight prefix sharing: check for in-flight
            # prefills with matching prefix to share partial KV blocks.
            # Skip when seed-bypass is active so seeded requests stay
            # bit-reproducible regardless of concurrent in-flight prompts.
            # Inflight sharing borrows a donor's live cache and
            # trims it to the matched prefix — not safe for hybrid (non-trimmable
            # recurrent layers). Hybrid reuse goes through boundary snapshots only.
            _inflight_entry = None
            if cached_kv is None and not _bypass_cache and not _hybrid_mode:
                try:
                    from .inflight_prefix_sharing import get_inflight_tracker

                    _tracker = get_inflight_tracker()
                    _inflight_entry = _tracker.find_prefix(
                        [int(t) for t in ids], self.model_name or ""
                    )
                    # : DO NOT borrow the donor's LIVE kv_cache_ref. It was
                    # unsnapshotted+untrimmed: a full-length match yields an empty
                    # ids_to_prefill (generate_step on a (1,0) tensor → IndexError),
                    # and a partial match decodes into the donor's still-growing
                    # cache → wrong RoPE offsets / cross-request corruption. Safe
                    # only when the donor unregisters before any other fast-path
                    # _run — which the now-working engine loop breaks (a long-lived
                    # engine-loop donor stays registered). The snapshot-based
                    # persistent prefix cache already covers safe sharing, so the
                    # live-borrow reuse is disabled (registration below is kept and
                    # is harmless).
                except Exception:
                    logger.debug("inflight prefix lookup failed", exc_info=True)

            # Register our prefill as in-flight for concurrent requests to share
            try:
                from .inflight_prefix_sharing import get_inflight_tracker

                get_inflight_tracker().register(
                    _inflight_req_id,
                    [int(t) for t in ids],
                    cache,
                    self.model_name or "",
                )
            except Exception:
                logger.debug("inflight prefix register failed", exc_info=True)

            # Thinking segment KV lookup — reuse reasoning KV from prior turns.
            # This is a non-functional STUB — it finds a stored segment but never
            # injects `_best.kv_data` into `cache`, so the KV is NOT actually reused. The old
            # code still did `cached_tokens += _best.num_tokens`, which (a) over-reported
            # prompt_tokens_details.cached_tokens by tokens that were never reused and, far
            # worse, (b) on the hybrid path `_capture_hybrid_prefix(start_offset=cached_tokens)`
            # then prefilled `full_ids[start_offset:-1]` from the inflated offset, SKIPPING
            # `num_tokens` real prompt tokens → cache/RoPE misalignment → wrong output. Until
            # the KV is genuinely injected, do not touch cached_tokens; just log the match.
            if self._thinking_store is not None and enable_thinking:
                try:
                    import hashlib as _hl

                    _conv_id = _hl.sha256(str(ids[:16]).encode()).hexdigest()[:16]
                    _conv_segs = self._thinking_store.get_conversation_segments(
                        _conv_id
                    )
                    if _conv_segs:
                        _best = max(_conv_segs, key=lambda s: s.last_accessed)
                        if _best.kv_data is not None:
                            logger.debug(
                                f"Thinking KV match (not yet injected): {_conv_id} → "
                                f"{_best.num_tokens} tokens, hash={_best.step_hash}"
                            )
                except Exception:
                    logger.debug("Thinking segment lookup failed", exc_info=True)

            gen_t0 = time.perf_counter()
            _timeout_deadline = gen_t0 + timeout_seconds
            first = True
            _itl_samples: list[float] = []
            _last_tok_time = 0.0
            _lprocs = logits_processors if logits_processors else None
            spec_prefill_done = False

            # SpecPrefill: sparse prefill for long prompts
            if (
                self._spec_prefill_enabled
                and self._spec_prefill_draft_model is not None
                and len(ids_to_prefill) >= self._spec_prefill_threshold
            ):
                from .spec_prefill import (
                    cleanup_rope,
                    score_tokens,
                    select_chunks,
                    sparse_prefill,
                )

                try:
                    importance = score_tokens(
                        self._spec_prefill_draft_model, ids_to_prefill
                    )
                    selected = select_chunks(
                        importance, keep_pct=self._spec_prefill_keep_rate
                    )
                    logits = sparse_prefill(model, ids_to_prefill, selected, cache)
                    mx.eval(logits)
                    ttft_s = time.perf_counter() - gen_t0
                    first = False
                    first_token = int(sampler(logits[:, -1:, :]))
                    tokens.append(first_token)
                    if logprobs:
                        _lp_logits = logits[:, -1, :].astype(mx.float32)
                        log_probs = _lp_logits - mx.logsumexp(
                            _lp_logits, axis=-1, keepdims=True
                        )
                        log_probs = mx.where(
                            mx.isnan(log_probs),
                            mx.array(-100.0, dtype=log_probs.dtype),
                            log_probs,
                        )
                        tok_lp = float(log_probs[0, first_token])
                        token_logprobs.append(
                            {"token_id": first_token, "logprob": tok_lp}
                        )
                    if first_token in stop_ids:
                        tokens.pop()
                        _stopped_by_stop_id = True
                    else:
                        detokenizer.add_token(first_token)
                        # Check stop suffix on first_token (was missing)
                        if stop_suffixes and any(
                            detokenizer.text.endswith(s) for s in stop_suffixes
                        ):
                            tokens.pop()
                            _stopped_by_suffix = True
                        else:
                            # Check if first_token starts a thinking segment
                            if (
                                think_start_token is not None
                                and first_token == think_start_token
                            ):
                                _in_thinking = True
                                _thinking_tokens = []
                    remaining = max_tokens - 1
                    if (
                        remaining > 0
                        and first_token not in stop_ids
                        and not _stopped_by_suffix
                    ):
                        for token, logits in generate_step(
                            mx.array([first_token]).reshape(1, -1),
                            model,
                            max_tokens=remaining,
                            sampler=sampler,
                            prompt_cache=cache,
                            logits_processors=_lprocs,
                        ):
                            tokens.append(token)
                            if logprobs:
                                _lp_logits = logits.astype(mx.float32)
                                log_probs = _lp_logits - mx.logsumexp(
                                    _lp_logits, axis=-1, keepdims=True
                                )
                                log_probs = mx.where(
                                    mx.isnan(log_probs),
                                    mx.array(-100.0, dtype=log_probs.dtype),
                                    log_probs,
                                )
                                tok_lp = float(log_probs[token])
                                token_logprobs.append(
                                    {"token_id": int(token), "logprob": tok_lp}
                                )
                            if token in stop_ids:
                                tokens.pop()
                                _stopped_by_stop_id = True
                                break
                            detokenizer.add_token(token)
                            if stop_suffixes:
                                if any(
                                    detokenizer.text.endswith(s) for s in stop_suffixes
                                ):
                                    tokens.pop()  # Exclude suffix-triggering token from count
                                    _stopped_by_suffix = True
                                    break
                            # Cancellation check — after append so the token
                            # is not silently lost (consistent with main loop).
                            if _is_cancelled(cancel_event):
                                mx.synchronize()
                                break
                            # Timeout check (was missing — SpecPrefill could run indefinitely)
                            if (
                                len(tokens) % 32 == 0
                                and time.perf_counter() > _timeout_deadline
                            ):
                                logger.warning(
                                    f"SpecPrefill generation timed out after {timeout_seconds}s ({len(tokens)} tokens)"
                                )
                                break
                            # Track thinking segment boundaries BEFORE budget check
                            # (was missing — thinking mode was non-functional in SpecPrefill)
                            if think_start_token is not None:
                                if not _in_thinking and token == think_start_token:
                                    _in_thinking = True
                                    _thinking_tokens = []
                                elif _in_thinking:
                                    _thinking_tokens.append(token)
                                    if token == think_end_token:
                                        _in_thinking = False
                            # Thinking budget enforcement (same as main loop).
                            # Only force-append think_end_token if the current token
                            # is NOT already the natural closing tag (avoids duplicate).
                            if thinking_budget is not None and _in_thinking:
                                thinking_tokens_used += 1
                                if (
                                    thinking_tokens_used >= thinking_budget
                                    and think_end_token is not None
                                ):
                                    _in_thinking = False
                                    if token != think_end_token:
                                        tokens.append(think_end_token)
                                        _thinking_tokens.append(think_end_token)
                                        detokenizer.add_token(think_end_token)
                                    break
                    cleanup_rope(model)
                    spec_prefill_done = True
                except Exception:
                    logger.warning(
                        "SpecPrefill failed, falling back to standard prefill",
                        exc_info=True,
                    )
                    tokens.clear()
                    detokenizer.reset()
                    cache = _create_prompt_cache_with_quant(
                        model, _req_kv_bits, self._kv_quant_group_size
                    )
                    ids_to_prefill = ids
                    first = True
                    _thinking_tokens = []
                    _in_thinking = False
                    thinking_tokens_used = 0

            # HYBRID boundary-snapshot capture. Chunk-prefill
            # the prompt (minus its last token) into `cache`, storing a trim=0
            # snapshot at each block boundary so future shared-prefix requests
            # reuse losslessly. generate_step then prefills only the final token
            # and starts decoding. Any failure falls through to a normal full
            # prefill below (ids_to_prefill unchanged).
            if _hybrid_mode and not spec_prefill_done and len(ids_to_prefill) > 1:
                try:
                    _resident = self._capture_hybrid_prefix(
                        model,
                        ids,
                        cache,
                        self._kv_prefix_cache,
                        start_offset=cached_tokens,
                        block=self._hybrid_prefix_block,
                    )
                    if _resident >= len(ids) - 1:
                        ids_to_prefill = ids[-1:]
                except Exception:
                    logger.warning(
                        "hybrid prefix capture failed — full prefill",
                        exc_info=True,
                    )

            if not spec_prefill_done:
                _timeout_check_interval = 32
                with _wired_limit_ctx(model):
                    for token, logits in generate_step(
                        ids_to_prefill,
                        model,
                        max_tokens=max_tokens,
                        sampler=sampler,
                        prompt_cache=cache,
                        logits_processors=_lprocs,
                        prefill_step_size=_prefill_step_size(),
                    ):
                        if first:
                            ttft_s = time.perf_counter() - gen_t0
                            first = False
                            # Feed cold-prefill throughput to the KV cache so the
                            # SSD tier can auto-gate fast-prefill models.
                            # Only on a cold prefill (cached==0) of a non-trivial
                            # prompt, where ttft_s ≈ pure prefill of ids_to_prefill.
                            if cached_tokens == 0 and ttft_s > 0:
                                try:
                                    _n_pf = (
                                        int(ids_to_prefill.shape[0])
                                        if hasattr(ids_to_prefill, "shape")
                                        else len(ids_to_prefill)
                                    )
                                    if _n_pf >= 256:
                                        self._kv_prefix_cache.note_prefill_tps(
                                            _n_pf / ttft_s
                                        )
                                except Exception:
                                    pass
                            # Prefill complete — remove from progress tracker
                            if _prefill_tracker is not None:
                                _prefill_tracker.update(
                                    _prefill_req_id,
                                    prompt_tokens,
                                    prompt_tokens,
                                    self.model_name or "default",
                                )
                            _last_tok_time = time.perf_counter()
                        else:
                            # ITL tracking
                            _tok_now = time.perf_counter()
                            _itl = _tok_now - _last_tok_time
                            _last_tok_time = _tok_now
                            if _itl > 0 and _itl < 10:
                                _itl_samples.append(_itl)
                        tokens.append(token)
                        # Request-level timeout: check every N tokens
                        if len(tokens) % _timeout_check_interval == 0:
                            if time.perf_counter() > _timeout_deadline:
                                logger.warning(
                                    f"Generation timed out after {timeout_seconds}s ({len(tokens)} tokens)"
                                )
                                break
                        # Progressive KV quantization (C6: keep memory flat during generation)
                        if _req_kv_bits is not None:
                            _progressive_quantize_kv_cache(
                                cache,
                                self._kv_quant_start,
                                self._kv_quant_group_size,
                                _req_kv_bits,
                                len(tokens),
                            )
                        # Compute logprobs BEFORE stop checks — logprobs for stop
                        # tokens are trimmed later via lp_result[:len(tokens)].
                        if logprobs:
                            import mlx.core as mx

                            _lp_logits = logits.astype(mx.float32)
                            log_probs = _lp_logits - mx.logsumexp(
                                _lp_logits, axis=-1, keepdims=True
                            )
                            log_probs = mx.where(
                                mx.isnan(log_probs),
                                mx.array(-100.0, dtype=log_probs.dtype),
                                log_probs,
                            )
                            tok_lp = float(log_probs[token])
                            entry = {"token_id": int(token), "logprob": tok_lp}
                            if top_logprobs and top_logprobs > 0:
                                k = min(top_logprobs, log_probs.shape[0])
                                sorted_idx = mx.argsort(-log_probs)
                                top_k_idx = sorted_idx[:k]
                                entry["top_logprobs"] = [
                                    {
                                        "token_id": int(top_k_idx[j]),
                                        "logprob": float(log_probs[int(top_k_idx[j])]),
                                    }
                                    for j in range(k)
                                ]
                            token_logprobs.append(entry)
                        # Stop ID check — must happen before thinking budget so that
                        # a stop token gets finish_reason="stop" even during thinking.
                        if token in stop_ids:
                            tokens.pop()  # Exclude stop token from output
                            _stopped_by_stop_id = True
                            break
                        # Always add token to detokenizer for incremental state
                        # consistency — previously only added when stop_suffixes
                        # was non-empty, leaving the detokenizer empty and its
                        # state stale when no suffix matching was requested.
                        detokenizer.add_token(token)
                        if stop_suffixes and any(
                            detokenizer.text.endswith(s) for s in stop_suffixes
                        ):
                            tokens.pop()  # Exclude suffix-triggering token from count
                            _stopped_by_suffix = True
                            break
                        # Cancellation check
                        if _is_cancelled(cancel_event):
                            mx.synchronize()
                            break
                        # Track thinking segment boundaries BEFORE budget check
                        # so that a natural </think token is detected first and
                        # the budget enforcement does not append a duplicate.
                        if think_start_token is not None:
                            if not _in_thinking and token == think_start_token:
                                _in_thinking = True
                                _thinking_tokens = []
                                self._lookahead_reasoning.check_thinking_state_text(
                                    "<think"
                                )
                            elif _in_thinking:
                                _thinking_tokens.append(token)
                                if token == think_end_token:
                                    _in_thinking = False
                                    self._lookahead_reasoning.check_thinking_state_text(
                                        "</think"
                                    )

                        # Thinking budget enforcement: cap thinking tokens.
                        # Only force-append think_end_token if the current token
                        # is NOT already the natural closing tag (avoids duplicate).
                        if thinking_budget is not None and _in_thinking:
                            thinking_tokens_used += 1
                            if (
                                thinking_tokens_used >= thinking_budget
                                and think_end_token is not None
                            ):
                                _in_thinking = False
                                # Add forced closing tag to tokens so it appears in
                                # tokenizer.decode(tokens) output (non-streaming path).
                                # Also add to detokenizer so detokenizer.text stays
                                # consistent with the tokens list — previously the
                                # forced tag was only in tokens[], causing
                                # detokenizer.text to miss it when suffix matching
                                # chose the detokenizer path for output assembly.
                                if token != think_end_token:
                                    tokens.append(think_end_token)
                                    _thinking_tokens.append(think_end_token)
                                    detokenizer.add_token(think_end_token)
                                break

            # Cache the completed KV state for future prefix matching
            # Quantize cache layers to save memory (mlx-lm pattern)
            if _req_kv_bits is not None:
                _maybe_quantize_kv_cache(
                    cache,
                    self._kv_quant_start,
                    self._kv_quant_group_size,
                    _req_kv_bits,
                )
            # For HYBRID models the boundary snapshots were
            # already stored DURING prefill (before any generation). The post-
            # generation cache holds prompt+generated tokens, and a hybrid
            # snapshot CANNOT be trimmed back to prompt length (ArraysCache
            # recurrent state isn't sliceable) — adding it would store a state
            # polluted by generated tokens and corrupt future reuse. So skip the
            # trim-based adds for hybrid; the boundary snapshots are correct.
            if prefix_cache is not None and not _hybrid_mode:
                prefix_cache.add(ids, cache)

            # KV cache breakpoints: save prefix entries at Anthropic
            # cache_control positions so future requests can reuse the
            # KV state up to each breakpoint (multi-turn speedup).
            if (
                _kv_breakpoint_token_positions
                and prefix_cache is not None
                and not _hybrid_mode
            ):
                from .kv_prefix_cache import cache_length as _cache_len_bp

                for bp_pos in _kv_breakpoint_token_positions:
                    if bp_pos < len(ids) and bp_pos >= 32:
                        bp_tokens = ids[:bp_pos]
                        # Trim relative to the LIVE cache length,
                        # not len(ids). This runs AFTER generation, so `cache` holds
                        # len(ids)+num_generated tokens; trim=len(ids)-bp_pos retained
                        # bp_pos+num_generated tokens under a bp_pos-token key → the entry
                        # carried the prior request's generated continuation and corrupted
                        # a future prefix-hit (BUG-A cross-request leak). The main
                        # prefix_cache.add(ids, cache) self-corrects via cache_len-prompt_len;
                        # this hand-computed snapshot bypassed that.
                        trim_count = max(0, _cache_len_bp(cache) - bp_pos)
                        try:
                            bp_cache = prefix_cache._snapshot_cache(
                                cache, trim=trim_count
                            )
                            prefix_cache.add(bp_tokens, bp_cache)
                        except Exception:
                            logger.debug(
                                "KV breakpoint prefix add failed at pos %d",
                                bp_pos,
                                exc_info=True,
                            )

            # Prompt cache: store KV state for exact-match reuse.
            # CRITICAL: store the cache trimmed to prompt-length and report
            # token_count=len(ids). Storing prompt+completion length means
            # the next request with same prompt sees cached_tokens > prompt
            # length, ids_to_prefill becomes empty, the engine re-feeds the
            # last prompt token against KV state that contains *prior*
            # completion tokens, and produces gibberish that continues the
            # earlier response (BUG-A cross-request leakage).
            if (
                not _pc_hit
                and not _bypass_cache
                and not _hybrid_mode
                and hasattr(self, "_prompt_cache")
                and self._prompt_cache is not None
            ):
                try:
                    from .kv_prefix_cache import cache_length as _cache_len
                    from .prompt_cache import compute_messages_hash

                    _pc_hash = compute_messages_hash(
                        [{"role": "user", "content": text}],
                        model=self.model_name,
                    )
                    _prompt_only_len = int(len(ids))
                    _full_cache_len = int(_cache_len(cache))
                    _trim = max(0, _full_cache_len - _prompt_only_len)
                    if _trim > 0 and prefix_cache is not None:
                        # Reuse prefix_cache's snapshot+trim (it knows how to
                        # detach and slice KV layers safely).
                        _trimmed_cache = prefix_cache._snapshot_cache(cache, trim=_trim)
                    else:
                        _trimmed_cache = cache
                    self._prompt_cache.store(
                        _pc_hash,
                        _trimmed_cache,
                        token_count=_prompt_only_len,
                    )
                except Exception:
                    logger.debug("prompt cache store failed", exc_info=True)

            # Store thinking segment KV for future reuse (if enabled)
            if _thinking_tokens and self._thinking_store is not None:
                try:
                    import hashlib as _hl

                    conv_id = _hl.sha256(str(ids[:16]).encode()).hexdigest()[:16]
                    # Snapshot the cache so the thinking store doesn't hold a
                    # reference to the same mutable list as prefix_cache.
                    _thinking_kv = [c for c in cache] if cache else None
                    self._thinking_store.store(
                        conversation_id=conv_id,
                        thinking_tokens=_thinking_tokens,
                        context_tokens=[int(t) for t in ids],
                        kv_data=_thinking_kv,
                    )
                except Exception:
                    logger.debug("Thinking segment store failed", exc_info=True)

            # Finalize detokenizer to flush any remaining partial UTF-8 bytes
            # before assembling final output text.
            try:
                detokenizer.finalize()
            except Exception:
                logger.debug("detokenizer finalize failed in fast path", exc_info=True)

            # When stop_suffix matching is active, use detokenizer text for
            # output because tokenizer.decode(tokens) may contain a partial
            # suffix that leaked across token boundaries. The detokenizer
            # has the complete incremental text including the suffix, which
            # we trim below. When no suffix matching, tokenizer.decode is
            # authoritative and avoids detokenizer state issues.
            if _stopped_by_suffix and stop_suffixes:
                output_text = detokenizer.text
                # Trim the matched suffix from detokenizer text
                for s in stop_suffixes:
                    if output_text.endswith(s):
                        output_text = output_text[: -len(s)]
                        break
                output_text = _clean_special_tokens(output_text)
            else:
                output_text = tokenizer.decode(tokens, skip_special_tokens=True)
            # GUARANTEED stop-suffix truncation. The per-token
            # `detokenizer.text.endswith(s)` check misses stops that don't land
            # exactly at the (space-padded, one-token-lagging) detokenizer
            # boundary — e.g. bare words like "banana"/"END"/"User:" — so the stop
            # string leaked into the output with finish_reason="stop" coming from
            # natural EOS. Truncate the final text at the FIRST occurrence of any
            # stop string (OpenAI semantics) so it can never leak.
            if stop_suffixes and output_text:
                _cut = len(output_text)
                for s in stop_suffixes:
                    if s:
                        _i = output_text.find(s)
                        if _i != -1:
                            _cut = min(_cut, _i)
                if _cut < len(output_text):
                    output_text = output_text[:_cut]
                    _stopped_by_suffix = True
            mx.synchronize()

            # Unregister from inflight prefix tracker
            try:
                from .inflight_prefix_sharing import get_inflight_tracker

                get_inflight_tracker().unregister(_inflight_req_id)
            except Exception:
                logger.debug("inflight prefix unregister failed", exc_info=True)

            return (
                tokens,
                output_text,
                token_logprobs,
                ttft_s,
                cached_tokens,
                _stopped_by_suffix,
                _stopped_by_stop_id,
                _itl_samples,
                _thinking_tokens,
            )

        def _run_with_lora():
            # LoRA concurrency keystone: acquire+apply / release+restore on the
            # EXECUTOR thread so the whole adapter lifecycle is serialized with this
            # request's generate_step — and therefore with every other request's, since the
            # executor is max_workers=1. This removes the cross-thread race where an
            # event-loop _restore_base mutated the shared model while a different request's
            # generation read it. (Gateway-side apply is deferred for self-managing engines.)
            _lora_applied = False
            if lora_adapter and getattr(self, "_lora_manager", None) is not None:
                try:
                    _lora_applied = self._lora_manager.acquire_adapter(lora_adapter)
                except Exception as _lora_err:
                    # A registered adapter can still fail to apply at
                    # load time (arch mismatch / bad weights / max_loras exhausted).
                    # Fall-through would silently serve BASE-model output with a 200
                    # for a request that explicitly asked for the fine-tuned model —
                    # raise so it fails loudly instead.
                    raise RuntimeError(
                        f"LoRA adapter '{lora_adapter}' could not be applied"
                    ) from _lora_err
                if not _lora_applied:
                    raise RuntimeError(
                        f"LoRA adapter '{lora_adapter}' could not be applied"
                    )
            try:
                return _run()
            finally:
                if _lora_applied and getattr(self, "_lora_manager", None) is not None:
                    try:
                        self._lora_manager.release_adapter(lora_adapter)
                    except Exception:
                        logger.debug(
                            "LoRA release failed (fast-path executor)", exc_info=True
                        )

        from .mlx_executor import get_mlx_executor

        executor = get_mlx_executor()
        loop = asyncio.get_running_loop()
        _fp_lock = getattr(self, "_fast_path_lock", None)
        if _fp_lock is not None:
            with _fp_lock:
                self._active_fast_path_count += 1
        try:
            try:
                (
                    tokens,
                    output_text,
                    token_logprobs,
                    ttft_s,
                    cached_tokens,
                    _stopped_by_suffix,
                    _stopped_by_stop_id,
                    _itl_samples,
                    _thinking_tokens,
                ) = await loop.run_in_executor(executor, _run_with_lora)
            except MemoryError:
                logger.warning(
                    "OOM during generation — returning memory_limit finish reason"
                )
                try:
                    from .inflight_prefix_sharing import get_inflight_tracker

                    get_inflight_tracker().unregister(_inflight_req_id)
                except Exception:
                    logger.debug(
                        "inflight prefix unregister failed in OOM handler",
                        exc_info=True,
                    )
                # Clear Metal buffers left behind by the OOM
                try:
                    import mlx.core as _mx

                    await loop.run_in_executor(
                        executor, lambda: (_mx.synchronize(), _mx.clear_cache())
                    )
                except Exception:
                    logger.debug("GPU cache cleanup failed after OOM", exc_info=True)
                return GenerationOutput(
                    finished=True,
                    finish_reason="memory_limit",
                    prompt_tokens=prompt_tokens,
                    completion_tokens=0,
                    error="OOM during generation",
                    ttft_ms=0.0,
                    cached_tokens=0,
                )
            except RuntimeError as e:
                if "memory" in str(e).lower() or "out of" in str(e).lower():
                    logger.warning(f"MLX OOM during generation: {e}")
                    try:
                        from .inflight_prefix_sharing import get_inflight_tracker

                        get_inflight_tracker().unregister(_inflight_req_id)
                    except Exception:
                        logger.debug(
                            "inflight prefix unregister failed in OOM handler",
                            exc_info=True,
                        )
                    # Clear Metal buffers left behind by the OOM
                    try:
                        import mlx.core as _mx

                        await loop.run_in_executor(
                            executor, lambda: (_mx.synchronize(), _mx.clear_cache())
                        )
                    except Exception:
                        logger.debug(
                            "GPU cache cleanup failed after OOM (RuntimeError path)",
                            exc_info=True,
                        )
                    return GenerationOutput(
                        finished=True,
                        finish_reason="memory_limit",
                        prompt_tokens=prompt_tokens,
                        completion_tokens=0,
                        error=str(e),
                        ttft_ms=0.0,
                        cached_tokens=0,
                    )
                try:
                    from .inflight_prefix_sharing import get_inflight_tracker

                    get_inflight_tracker().unregister(_inflight_req_id)
                except Exception:
                    logger.debug(
                        "inflight prefix unregister failed in error handler",
                        exc_info=True,
                    )
                # Return error output for non-OOM RuntimeError too (e.g. shape
                # mismatch, unsupported op) instead of propagating to caller.
                return GenerationOutput(
                    finished=True,
                    finish_reason="error",
                    prompt_tokens=prompt_tokens,
                    completion_tokens=0,
                    error=f"RuntimeError during generation: {e}",
                    ttft_ms=0.0,
                    cached_tokens=0,
                )
            except Exception as e:
                logger.error(f"Unexpected error during generation: {e}", exc_info=True)
                try:
                    from .inflight_prefix_sharing import get_inflight_tracker

                    get_inflight_tracker().unregister(_inflight_req_id)
                except Exception:
                    logger.debug(
                        "inflight prefix unregister failed in error handler",
                        exc_info=True,
                    )
                # Return an error GenerationOutput instead of propagating the
                # exception to the caller (which expects GenerationOutput, not
                # an exception). Previously this re-raised, causing unhandled
                # exceptions in the gateway handler.
                return GenerationOutput(
                    finished=True,
                    finish_reason="error",
                    prompt_tokens=prompt_tokens,
                    completion_tokens=0,
                    error=f"Generation failed: {e}",
                    ttft_ms=0.0,
                    cached_tokens=0,
                )

            # Stop-suffix truncation can leave completion_tokens (len(tokens))
            # counting tokens PAST the returned text — the per-token stop check missed
            # (stop not on a detokenizer boundary), generation ran to max_tokens, then
            # the text was find()-truncated at the stop. Re-encode the (truncated,
            # pre-reasoning-parse → TOTAL) text and report it when it's materially
            # fewer tokens, so we don't over-bill the discarded tail. None → len(tokens).
            _ct_override = None
            if _stopped_by_suffix and stop_suffixes and output_text is not None:
                try:
                    _ct_re = len(
                        tokenizer.encode(output_text, add_special_tokens=False)
                    )
                    if _ct_re < len(tokens):
                        _ct_override = _ct_re
                except Exception:
                    _ct_override = None

            # Decode token strings for logprobs
            lp_result = None
            if logprobs and token_logprobs:
                for lp_entry in token_logprobs:
                    tid = lp_entry["token_id"]
                    try:
                        lp_entry["token"] = tokenizer.decode([tid])
                        # Raw token bytes (NOT the decoded-string bytes, which
                        # are the U+FFFD replacement for a split multi-byte char).
                        from .text_utils import token_id_to_bytes

                        lp_entry["bytes"] = token_id_to_bytes(
                            tokenizer, tid, lp_entry["token"]
                        )
                    except Exception:
                        logger.debug("logprob token decode failed", exc_info=True)
                        lp_entry["token"] = ""
                        lp_entry["bytes"] = []
                    if "top_logprobs" in lp_entry:
                        for tlp in lp_entry["top_logprobs"]:
                            try:
                                tlp["token"] = tokenizer.decode([tlp["token_id"]])
                                from .text_utils import token_id_to_bytes

                                tlp["bytes"] = token_id_to_bytes(
                                    tokenizer, tlp["token_id"], tlp["token"]
                                )
                            except Exception:
                                logger.debug(
                                    "top_logprob token decode failed", exc_info=True
                                )
                                tlp["token"] = ""
                                tlp["bytes"] = []
                lp_result = token_logprobs

            output_text = _clean_special_tokens(output_text)

            # Trim stop suffix from output text when matched during generation
            if _stopped_by_suffix and stop_suffixes:
                for s in stop_suffixes:
                    if output_text.endswith(s):
                        output_text = output_text[: -len(s)]
                        break

            # Determine finish_reason.
            # Priority: cancel > stop (suffix or stop_id) > length
            # When cancel_event or timeout triggers, the loop breaks without
            # setting _stopped_by_suffix or _stopped_by_stop_id, so those
            # tokens correctly show up as "stop" only when genuinely stopped.
            _cancelled = _is_cancelled(cancel_event)
            if _cancelled or _stopped_by_suffix or _stopped_by_stop_id:
                finish_reason = "stop"
            else:
                finish_reason = "length"

            # BUG FIX: When the first token is a stop_id (SpecPrefill path),
            # it is popped from `tokens` but its logprob entry remains in
            # `token_logprobs`. Trim the stale entry so logprobs count matches
            # `completion_tokens`.
            if lp_result is not None:
                lp_result = lp_result[: len(tokens)]

            # Record TTFT + ITL in Prometheus
            _ttft_ms_val = round(ttft_s * 1000, 1)
            if ttft_s > 0:
                try:
                    from yunshu_gateway.middleware.prometheus_exporter import (
                        get_prometheus_metrics,
                    )

                    pm = get_prometheus_metrics()
                    _ml = {"model_id": self.model_label}
                    pm.observe_histogram("ttft_seconds", ttft_s, labels=_ml)
                    if cached_tokens > 0:
                        pm.inc_counter("kv_prefix_cache_hits", labels=_ml)
                    else:
                        pm.inc_counter("kv_prefix_cache_misses", labels=_ml)
                    # ITL: record individual inter-token latency samples into histogram
                    if _itl_samples:
                        for _itl_sample in _itl_samples:
                            pm.observe_histogram("itl_seconds", _itl_sample, labels=_ml)
                except Exception:
                    logger.debug("TTFT/ITL prometheus recording failed", exc_info=True)

            # Record in ServerMetrics (consistency with engine loop path)
            try:
                from .server_metrics import get_server_metrics

                _sm = get_server_metrics()
                _total_gen_s = sum(_itl_samples) + ttft_s if _itl_samples else ttft_s
                _sm.record_request_complete(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=len(tokens),
                    cached_tokens=cached_tokens,
                    prefill_duration=ttft_s,
                    generation_duration=_total_gen_s,
                    model_id=self.model_name,
                )
                if _itl_samples:
                    for _itl in _itl_samples:
                        _sm.record_itl(_itl)
            except Exception:
                logger.debug(
                    "ServerMetrics recording failed in fast path", exc_info=True
                )

            self._total_reasoning_tokens += len(_thinking_tokens)

            # Reasoning parser: supplement token-level tracking with model-specific
            # reasoning extraction when thinking tokens were not explicitly tracked
            _reasoning_tok = len(_thinking_tokens)
            if output_text:
                try:
                    from .reasoning_parser import get_reasoning_parser

                    rp = get_reasoning_parser(self.model_name)
                    rp_out = rp.parse(output_text)
                    if rp_out.reasoning:
                        _reasoning_tok = rp_out.reasoning_tokens
                        if rp_out.content != output_text:
                            output_text = rp_out.content
                except Exception:
                    logger.debug("reasoning_parser failed in fast path", exc_info=True)

            return GenerationOutput(
                text=output_text,
                new_text=output_text,
                prompt_tokens=prompt_tokens,
                completion_tokens=_ct_override
                if _ct_override is not None
                else len(tokens),
                finished=True,
                finish_reason=finish_reason,
                cached_tokens=cached_tokens,
                logprobs=lp_result,
                ttft_ms=_ttft_ms_val,
                reasoning_tokens=_reasoning_tok,
                # A user stop sequence fired (vs natural EOS) — both map to "stop".
                stopped_by_stop_sequence=bool(_stopped_by_suffix),
            )
        finally:
            # LoRA release+restore now happens inside _run_with_lora on the executor
            # thread — not here on the event loop.
            _fp_lock = getattr(self, "_fast_path_lock", None)
            if _fp_lock is not None:
                with _fp_lock:
                    self._active_fast_path_count -= 1

    async def stream_generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 1.0,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        logit_bias: dict[int, float] | None = None,
        stop: list[str] | None = None,
        stop_token_ids: list[int] | None = None,
        seed: int | None = None,
        json_schema: dict | str | None = None,
        spec_decode: bool = False,
        use_engine_loop: bool | None = None,
        enable_thinking: bool | None = None,
        thinking_budget: int | None = None,
        reasoning_effort: str | None = None,
        xtc_probability: float = 0.0,
        xtc_threshold: float = 0.0,
        priority: int = 0,
        logprobs: bool | int = False,
        top_logprobs: int | None = None,
        logits_processors: list | None = None,
        cancel_event: asyncio.Event | None = None,
        timeout_seconds: float | None = None,
        images: list | None = None,
        lora_adapter: str | None = None,
        kv_cache_breakpoints: list[int] | None = None,
        min_tokens: int = 0,
        ignore_eos: bool = False,
        suppress_tokens: list[int] | None = None,
    ) -> AsyncIterator[GenerationOutput]:
        """Streaming text generation.

        Default uses fast path (direct generate_step on executor) for single
        requests. Set use_engine_loop=True for continuous batching path.
        If use_engine_loop is None, uses YUNSHU_ENGINE_LOOP env var.

        When cancel_event is provided (e.g. from gateway disconnect detection),
        it is checked alongside the engine's internal cancel event to allow
        cooperative cancellation from the HTTP layer.
        """
        if not self._loaded:
            await self.start()

        _use_engine_loop = self._should_use_engine_loop(use_engine_loop)
        # : spec_decode falls back to the fast path — see the non-streaming
        # path for the full rationale (fast-path spec routes are not currently
        # lossless and give no speedup on Apple Silicon).
        if spec_decode:
            logger.debug(
                "spec_decode requested — using standard streaming generation "
                "(fast-path spec routes are not currently lossless)"
            )
            spec_decode = False
        # Resolve reasoning_effort → thinking_budget if not explicitly set
        if thinking_budget is None and reasoning_effort is not None:
            thinking_budget = _REASONING_EFFORT_MAP.get(reasoning_effort, 8192)
            if enable_thinking is None:
                enable_thinking = True

        # Gemma-4 default: its chat template enables thinking by default but
        # emits inline `thought` tokens that don't auto-stop, producing output
        # like "4thought\nThinking Process: …" for simple prompts. Default
        # enable_thinking=False for gemma-4 unless caller passed reasoning_effort.
        if enable_thinking is None and reasoning_effort is None:
            _mn = getattr(self, "model_name", None)
            if isinstance(_mn, str) and "gemma-4" in _mn.lower():
                enable_thinking = False

        # Early return: max_tokens <= 0 produces no output (matches generate() guard)
        if max_tokens <= 0:
            yield GenerationOutput(
                text="",
                new_text="",
                prompt_tokens=0,
                completion_tokens=0,
                finished=True,
                finish_reason="length",
            )
            return

        # Memory guard preflight check
        guard_rejection = self._check_memory_guard(prompt, max_tokens)
        if guard_rejection is not None:
            yield guard_rejection
            return

        # Register with request tracker for cancellation support (all paths)
        import uuid as _uuid

        _stream_req_id = f"stream-{_uuid.uuid4().hex[:8]}"
        try:
            from .request_tracker import get_request_tracker

            _tracker = get_request_tracker()
            _active_gen = _tracker.register(_stream_req_id, self.model_name or "")
            _cancel_event = _active_gen.cancel_event
        except Exception:
            logger.debug("request tracker registration failed", exc_info=True)
            _cancel_event = None
            _tracker = None

        # If the gateway passes an external cancel_event, wrap both events
        # so that checking .is_set() on the wrapper detects either source.
        if cancel_event is not None:
            _internal = _cancel_event
            _external = cancel_event

            class _CompositeCancelEvent:
                """Proxy that returns True if either the internal or external event is set.

                Thread-safe: reads asyncio.Event._value directly (GIL-protected bool)
                instead of calling .is_set() which is not safe from executor threads.
                """

                def is_set(self):
                    if _internal is not None:
                        # asyncio.Event: read _value (GIL-protected bool)
                        if hasattr(_internal, "_value"):
                            if _internal._value:
                                return True
                        elif _internal.is_set():
                            return True
                    return _is_cancelled(_external)

            _cancel_event = _CompositeCancelEvent()

        # Speculative decoding path (Phase 4)
        if (
            spec_decode
            and not logprobs
            and not _use_engine_loop
            and self._spec_enabled
            and self._spec_decoder is not None
        ):
            try:
                async for output in self._stream_generate_speculative(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    min_p=min_p,
                    repetition_penalty=repetition_penalty,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                    logit_bias=logit_bias,
                    logprobs=logprobs,
                    top_logprobs=top_logprobs,
                    stop=stop,
                    stop_token_ids=stop_token_ids,
                    seed=seed,
                    enable_thinking=enable_thinking,
                    thinking_budget=thinking_budget,
                    xtc_probability=xtc_probability,
                    xtc_threshold=xtc_threshold,
                    json_schema=json_schema,
                    cancel_event=_cancel_event,
                    logits_processors=logits_processors,
                    lora_adapter=lora_adapter,
                ):
                    yield output
            finally:
                if _tracker is not None:
                    try:
                        _tracker.unregister(_stream_req_id)
                    except Exception:
                        logger.debug("request tracker cleanup failed", exc_info=True)
            return

        # MTP speculative decoding streaming (built-in mlx-lm MTPDecoder path).
        # NOTE: `spec_decode` is forced False for streaming at the top
        # of this method, so this branch is currently UNREACHABLE — _stream_generate_mtp
        # is dead in the streaming flow today. The live opt-in MTP backend
        # (YUNSHU_MTP=1, mlx-vlm) is a SEPARATE path: stream_chat delegates to it
        # and emits a single chunk with `stop` trimmed post-hoc, so it has no
        # multi-token-stop streaming leak. The hold-back fix in _stream_generate_mtp
        # is defensive — correct if this branch is ever re-enabled.
        if (
            spec_decode
            and not logprobs
            and self._mtp_decoder is not None
            and not _use_engine_loop
        ):
            try:
                async for output in self._stream_generate_mtp(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    min_p=min_p,
                    repetition_penalty=repetition_penalty,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                    logit_bias=logit_bias,
                    logprobs=logprobs,
                    top_logprobs=top_logprobs,
                    stop=stop,
                    stop_token_ids=stop_token_ids,
                    seed=seed,
                    cancel_event=_cancel_event,
                    enable_thinking=enable_thinking,
                    thinking_budget=thinking_budget,
                    timeout_seconds=timeout_seconds or 300.0,
                    json_schema=json_schema,
                    xtc_probability=xtc_probability,
                    xtc_threshold=xtc_threshold,
                    logits_processors=logits_processors,
                    lora_adapter=lora_adapter,
                    kv_cache_breakpoints=kv_cache_breakpoints,
                    min_tokens=min_tokens,
                    ignore_eos=ignore_eos,
                    suppress_tokens=suppress_tokens,
                ):
                    yield output
            finally:
                if _tracker is not None:
                    try:
                        _tracker.unregister(_stream_req_id)
                    except Exception:
                        logger.debug("request tracker cleanup failed", exc_info=True)
            return

        # N-gram speculative decoding streaming (model-free). The
        # real impl `_stream_generate_ngram_spec` EXISTS but is BROKEN — it
        # early-terminates (live: a "count to 8" prompt streamed only "1 " vs the
        # correct full sequence). Streaming n-gram spec is also bandwidth-bound
        # (≤baseline on Apple Silicon), so rather than serve truncated output we
        # fall back to CORRECT plain generation. (The old comment falsely claimed
        # the method was "not yet implemented"; the truth is it's implemented but
        # buggy — fixing it is low-value, deferred.) Output is correct; only the
        # spec speedup is forgone for streaming.
        if (
            spec_decode
            and not logprobs
            and self._ngram_proposer is not None
            and not _use_engine_loop
        ):
            try:
                async for output in self._stream_generate_fast(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    min_p=min_p,
                    repetition_penalty=repetition_penalty,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                    logit_bias=logit_bias,
                    stop=stop,
                    stop_token_ids=stop_token_ids,
                    seed=seed,
                    json_schema=json_schema,
                    logprobs=logprobs,
                    top_logprobs=top_logprobs,
                    xtc_probability=xtc_probability,
                    xtc_threshold=xtc_threshold,
                    cancel_event=_cancel_event,
                    timeout_seconds=timeout_seconds or 300.0,
                    logits_processors=logits_processors,
                    enable_thinking=enable_thinking,
                    thinking_budget=thinking_budget,
                    lora_adapter=lora_adapter,
                    min_tokens=min_tokens,
                    ignore_eos=ignore_eos,
                    suppress_tokens=suppress_tokens,
                ):
                    yield output
            finally:
                if _tracker is not None:
                    try:
                        _tracker.unregister(_stream_req_id)
                    except Exception:
                        logger.debug("request tracker cleanup failed", exc_info=True)
            return

        # Fast path: bypass EngineCore for single-request streaming
        if not _use_engine_loop:
            try:
                async for output in self._stream_generate_fast(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    min_p=min_p,
                    repetition_penalty=repetition_penalty,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                    logit_bias=logit_bias,
                    stop=stop,
                    stop_token_ids=stop_token_ids,
                    seed=seed,
                    enable_thinking=enable_thinking,
                    thinking_budget=thinking_budget,
                    xtc_probability=xtc_probability,
                    xtc_threshold=xtc_threshold,
                    json_schema=json_schema,
                    cancel_event=_cancel_event,
                    logprobs=bool(logprobs),
                    top_logprobs=top_logprobs,
                    logits_processors=logits_processors,
                    priority=priority,
                    timeout_seconds=timeout_seconds or 300.0,
                    lora_adapter=lora_adapter,
                    kv_cache_breakpoints=kv_cache_breakpoints,
                    min_tokens=min_tokens,
                    ignore_eos=ignore_eos,
                    suppress_tokens=suppress_tokens,
                ):
                    yield output
            finally:
                if _tracker is not None:
                    try:
                        _tracker.unregister(_stream_req_id)
                    except Exception:
                        logger.debug("request tracker cleanup failed", exc_info=True)
            return

        # Engine loop path: continuous batching with scheduler
        await self._ensure_engine_core()
        _stream_t0 = time.perf_counter()
        request_id = await self._engine_core.add_request(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            logit_bias=logit_bias,
            stop=stop,
            stop_token_ids=stop_token_ids,
            seed=seed,
            json_schema=json_schema,
            enable_thinking=enable_thinking,
            thinking_budget=thinking_budget,
            priority=priority,
            logprobs=bool(logprobs),
            top_logprobs=top_logprobs,
            xtc_probability=xtc_probability,
            xtc_threshold=xtc_threshold,
            reasoning_effort=reasoning_effort,
            logits_processors=logits_processors,
            images=images,
            lora_adapter=lora_adapter,
        )

        finished_normally = False
        _first_token = True
        _stream_ttft_ms = 0.0
        try:
            async for output in self._engine_core.stream_outputs(
                request_id, cancel_event=_cancel_event
            ):
                # Check cancel event (gateway disconnect or internal cancel)
                if _cancel_event is not None and _cancel_event.is_set():
                    logger.debug(
                        f"Cancel event triggered during streaming: {request_id}"
                    )
                    # Yield terminal stop chunk so consumer sees finished=True
                    yield GenerationOutput(
                        text="",
                        new_text="",
                        prompt_tokens=getattr(output, "prompt_tokens", 0)
                        if hasattr(output, "prompt_tokens")
                        else 0,
                        completion_tokens=getattr(output, "completion_tokens", 0)
                        if hasattr(output, "completion_tokens")
                        else 0,
                        finished=True,
                        finish_reason="stop",
                        ttft_ms=_stream_ttft_ms,
                        cached_tokens=getattr(output, "cached_tokens", 0),
                        reasoning_tokens=getattr(output, "reasoning_tokens", 0),
                    )
                    break
                cleaned = _clean_special_tokens(output.new_text)
                finish_reason = output.finish_reason
                if finish_reason == "memory_exceeded":
                    finish_reason = "memory_limit"
                # Guard: finish_reason must never be None when finished=True
                if output.finished and finish_reason is None:
                    finish_reason = "stop"
                # Compute TTFT on first streamed output
                _ttft_ms = 0.0
                if _first_token:
                    _stream_ttft_ms = round(
                        (time.perf_counter() - _stream_t0) * 1000, 1
                    )
                    _ttft_ms = _stream_ttft_ms
                    _first_token = False
                    # Record TTFT in Prometheus (consistency with fast path)
                    try:
                        from yunshu_gateway.middleware.prometheus_exporter import (
                            get_prometheus_metrics,
                        )

                        pm = get_prometheus_metrics()
                        pm.observe_histogram(
                            "ttft_seconds",
                            _stream_ttft_ms / 1000.0,
                            labels={"model_id": self.model_label},
                        )
                    except Exception:
                        logger.debug(
                            "engine loop streaming TTFT prometheus recording failed",
                            exc_info=True,
                        )
                gen_output = GenerationOutput(
                    text=_clean_special_tokens(output.output_text),
                    new_text=cleaned,
                    prompt_tokens=output.prompt_tokens,
                    completion_tokens=output.completion_tokens,
                    finished=output.finished,
                    finish_reason=finish_reason,
                    reasoning_tokens=getattr(output, "reasoning_tokens", 0),
                    cached_tokens=getattr(output, "cached_tokens", 0),
                    logprobs=getattr(output, "logprobs", None),
                    ttft_ms=_ttft_ms,
                    current_state=getattr(output, "current_state", None),
                    error=getattr(output, "error", None),
                    prefill_progress=getattr(output, "prefill_progress", None),
                )
                if output.finished:
                    finished_normally = True
                yield gen_output
        except GeneratorExit:
            logger.debug(f"Client disconnected during streaming: {request_id}")
        finally:
            if not finished_normally:
                try:
                    await self._engine_core.abort_request(request_id)
                except Exception:
                    logger.debug("stream abort failed", exc_info=True)
            if _tracker is not None:
                try:
                    _tracker.unregister(_stream_req_id)
                except Exception:
                    logger.debug("request tracker cleanup failed", exc_info=True)

    async def _stream_generate_fast(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 1.0,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        logit_bias: dict[int, float] | None = None,
        stop: list[str] | None = None,
        stop_token_ids: list[int] | None = None,
        seed: int | None = None,
        enable_thinking: bool | None = None,
        thinking_budget: int | None = None,
        xtc_probability: float = 0.0,
        xtc_threshold: float = 0.0,
        json_schema: dict | str | None = None,
        cancel_event: asyncio.Event | None = None,
        logprobs: bool = False,
        top_logprobs: int | None = None,
        logits_processors: list | None = None,
        priority: int = 0,
        timeout_seconds: float = 300.0,
        lora_adapter: str | None = None,
        kv_cache_breakpoints: list[int] | None = None,
        min_tokens: int = 0,
        ignore_eos: bool = False,
        suppress_tokens: list[int] | None = None,
    ) -> AsyncIterator[GenerationOutput]:
        """Fast streaming: runs generate_step on executor, yields via asyncio.Queue.

        When YUNSHU_STREAMING_PIPELINE=1, wraps generation with:
        - StreamingBackpressureController to prevent OOM on slow clients
        - TokenPipeline for GPU/CPU overlap (future: full pipeline)
        - PrefetchSampler for sampling plan pre-computation (future: per-step)
        """
        from mlx_lm.generate import generate_step
        from mlx_lm.sample_utils import make_sampler

        # Streaming optimizer components
        from .streaming_optimizer import StreamingBackpressureController

        _backpressure = StreamingBackpressureController(max_queue_size=100)

        # TokenPipeline for GPU/CPU overlap — activated via YUNSHU_STREAMING_PIPELINE=1
        _pipeline = None
        if self._streaming_pipeline_enabled:
            from .streaming_optimizer import PipelineConfig, TokenPipeline

            _pipeline = TokenPipeline(PipelineConfig(enable_overlap=True))
            _pipeline.start_pipeline(request=None)
            logger.debug("TokenPipeline active for streaming fast path")

        tokenizer = self._tokenizer
        model = self._model

        # Model-specific preprocessing
        if self._preprocessor_registry is not None:
            try:
                model_config = {"model_type": self.model_name or ""}
                if hasattr(model, "config") and hasattr(model.config, "model_type"):
                    model_config["model_type"] = model.config.model_type
                preprocessor = self._preprocessor_registry.detect(model_config)
                if preprocessor is not None:
                    processed = preprocessor.preprocess(prompt, tokenizer=tokenizer)
                    if processed.token_ids:
                        prompt = processed.token_ids
            except Exception:
                logger.debug("model preprocessor failed in streaming", exc_info=True)

        # ── Pre-encoding context window truncation (message-level) ──
        if isinstance(prompt, list) and prompt and isinstance(prompt[0], dict):
            try:
                _max_ctx_pre = getattr(model, "max_seq_len", None)
                if _max_ctx_pre is None:
                    _max_ctx_pre = getattr(
                        getattr(model, "config", None), "max_seq_len", None
                    ) or getattr(getattr(model, "args", None), "max_seq_len", None)
                if _max_ctx_pre and _max_ctx_pre > 0:
                    _thinking_overhead = (
                        thinking_budget if (thinking_budget and enable_thinking) else 0
                    )
                    _generation_budget = max_tokens + _thinking_overhead
                    _est_tokens = sum(
                        len(str(m.get("content", ""))) // 4 + 4 for m in prompt
                    )
                    if _est_tokens + _generation_budget > _max_ctx_pre:
                        from .context_window import ContextWindowManager

                        ctx_mgr = ContextWindowManager(
                            token_counter=lambda text: len(tokenizer.encode(text)),
                        )
                        result = ctx_mgr.compute_truncation(
                            messages=prompt,
                            max_tokens=_max_ctx_pre - _generation_budget,
                            strategy="importance_aware",
                        )
                        prompt = result.messages
                        logger.debug(
                            "Streaming fast path pre-encode truncation: estimated %d → %d tokens",
                            _est_tokens,
                            result.truncated_token_count,
                        )
            except Exception:
                logger.debug(
                    "context window truncation skipped in streaming fast path",
                    exc_info=True,
                )

        if isinstance(prompt, list) and prompt and isinstance(prompt[0], dict):
            # Route through _apply_chat_template (NOT raw) — same reason as the
            # non-streaming fast path: tool_calls.arguments str→dict normalization,
            # the family message adapter, dangling-<think> closing, and the
            # enable_thinking-unsupported retry. The default streaming path otherwise
            # broke multi-turn tool conversations / Mistral-Gemma role rules.
            prompt = self._apply_chat_template(prompt, enable_thinking=enable_thinking)

        input_ids = self._encode_prompt(tokenizer, prompt)
        prompt_tokens = len(input_ids)

        # Guard against empty prompt (same as _generate_fast)
        if not input_ids:
            bos_id = getattr(tokenizer, "bos_token_id", None)
            if bos_id is not None:
                input_ids = [bos_id]
            else:
                eos_id = getattr(tokenizer, "eos_token_id", 1)
                input_ids = [eos_id]
            prompt_tokens = len(input_ids)

        # ── Context window: clamp generation + truncate over-long prompt ──
        # Same fix as the non-streaming fast path — resolve the REAL
        # context window (mlx-lm uses max_position_embeddings, not max_seq_len, so
        # the old resolution was dead for most models) and clamp max_tokens so
        # prompt+generation can't decode past the window into RoPE garbage.
        _max_ctx = _resolve_model_max_ctx(model)
        if _max_ctx and _max_ctx > 0:
            if prompt_tokens >= _max_ctx:
                _orig = prompt_tokens
                input_ids = input_ids[-max(1, _max_ctx - 1) :]
                prompt_tokens = len(input_ids)
                logger.warning(
                    "Streaming fast path prompt truncated to fit context: %d → %d tokens (ctx=%d)",
                    _orig,
                    prompt_tokens,
                    _max_ctx,
                )
            _room = _max_ctx - prompt_tokens
            if _room >= 1 and max_tokens > _room:
                logger.info(
                    "Streaming fast path clamped max_tokens %d → %d to fit context "
                    "(prompt=%d, ctx=%d)",
                    max_tokens,
                    _room,
                    prompt_tokens,
                    _max_ctx,
                )
                max_tokens = _room

        # Convert KV cache breakpoint char offsets to token positions.
        _stream_kv_breakpoints: list[int] = []
        if kv_cache_breakpoints and isinstance(prompt, str) and len(prompt) > 0:
            for char_off in kv_cache_breakpoints:
                if char_off <= 0 or char_off > len(prompt):
                    continue
                try:
                    prefix_ids = tokenizer.encode(prompt[:char_off])
                    _stream_kv_breakpoints.append(len(prefix_ids))
                except Exception:
                    logger.debug(
                        "KV breakpoint char->token conversion failed at offset %d",
                        char_off,
                        exc_info=True,
                    )

        # normalize eos ids — some tokenizers (Qwen3.6-27B) expose eos_token_ids
        # as a BARE INT, which crashed stop_ids.update(...) with "'int' object is not
        # iterable". The non-streaming _generate_fast got this fix; the streaming
        # twin was missed → every streaming request on that tokenizer raised → client got
        # finish_reason="error", no content. Accept both shapes.
        stop_ids = set()
        # collect EOS separately (ignore_eos / min_tokens) — see _generate_fast.
        _eos_ids: set[int] = set()
        _eid = getattr(tokenizer, "eos_token_id", None)
        if _eid is not None:
            _eos_ids.update(_eid if isinstance(_eid, (list, tuple, set)) else (_eid,))
        _eids = getattr(tokenizer, "eos_token_ids", None)
        if _eids is not None:
            _eos_ids.update(
                _eids if isinstance(_eids, (list, tuple, set)) else (_eids,)
            )
        if not ignore_eos:
            stop_ids.update(_eos_ids)

        # : always string-match user stops (see _generate_fast)
        # — a bare stop's encoded id rarely equals the space-prefixed token the
        # model emits, so token-id matching alone silently missed most stops.
        stop_suffixes = []
        if stop:
            for s in stop:
                if not s:
                    continue
                ids = tokenizer.encode(s)
                if len(ids) == 1:
                    stop_ids.add(ids[0])
                stop_suffixes.append(s)
        if stop_token_ids:
            stop_ids.update(stop_token_ids)

        # Multi-token stop hold-back : withhold any streamed text that
        # could be the start of a multi-token stop string so its prefix never
        # leaks before the match completes (SSE is append-only). Active whenever
        # string stops exist. When logprobs is off the steady-state emit uses
        # feed() (per-token lp_entry is None anyway). When logprobs is ON we
        # route through feed_lp() instead, which keeps each token's logprob
        # aligned to its text. Previously hold-back was disabled under logprobs
        # (`and not logprobs`), which let a multi-token stop prefix LEAK into the
        # streamed output when logprobs was on . When inactive
        # (no stops), the buffer is pure passthrough.
        _hb_active = bool(stop_suffixes)
        _hb = StopHoldbackBuffer(stop_suffixes if _hb_active else None)

        # route temp>0 through _build_temp_sampler, like the non-streaming
        # fast path (3160). The bare make_sampler uses mlx-lm's categorical_sampling,
        # which is @mx.compile(inputs=mx.random.state) — the PRNG-cache trap that
        # makes `seed` a no-op (non-reproducible) and collapses concurrent/n>1 temp>0
        # STREAMING requests to identical token streams. This streaming path was the
        # un-propagated sibling of the non-streaming fix.
        if temperature is not None and temperature > 1e-6:
            sampler = _build_temp_sampler(
                temperature=temperature,
                top_p=top_p,
                top_k=top_k if top_k and top_k > 0 else 0,
                min_p=min_p if min_p else 0.0,
                seed=seed,
                xtc_probability=xtc_probability,
                xtc_threshold=xtc_threshold,
            )
        else:
            sampler = make_sampler(
                temp=temperature,
                top_p=top_p,
                top_k=top_k if top_k > 0 else 0,
                min_p=min_p,
                xtc_probability=xtc_probability,
                xtc_threshold=xtc_threshold,
            )

        # Grammar constraint for streaming fast path
        if json_schema is not None:
            try:
                sampler = _build_constrained_sampler(sampler, json_schema, tokenizer)
            except Exception:
                logger.warning(
                    "Grammar constraint setup failed in streaming", exc_info=True
                )

        # Build logits processors for penalty/bias params
        # length-gated KV-quant bits for THIS request (stream path).
        _req_kv_bits = self._effective_kv_quant_bits(prompt_tokens + max_tokens)
        _custom_logits_processors = logits_processors or []
        logits_processors = []
        if repetition_penalty != 1.0:

            def _repetition_penalty(tokens, logits, rp=repetition_penalty, ctx=20):
                if len(tokens) > 0:
                    recent = tokens[-ctx:]
                    import mlx.core as _mx

                    sel = logits[..., recent]
                    sel = _mx.where(sel < 0, sel * rp, sel / rp)
                    logits[..., _mx.array(recent)] = sel
                return logits

            logits_processors.append(_repetition_penalty)
        if frequency_penalty != 0.0 or presence_penalty != 0.0:
            # Maintain incremental counts dict instead of rebuilding
            # from full tokens list every step (O(T²) → O(T) total). The
            # closure captures a mutable state dict so successive calls only
            # observe the newest token.
            _fp_state: dict[str, object] = {"counts": {}, "last_len": -1}

            # The prompt offset inside the processor is 1, NOT the full
            # prompt length. mlx-lm prefills all-but-the-last prompt token OUTSIDE
            # _step, so the `tokens` accumulator handed to logits_processors is
            # [last_prompt_token, gen1, gen2, ...]. With n_prompt=prompt_tokens the
            # guard `cur_len <= n_prompt` made frequency/presence penalty INERT for
            # the first ~prompt_len generated tokens (short completions never
            # penalized at all) and then tokens[n_prompt:] under-counted repeats.
            def _freq_pres_penalty(
                tokens,
                logits,
                fp=frequency_penalty,
                pp=presence_penalty,
                n_prompt=1,
                _st=_fp_state,
            ):
                counts: dict[int, int] = _st["counts"]  # type: ignore[assignment]
                last_len = int(_st["last_len"])  # type: ignore[arg-type]
                cur_len = len(tokens)
                if cur_len <= n_prompt:
                    _st["last_len"] = cur_len
                    return logits
                # Reset/rebuild if tokens shrank (spec-decode rollback) or
                # we have not yet started incremental accounting.
                if cur_len < last_len or last_len < n_prompt:
                    counts = {}
                    for t in tokens[n_prompt:]:
                        counts[int(t)] = counts.get(int(t), 0) + 1
                    _st["counts"] = counts
                else:
                    start = max(last_len, n_prompt)
                    for t in tokens[start:]:
                        counts[int(t)] = counts.get(int(t), 0) + 1
                _st["last_len"] = cur_len
                for tid, cnt in counts.items():
                    # OpenAI permits frequency/presence penalty in [-2.0, 2.0];
                    # negative values BOOST the token (encourage repetition).
                    # Guarding with `> 0` silently dropped negative penalties
                    # (verified: fp=-2.0 produced output identical to fp=0).
                    if fp != 0.0:
                        logits[..., tid] = logits[..., tid] - fp * cnt
                    if pp != 0.0 and cnt > 0:
                        logits[..., tid] = logits[..., tid] - pp
                return logits

            logits_processors.append(_freq_pres_penalty)
        if logit_bias:

            def _logit_bias_proc(_tokens, logits, biases=logit_bias):
                # Skip token ids outside [0, vocab) — a user-supplied out-of-range
                # or negative id would otherwise index out of bounds and crash the
                # request (MLX: "Cannot squeeze axis 1 with size 0"). OpenAI ignores
                # invalid logit_bias token ids rather than erroring.
                vocab = logits.shape[-1]
                for tid, bias in biases.items():
                    if 0 <= tid < vocab:
                        logits[..., tid] = logits[..., tid] + bias
                return logits

            logits_processors.append(_logit_bias_proc)

        # suppress_tokens + min_tokens (mirror _generate_fast).
        if suppress_tokens:
            _sup = [int(t) for t in suppress_tokens]

            def _suppress_proc(_tokens, logits, sup=_sup):
                vocab = logits.shape[-1]
                for tid in sup:
                    if 0 <= tid < vocab:
                        logits[..., tid] = -float("inf")
                return logits

            logits_processors.append(_suppress_proc)
        # skip min_tokens EOS-masking when a JSON/grammar constraint is
        # active — the un-propagated streaming sibling of the non-streaming
        # fix (3336). The constraint already enforces a structural minimum, and
        # masking EOS at its DONE state leaves the constrained sampler an
        # all-(-inf) allowed set → its argmax fallback emits an INVALID non-EOS
        # token after a complete JSON value.
        if min_tokens and min_tokens > 0 and stop_ids and json_schema is None:
            _mask_ids = list(stop_ids)

            def _min_tokens_proc(tokens, logits, ids=_mask_ids, floor=int(min_tokens)):
                if (len(tokens) - 1) < floor:
                    vocab = logits.shape[-1]
                    for tid in ids:
                        if 0 <= tid < vocab:
                            logits[..., tid] = -float("inf")
                return logits

            logits_processors.append(_min_tokens_proc)

        # SAMP-2: Wrap user-provided custom logits processors to adapt signature.
        if _custom_logits_processors:
            logits_processors.extend(
                _wrap_custom_logits_processor(p) for p in _custom_logits_processors
            )

        # Thread-safe bridge: executor puts via call_soon_threadsafe so the
        # event loop's async consumer is woken for every token.
        _sentinel = object()
        _q: asyncio.Queue = asyncio.Queue(maxsize=512)
        loop = asyncio.get_running_loop()
        # Cross-thread cancel: set by the async consumer on timeout so the
        # GPU generation loop in _run_inner stops producing tokens.
        _timeout_cancel = threading.Event()

        def _put(item):
            # Backpressure-aware queue with retry: avoid blocking the MLX
            # executor thread with long sleeps — GPU work from other
            # requests would stall. Use short yields instead.
            if _q.qsize() > 400:  # 78% of 512
                time.sleep(0.0001)  # minimal yield — 0.1ms, not 1ms
            # NOTE: We do NOT call _q.get_nowait() here because this
            # function runs on the MLX executor thread (not the asyncio
            # event loop thread). asyncio.Queue.get_nowait() mutates the
            # internal deque AND calls _wakeup_next() which modifies
            # asyncio.Future objects — neither operation is thread-safe.
            for _attempt in range(10):
                if not _q.full():
                    with suppress(RuntimeError):  # Event loop closed — consumer gone
                        loop.call_soon_threadsafe(_q.put_nowait, item)
                    return
                if _attempt < 9:
                    time.sleep(0.0005)  # 0.5ms per attempt, not 5ms
            # Queue is persistently full — put an error sentinel so the
            # consumer sees finish_reason="error" instead of silently
            # missing content.
            logger.warning(
                "Streaming queue overflow after 50ms — sending error sentinel. "
                "Client will see finish_reason=error."
            )
            try:
                loop.call_soon_threadsafe(
                    _q.put_nowait,
                    Exception("Streaming queue overflow — output truncated"),
                )
            except Exception:
                logger.debug(
                    "Failed to put error sentinel into streaming queue", exc_info=True
                )

        # (LoRA concurrency keystone): acquire+apply / release+restore moved into
        # _run_with_lora on the executor thread (serialized with generate_step), not here on
        # the event loop. See the non-streaming _generate_fast for the rationale.

        # Inflight prefix sharing: defined at _run level so it's accessible
        # from exception handlers even if _run_inner crashes early
        _inflight_req_id = f"fp-s-{int(time.monotonic() * 1e6)}"

        def _unregister_inflight():
            try:
                from .inflight_prefix_sharing import get_inflight_tracker

                get_inflight_tracker().unregister(_inflight_req_id)
            except Exception:
                logger.debug(
                    "inflight prefix unregister failed in streaming", exc_info=True
                )

        _stream_gen_t0 = time.perf_counter()  # TTFT timing for streaming fast path
        # a TOTAL-generation deadline for the streaming path. The consumer's
        # asyncio.wait_for(_q.get(), timeout_seconds) only catches an INACTIVITY gap
        # (no token for timeout_seconds); a stream that keeps emitting tokens steadily
        # would run to max_tokens, blowing far past the user's timeout. The non-streaming
        # path enforces gen_t0 + timeout_seconds as a hard total deadline — mirror it in
        # the streaming GPU loop so `timeout` means the same (total wall time) for both.
        _stream_timeout_deadline = (
            _stream_gen_t0 + timeout_seconds if timeout_seconds else None
        )
        _stream_ttft_recorded = [False]  # mutable box to track first-token observation
        _stream_ttft_box = [0.0]  # mutable box for TTFT value
        # define BEFORE the _run_inner closure is submitted to the
        # executor — the closure reads/writes _cached_tokens_box[0] (line ~4377),
        # and if the executor thread reached it before the old definition (which
        # sat AFTER run_in_executor), it raised NameError. Defining it here
        # closes the ordering race.
        _cached_tokens_box = [0]  # mutable box for inner _run_inner to set
        _stream_itl_samples = []  # ITL samples for streaming fast path

        # Create detokenizer at _run scope so the error handler can finalize
        # it even if _run_inner() crashes before its own cleanup paths run.
        _detokenizer_ref = [None]  # mutable box shared with _run_inner
        # Hybrid-model guard : the prefix cache relies on trimming the
        # cached KV; hybrid/recurrent models (Qwen3.5 ArraysCache) are NOT
        # trimmable and reusing such a cache silently corrupts the recurrent
        # layers → grossly wrong logits. The non-streaming path disables the
        # cache for them; the streaming path must do the same (was missing).
        # Also bypass for seeded sampling (reproducibility needs a fresh cache).
        _stream_bypass_cache = (
            (seed is not None and temperature > 0)
            or not self._cache_supports_trim(self._model)
            # bypass when a LoRA adapter is active — the prefix cache is not
            # adapter-keyed, so reusing KV computed under a different adapter (or base)
            # silently serves the wrong adapter's state. Mirrors _generate_fast.
            or (lora_adapter is not None)
        )
        prefix_cache = None if _stream_bypass_cache else self._kv_prefix_cache

        def _save_breakpoint_prefixes(token_ids, kv_cache):
            """Save KV prefix cache entries at breakpoint positions."""
            if not _stream_kv_breakpoints or prefix_cache is None:
                return
            from .kv_prefix_cache import cache_length as _cache_len_bp

            for bp_pos in _stream_kv_breakpoints:
                if bp_pos < len(token_ids) and bp_pos >= 32:
                    bp_tokens = token_ids[:bp_pos]
                    # trim relative to the LIVE cache length (which
                    # includes generated tokens), not len(token_ids) — see the
                    # non-streaming twin. Storing bp_pos+num_generated tokens under a
                    # bp_pos-token key leaked the prior completion into a future hit.
                    trim_count = max(0, _cache_len_bp(kv_cache) - bp_pos)
                    try:
                        bp_cache = prefix_cache._snapshot_cache(
                            kv_cache, trim=trim_count
                        )
                        prefix_cache.add(bp_tokens, bp_cache)
                    except Exception:
                        logger.debug(
                            "KV breakpoint prefix add failed at pos %d",
                            bp_pos,
                            exc_info=True,
                        )

        def _run_inner():
            import mlx.core as mx

            if seed is not None:
                mx.random.seed(seed)
            ids = mx.array(input_ids)
            detokenizer = tokenizer.detokenizer
            detokenizer.reset()
            if self._lookahead_reasoning is not None:
                self._lookahead_reasoning._in_thinking = False
                self._lookahead_reasoning._thinking_tokens = []
                self._lookahead_reasoning._recent_accepts = []
            _detokenizer_ref[0] = detokenizer
            n_tok = 0
            thinking_tokens_used = 0
            think_end_token = None
            think_start_token = None
            _first_token = True
            _in_thinking = False
            _thinking_tokens: list[int] = []

            # Prefill progress tracking for streaming fast path
            _prefill_req_id = f"fp-s-{id(_run_inner)}-{int(time.monotonic() * 1e6)}"
            _prefill_tracker = None
            try:
                from .prefill_progress import get_prefill_tracker

                _prefill_tracker = get_prefill_tracker()
                _prefill_tracker.update(
                    _prefill_req_id, 0, prompt_tokens, self.model_name or "default"
                )
            except Exception:
                logger.debug("prefill tracker setup failed", exc_info=True)
                _prefill_tracker = None

            # resolve the think tokens UNCONDITIONALLY (was gated on
            # `thinking_budget is not None or enable_thinking`). enable_thinking defaults
            # to None end-to-end, yet Qwen3/Qwen3.5/DeepSeek-R1 chat templates are
            # default-ON: they inject the OPENING <think> into the PROMPT, so a plain
            # default-param request still generates a chain-of-thought. With the old gate,
            # think_end_token stayed None → the pre-seed below was skipped →
            # _in_thinking never flipped → the ENTIRE CoT leaked into visible delta.content
            # and reasoning_tokens stayed 0 (streaming diverged from the non-streaming fast
            # path, which recovers via the post-hoc closing-only reasoning parser). The
            # single-token guard already prevents false positives; the pre-seed is gated on
            # the prompt actually ending with an open <think>, so non-thinking models are
            # unaffected.
            # resolve via the bracketed-form helper (the bare "</think" encoded to
            # 2 tokens for Qwen3/DeepSeek-R1 → guard failed → CoT leaked into content).
            think_start_token, think_end_token = _resolve_think_token_ids(tokenizer)

            # chat-template-injected <think> models (Qwen3/Qwen3.5/
            # DeepSeek-R1) put the OPENING <think> in the PROMPT, so the model
            # OUTPUT contains only </think>. Without pre-seeding, _in_thinking
            # never flipped True → the ENTIRE chain-of-thought leaked into visible
            # delta.content and reasoning_tokens stayed 0 (streaming diverged from
            # the non-streaming path, which uses the closing-only reasoning parser).
            # Pre-seed when the rendered prompt ends with an open <think>.
            if think_end_token is not None and not _in_thinking:
                try:
                    from .thinking_budget import detect_needs_think_prefix

                    if detect_needs_think_prefix(list(input_ids), tokenizer):
                        _in_thinking = True
                        _thinking_tokens = []
                except Exception:
                    logger.debug("think-prefix detection failed", exc_info=True)

            # KV prefix cache for streaming
            # Proactive memory pressure eviction (vllm-mlx pattern)
            if prefix_cache is not None and self._mem_pressure_threshold > 0:
                prefix_cache.evict_under_pressure(self._mem_pressure_threshold)
                # Also evict from paged KV manager when enabled
                if self._kv_manager is not None:
                    try:
                        self._kv_manager.memory_pressure_evict(
                            self._mem_pressure_threshold / 100.0
                        )
                    except Exception:
                        logger.debug("paged KV pressure eviction failed", exc_info=True)
            try:
                cached_kv, _, matched = (
                    prefix_cache.get(ids)
                    if prefix_cache is not None
                    else (None, None, 0)
                )
            except Exception:
                logger.warning(
                    "KV prefix cache get failed in streaming — falling back to full prefill",
                    exc_info=True,
                )
                cached_kv, _, matched = None, None, 0
            cache = (
                cached_kv
                if cached_kv is not None
                else _create_prompt_cache_with_quant(
                    model, _req_kv_bits, self._kv_quant_group_size
                )
            )
            ids_to_prefill = ids[matched:] if cached_kv is not None else ids
            if len(ids_to_prefill) == 0 and len(ids) > 0:
                ids_to_prefill = ids[-1:]
            _stream_cached_tokens = matched
            _cached_tokens_box[0] = _stream_cached_tokens

            # Inflight prefix sharing ()
            _inflight_entry = None
            if cached_kv is None:
                try:
                    from .inflight_prefix_sharing import get_inflight_tracker

                    _tracker = get_inflight_tracker()
                    _inflight_entry = _tracker.find_prefix(
                        [int(t) for t in ids], self.model_name or ""
                    )
                    # BUG #1: live-borrow reuse disabled — see the
                    # non-streaming site for the full rationale (empty-prefill
                    # crash + cross-request corruption against a growing donor
                    # cache, unsafe under the now-working engine loop).
                except Exception:
                    logger.debug(
                        "inflight prefix lookup failed in streaming", exc_info=True
                    )

            # Register our prefill as in-flight for concurrent requests to share
            try:
                from .inflight_prefix_sharing import get_inflight_tracker

                get_inflight_tracker().register(
                    _inflight_req_id,
                    [int(t) for t in ids],
                    cache,
                    self.model_name or "",
                )
            except Exception:
                logger.debug(
                    "inflight prefix register failed in streaming", exc_info=True
                )

            _lprocs = logits_processors if logits_processors else None
            _last_stream_tok_time = 0.0
            # mlx-lm's generate_step quantizes the cache per-step,
            # but RotatingKVCache.to_quantized() is NYI → passing kv_bits crashes the
            # stream for sliding-window models (Gemma-3/Gemma-4/gpt-oss/Cohere2). Suppress
            # KV quant for those (matches mlx-lm's CLI + the non-streaming guard above).
            _stream_kv_bits = _req_kv_bits
            if _stream_kv_bits is not None:
                try:
                    from mlx_lm.models.cache import RotatingKVCache

                    if any(isinstance(c, RotatingKVCache) for c in cache):
                        _stream_kv_bits = None
                except Exception:
                    pass
            with _wired_limit_ctx(model):
                for token, logits in generate_step(
                    ids_to_prefill,
                    model,
                    max_tokens=max_tokens,
                    sampler=sampler,
                    prompt_cache=cache,
                    logits_processors=_lprocs,
                    prefill_step_size=_prefill_step_size(),
                    # the streaming path had NO KV-quant — mlx-lm
                    # quantizes the cache per-step internally, but only when these
                    # are passed, so YUNSHU_KV_QUANT_BITS gave zero in-flight memory
                    # benefit on the (default) streaming path. None → mlx-lm no-ops.
                    kv_bits=_stream_kv_bits,
                    kv_group_size=self._kv_quant_group_size,
                    quantized_kv_start=self._kv_quant_start,
                ):
                    n_tok += 1
                    # Check stop_ids BEFORE adding to detokenizer to avoid emitting stop text
                    stop_hit = token in stop_ids
                    suffix_hit = False
                    if not stop_hit:
                        detokenizer.add_token(token)
                        if stop_suffixes:
                            suffix_hit = any(
                                detokenizer.text.endswith(s) for s in stop_suffixes
                            )
                    # Compute per-token logprobs (same pattern as _generate_fast)
                    _lp_entry = None
                    if logprobs and logits is not None:
                        import mlx.core as _mx

                        _lp_logits = logits.astype(_mx.float32)
                        _log_probs = _lp_logits - _mx.logsumexp(
                            _lp_logits, axis=-1, keepdims=True
                        )
                        _tok_lp = float(_log_probs[token])
                        if _tok_lp != _tok_lp or _tok_lp == float("-inf"):
                            _tok_lp = -100.0
                        _lp_entry = {"token_id": int(token), "logprob": _tok_lp}
                        if top_logprobs and top_logprobs > 0:
                            _k = min(top_logprobs, _log_probs.shape[0])
                            _sorted_idx = _mx.argsort(-_log_probs)
                            _top_k_idx = _sorted_idx[:_k]
                            _top_entries = []
                            for j in range(_k):
                                _tlp = float(_log_probs[int(_top_k_idx[j])])
                                if _tlp != _tlp or _tlp == float("-inf"):
                                    _tlp = -100.0
                                _top_entries.append(
                                    {"token_id": int(_top_k_idx[j]), "logprob": _tlp}
                                )
                            _lp_entry["top_logprobs"] = _top_entries
                    # TokenPipeline: submit GPU stages for tracking
                    if _pipeline is not None and _pipeline.is_running:
                        _ptok = _pipeline.submit_stage1_result(
                            logits=None, token_id=int(token)
                        )
                        _ptok = _pipeline.submit_stage2_result(
                            _ptok, sampled_id=int(token)
                        )
                    # Check cancellation
                    if _is_cancelled(cancel_event):
                        mx.synchronize()
                        # Flush remaining detokenizer bytes before cancelling
                        try:
                            detokenizer.finalize()
                            remaining = detokenizer.last_segment
                            if _hb_active:
                                # Not a stop match — flush held-back text (real
                                # output that merely looked like a stop-prefix).
                                remaining = _hb.feed(remaining) + _hb.flush()
                            if remaining:
                                _put(
                                    (
                                        remaining,
                                        n_tok,
                                        None,
                                        len(_thinking_tokens),
                                        None,
                                        "reasoning" if _in_thinking else "normal",
                                    )
                                )
                        except Exception:
                            logger.debug(
                                "detokenizer finalize in cancel handler failed",
                                exc_info=True,
                            )
                        # Emit terminal stop chunk so consumer sees finished=True
                        _put(
                            (
                                "",
                                n_tok,
                                "stop",
                                len(_thinking_tokens),
                                None,
                                "reasoning" if _in_thinking else "normal",
                            )
                        )
                        if _pipeline is not None:
                            _pipeline.finish()
                        if _prefill_tracker is not None:
                            _prefill_tracker.remove(_prefill_req_id)
                        _unregister_inflight()
                        return
                    # Check timeout-driven cancel from consumer (inactivity) OR the total
                    # generation deadline — both finalize + emit a "timeout"
                    # terminal chunk and stop the GPU loop.
                    if _timeout_cancel.is_set() or (
                        _stream_timeout_deadline is not None
                        and time.perf_counter() > _stream_timeout_deadline
                    ):
                        mx.synchronize()
                        try:
                            detokenizer.finalize()
                            remaining = detokenizer.last_segment
                            if _hb_active:
                                # Not a stop match — flush held-back text (real
                                # output that merely looked like a stop-prefix).
                                remaining = _hb.feed(remaining) + _hb.flush()
                            if remaining:
                                _put(
                                    (
                                        remaining,
                                        n_tok,
                                        None,
                                        len(_thinking_tokens),
                                        None,
                                        "reasoning" if _in_thinking else "normal",
                                    )
                                )
                        except Exception:
                            logger.debug(
                                "detokenizer finalize in timeout cancel failed",
                                exc_info=True,
                            )
                        _put(
                            (
                                "",
                                n_tok,
                                "timeout",
                                len(_thinking_tokens),
                                None,
                                "reasoning" if _in_thinking else "normal",
                            )
                        )
                        if _pipeline is not None:
                            _pipeline.finish()
                        if _prefill_tracker is not None:
                            _prefill_tracker.remove(_prefill_req_id)
                        _unregister_inflight()
                        return
                    # Prefill complete on first token — remove from progress tracker
                    if _first_token:
                        _first_token = False
                        # Record TTFT for Prometheus
                        if not _stream_ttft_recorded[0]:
                            _stream_ttft_recorded[0] = True
                            _stream_ttft_box[0] = time.perf_counter() - _stream_gen_t0
                        _last_stream_tok_time = time.perf_counter()
                        if _prefill_tracker is not None:
                            _prefill_tracker.update(
                                _prefill_req_id,
                                prompt_tokens,
                                prompt_tokens,
                                self.model_name or "default",
                            )
                    else:
                        # ITL tracking for streaming fast path
                        _tok_now = time.perf_counter()
                        _itl = _tok_now - _last_stream_tok_time
                        _last_stream_tok_time = _tok_now
                        if _itl > 0 and _itl < 10:
                            _stream_itl_samples.append(_itl)
                    new_text = "" if stop_hit else detokenizer.last_segment
                    if suffix_hit and not stop_hit and not _hb_active:
                        # (Non-hold-back path) Trim ONLY the matched stop suffix
                        # from this segment so leading non-stop text is still
                        # streamed (e.g. emit "done" from "doneSTOP"). When
                        # _hb_active, the hold-back buffer handles this via
                        # take_stopped() instead, so skip the in-segment trim.
                        for _s in stop_suffixes:
                            if new_text.endswith(_s):
                                new_text = new_text[: -len(_s)]
                                break
                        else:
                            new_text = ""
                    # Exclude stop/suffix-triggering token from completion count
                    if stop_hit or suffix_hit:
                        n_tok -= 1
                    # Track thinking segment boundaries BEFORE budget check
                    # so that a natural </think token is detected first and
                    # the budget enforcement does not append a duplicate.
                    # Skip stop/suffix tokens from thinking tracking since they
                    # are excluded from completion_tokens (invariant must hold).
                    if think_start_token is not None and not (stop_hit or suffix_hit):
                        if not _in_thinking and token == think_start_token:
                            _in_thinking = True
                            _thinking_tokens = []
                            self._lookahead_reasoning.check_thinking_state_text(
                                "<think"
                            )
                        elif _in_thinking:
                            _thinking_tokens.append(token)
                            if token == think_end_token:
                                _in_thinking = False
                                self._lookahead_reasoning.check_thinking_state_text(
                                    "</think"
                                )
                    # Thinking budget enforcement in streaming.
                    # Only force-append think_end_token + add to detokenizer
                    # if the current token is NOT already the natural closing tag.
                    if thinking_budget is not None and _in_thinking:
                        thinking_tokens_used += 1
                        if (
                            thinking_tokens_used >= thinking_budget
                            and think_end_token is not None
                        ):
                            # Token was already appended at line ~3298 — don't
                            # duplicate it. Only force-emit the closing tag.
                            _in_thinking = False
                            # Emit current token's text first (still reasoning content)
                            if new_text:
                                _put(
                                    (
                                        new_text,
                                        n_tok,
                                        None,
                                        len(_thinking_tokens),
                                        _lp_entry,
                                        "reasoning",
                                    )
                                )
                            # Only force-emit closing tag if the token isn't already it
                            if token != think_end_token:
                                n_tok += 1  # Count the forced closing tag token
                                detokenizer.add_token(think_end_token)
                                _thinking_tokens.append(think_end_token)
                                _end_text = detokenizer.last_segment
                                if _end_text:
                                    _put(
                                        (
                                            _end_text,
                                            n_tok,
                                            None,
                                            len(_thinking_tokens),
                                            None,
                                            "reasoning",
                                        )
                                    )
                            # Store thinking segment before returning
                            if _thinking_tokens and self._thinking_store is not None:
                                _store_thinking_segment(
                                    ids,
                                    _thinking_tokens,
                                    self._thinking_store,
                                    kv_cache=cache,
                                )
                            detokenizer.finalize()
                            _remaining = detokenizer.last_segment
                            if _hb_active:
                                _remaining = _hb.feed(_remaining) + _hb.flush()
                            if _remaining:
                                _put(
                                    (
                                        _remaining,
                                        n_tok,
                                        None,
                                        len(_thinking_tokens),
                                        None,
                                        "normal",
                                    )
                                )
                            # thinking_budget exhausted → "length" (matches fast-path and
                            # OpenAI semantics: budget == token limit)
                            _put(
                                (
                                    "",
                                    n_tok,
                                    "length",
                                    len(_thinking_tokens),
                                    None,
                                    "normal",
                                )
                            )
                            if _pipeline is not None:
                                _pipeline.finish()
                            if prefix_cache is not None:
                                prefix_cache.add(ids, cache)
                            _save_breakpoint_prefixes(ids, cache)
                            mx.synchronize()
                            if _prefill_tracker is not None:
                                _prefill_tracker.remove(_prefill_req_id)
                            _unregister_inflight()
                            return
                    _is_stopping = stop_hit or suffix_hit
                    # the closing </think> token flipped _in_thinking to
                    # False at line ~5293 BEFORE this state is computed, so its own
                    # text ("</think>") would leak into visible content. Keep the
                    # boundary token classified as reasoning so the tag stays out of
                    # delta.content (matches the non-streaming parser stripping it).
                    _just_closed = (
                        think_end_token is not None
                        and token == think_end_token
                        and not (stop_hit or suffix_hit)
                    )
                    _cur_state = (
                        "reasoning" if (_in_thinking or _just_closed) else "normal"
                    )
                    if _is_stopping:
                        # Emit current text without finish_reason so the consumer
                        # reads it before breaking on done=True below.
                        if _hb_active and logprobs:
                            # Logprob-aware hold-back: each token's lp rides the
                            # chunk that first reveals its text; stop prefixes are
                            # withheld so they never leak with logprobs on.
                            for _t, _lp in _hb.feed_lp(new_text, _lp_entry):
                                if _t or _lp is not None:
                                    _put(
                                        (
                                            _t,
                                            n_tok,
                                            None,
                                            len(_thinking_tokens),
                                            _lp,
                                            _cur_state,
                                        )
                                    )
                        elif _hb_active:
                            _emit = _hb.feed(new_text)
                            if _emit:
                                _put(
                                    (
                                        _emit,
                                        n_tok,
                                        None,
                                        len(_thinking_tokens),
                                        _lp_entry,
                                        _cur_state,
                                    )
                                )
                        elif new_text:
                            _put(
                                (
                                    new_text,
                                    n_tok,
                                    None,
                                    len(_thinking_tokens),
                                    _lp_entry,
                                    _cur_state,
                                )
                            )
                    else:
                        if _hb_active and logprobs:
                            for _t, _lp in _hb.feed_lp(new_text, _lp_entry):
                                if _t or _lp is not None:
                                    _put(
                                        (
                                            _t,
                                            n_tok,
                                            None,
                                            len(_thinking_tokens),
                                            _lp,
                                            _cur_state,
                                        )
                                    )
                        elif _hb_active:
                            _emit = _hb.feed(new_text)
                            if _emit:
                                _put(
                                    (
                                        _emit,
                                        n_tok,
                                        None,
                                        len(_thinking_tokens),
                                        _lp_entry,
                                        _cur_state,
                                    )
                                )
                        else:
                            _put(
                                (
                                    new_text,
                                    n_tok,
                                    None,
                                    len(_thinking_tokens),
                                    _lp_entry,
                                    _cur_state,
                                )
                            )
                    # A complete stop that landed MID-segment (e.g. one token decodes to
                    # "aSTOPb") is missed by the endswith() suffix_hit but is now held in
                    # the buffer — fire the stop so generation ends and take_stopped()
                    # drops it (the streaming backstop the non-streaming find() path had).
                    if _hb_active and not _is_stopping and _hb.contains_stop():
                        _is_stopping = True
                    if _is_stopping:
                        # Store thinking segment on stop
                        if _thinking_tokens and self._thinking_store is not None:
                            _store_thinking_segment(
                                ids,
                                _thinking_tokens,
                                self._thinking_store,
                                kv_cache=cache,
                            )
                        detokenizer.finalize()
                        _remaining = detokenizer.last_segment
                        if _hb_active:
                            # Feed the final bytes through the buffer, then flush
                            # with the matched stop trimmed. This correctly
                            # discards a multi-token stop whose prefix was held
                            # back across earlier chunks (e.g. "\n\n" as two "\n"
                            # tokens) without leaking it.
                            _emit = _hb.feed(_remaining)
                            if _emit:
                                _put(
                                    (
                                        _emit,
                                        n_tok,
                                        None,
                                        len(_thinking_tokens),
                                        None,
                                        _cur_state,
                                    )
                                )
                            _tail = _hb.take_stopped()
                            if _tail:
                                _put(
                                    (
                                        _tail,
                                        n_tok,
                                        None,
                                        len(_thinking_tokens),
                                        None,
                                        _cur_state,
                                    )
                                )
                        else:
                            # Trim stop suffix from remaining text — the suffix may
                            # span multiple tokens, so detokenizer.text still contains
                            # it even after finalize(). Without this, the partial
                            # suffix text leaks into the output.
                            if suffix_hit and stop_suffixes and _remaining:
                                for s in stop_suffixes:
                                    if _remaining.endswith(s):
                                        _remaining = _remaining[: -len(s)]
                                        break
                            if _remaining:
                                _put(
                                    (
                                        _remaining,
                                        n_tok,
                                        None,
                                        len(_thinking_tokens),
                                        None,
                                        _cur_state,
                                    )
                                )
                        # Final stop chunk — consumer breaks on this
                        _put(
                            ("", n_tok, "stop", len(_thinking_tokens), None, _cur_state)
                        )
                        if _pipeline is not None:
                            _pipeline.finish()
                        if prefix_cache is not None:
                            prefix_cache.add(ids, cache)
                        _save_breakpoint_prefixes(ids, cache)
                        mx.synchronize()
                        _unregister_inflight()
                        return
                # Store thinking segment at end of generation
                if _thinking_tokens and self._thinking_store is not None:
                    _store_thinking_segment(
                        ids, _thinking_tokens, self._thinking_store, kv_cache=cache
                    )
                if prefix_cache is not None:
                    prefix_cache.add(ids, cache)
                _save_breakpoint_prefixes(ids, cache)
                detokenizer.finalize()
                remaining = detokenizer.last_segment
                if _hb_active:
                    # Generation ended without a stop match — flush held-back
                    # text (it was real output that looked like a stop-prefix).
                    remaining = _hb.feed(remaining) + _hb.flush()
                _final_state = "reasoning" if _in_thinking else "normal"
                if remaining:
                    _put(
                        (
                            remaining,
                            n_tok,
                            None,
                            len(_thinking_tokens),
                            None,
                            _final_state,
                        )
                    )
                _put(("", n_tok, "length", len(_thinking_tokens), None, _final_state))
                mx.synchronize()
                # Finish pipeline tracking at end of generation
                if _pipeline is not None:
                    _pipeline.finish()
                # Clean up prefill progress entry (may persist if first token wasn't reached)
                if _prefill_tracker is not None:
                    _prefill_tracker.remove(_prefill_req_id)
                _unregister_inflight()

        def _finalize_detokenizer():
            """Finalize the detokenizer if it was created, to flush internal byte buffers."""
            _dtk = _detokenizer_ref[0]
            if _dtk is not None:
                try:
                    _dtk.finalize()
                except Exception:
                    logger.debug(
                        "detokenizer finalize in error handler failed", exc_info=True
                    )

        def _run():
            try:
                _run_inner()
                # Normal exit: clear Metal cache to prevent GPU memory
                # accumulation across sequential streaming requests.
                try:
                    import mlx.core as _cleanup_mx

                    _cleanup_mx.synchronize()
                    _cleanup_mx.clear_cache()
                except Exception:
                    pass
            except (MemoryError, RuntimeError) as e:
                _unregister_inflight()
                _finalize_detokenizer()
                try:
                    import mlx.core as _cleanup_mx

                    _cleanup_mx.synchronize()
                    _cleanup_mx.clear_cache()
                    import gc

                    gc.collect()  # Force GC of KV cache tensors
                except Exception:
                    logger.debug(
                        "GPU cache cleanup failed in streaming OOM handler",
                        exc_info=True,
                    )
                if isinstance(e, MemoryError) or "memory" in str(e).lower():
                    logger.warning(f"OOM during streaming: {e}")
                _put(e)
            except Exception as e:
                _unregister_inflight()
                _finalize_detokenizer()
                try:
                    import mlx.core as _cleanup_mx

                    _cleanup_mx.synchronize()
                    _cleanup_mx.clear_cache()
                except Exception:
                    logger.debug(
                        "GPU cache cleanup failed in streaming error handler",
                        exc_info=True,
                    )
                _put(e)
            finally:
                _put(_sentinel)

        def _run_with_lora():
            # (LoRA concurrency keystone): adapter lifecycle on the executor thread,
            # serialized with generate_step (see the non-streaming twin).
            _lora_applied = False
            if lora_adapter and getattr(self, "_lora_manager", None) is not None:
                try:
                    _lora_applied = self._lora_manager.acquire_adapter(lora_adapter)
                except Exception as _le:  # fail loud, don't serve base
                    raise RuntimeError(
                        f"LoRA adapter '{lora_adapter}' could not be applied"
                    ) from _le
                if not _lora_applied:
                    raise RuntimeError(
                        f"LoRA adapter '{lora_adapter}' could not be applied"
                    )
            try:
                return _run()
            finally:
                if _lora_applied and getattr(self, "_lora_manager", None) is not None:
                    try:
                        self._lora_manager.release_adapter(lora_adapter)
                    except Exception:
                        logger.debug(
                            "LoRA release failed (stream executor)", exc_info=True
                        )

        from .mlx_executor import get_mlx_executor

        executor = get_mlx_executor()
        future = loop.run_in_executor(executor, _run_with_lora)

        accumulated = ""
        n_tok = 0
        _reasoning_tokens = 0
        # _cached_tokens_box now defined above, before run_in_executor .
        _fp_lock = getattr(self, "_fast_path_lock", None)
        if _fp_lock is not None:
            with _fp_lock:
                self._active_fast_path_count += 1
        try:
            while True:
                try:
                    item = await asyncio.wait_for(_q.get(), timeout=timeout_seconds)
                except TimeoutError:
                    logger.warning(
                        f"Streaming fast path timeout: no token for {timeout_seconds}s"
                    )
                    _timeout_cancel.set()  # Signal GPU loop to stop
                    # Yield terminal output so consumer sees finished=True
                    yield GenerationOutput(
                        text=_clean_special_tokens(accumulated) if accumulated else "",
                        new_text="",
                        prompt_tokens=prompt_tokens,
                        completion_tokens=n_tok,
                        finished=True,
                        finish_reason="timeout",
                        error=f"Streaming timeout: no token for {timeout_seconds}s",
                        ttft_ms=round(_stream_ttft_box[0] * 1000, 1)
                        if _stream_ttft_box[0] > 0
                        else 0.0,
                        cached_tokens=_cached_tokens_box[0],
                        reasoning_tokens=_reasoning_tokens,
                    )
                    break
                if item is _sentinel:
                    break
                if isinstance(item, BaseException):
                    err_msg = str(item).lower()
                    is_oom = isinstance(item, MemoryError) or "memory" in err_msg
                    if is_oom:
                        yield GenerationOutput(
                            text=_clean_special_tokens(accumulated)
                            if accumulated
                            else "",
                            new_text="",
                            prompt_tokens=prompt_tokens,
                            completion_tokens=n_tok,
                            finished=True,
                            finish_reason="memory_limit",
                            error=str(item),
                            ttft_ms=round(_stream_ttft_box[0] * 1000, 1)
                            if _stream_ttft_box[0] > 0
                            else 0.0,
                            cached_tokens=_cached_tokens_box[0],
                            reasoning_tokens=_reasoning_tokens,
                        )
                    else:
                        # Non-OOM exception: yield terminal error output
                        yield GenerationOutput(
                            text=_clean_special_tokens(accumulated)
                            if accumulated
                            else "",
                            new_text="",
                            prompt_tokens=prompt_tokens,
                            completion_tokens=n_tok,
                            finished=True,
                            finish_reason="error",
                            error=str(item),
                            ttft_ms=round(_stream_ttft_box[0] * 1000, 1)
                            if _stream_ttft_box[0] > 0
                            else 0.0,
                            cached_tokens=_cached_tokens_box[0],
                            reasoning_tokens=_reasoning_tokens,
                        )
                    break
                if len(item) >= 6:
                    (
                        new_text,
                        tok_count,
                        _fr_val,
                        _reasoning_tokens,
                        _lp_entry,
                        _cur_state,
                    ) = item[:6]
                elif len(item) == 5:
                    new_text, tok_count, _fr_val, _reasoning_tokens, _lp_entry = item
                    _cur_state = None
                elif len(item) == 4:
                    new_text, tok_count, _fr_val, _reasoning_tokens = item
                    _lp_entry = None
                    _cur_state = None
                else:
                    new_text, tok_count, _fr_val = item
                    _lp_entry = None
                    _cur_state = None
                # Backward compat: _fr_val may be bool (from old-style tuples)
                # or str ("stop"/"length") or None (intermediate token)
                if isinstance(_fr_val, bool):
                    finish_reason = "stop" if _fr_val else None
                    done = _fr_val
                else:
                    finish_reason = _fr_val  # str or None
                    done = _fr_val is not None
                accumulated += new_text
                if len(accumulated) > _MAX_STREAMING_TEXT_BUFFER:
                    logger.error(
                        "Streaming text buffer exceeded 1MB limit (%d bytes) — truncating",
                        len(accumulated),
                    )
                    yield GenerationOutput(
                        text=_clean_special_tokens(accumulated),
                        new_text="",
                        prompt_tokens=prompt_tokens,
                        completion_tokens=tok_count,
                        finished=True,
                        finish_reason="length",
                        error="Streaming text buffer exceeded 1MB limit",
                        cached_tokens=_cached_tokens_box[0],
                        reasoning_tokens=_reasoning_tokens,
                        ttft_ms=round(_stream_ttft_box[0] * 1000, 1)
                        if _stream_ttft_box[0] > 0
                        else 0.0,
                    )
                    break
                n_tok = tok_count

                # TokenPipeline: run stage 3 overlap for stats tracking
                if (
                    _pipeline is not None
                    and _pipeline.is_running
                    and _pipeline._current is not None
                ):
                    await _pipeline.next_token(
                        detokenize_fn=lambda _tid, _t=new_text: _t,
                    )

                # Streaming backpressure: slow down if client can't keep up
                if _backpressure.check_backpressure(_q.qsize()):
                    _delay = _backpressure.get_delay_ms(_q.qsize())
                    if _delay > 0:
                        await asyncio.sleep(_delay / 1000)

                # Record TTFT in Prometheus on first token
                if _stream_ttft_recorded[0] and _stream_ttft_box[0] > 0:
                    try:
                        from yunshu_gateway.middleware.prometheus_exporter import (
                            get_prometheus_metrics,
                        )

                        pm = get_prometheus_metrics()
                        pm.observe_histogram(
                            "ttft_seconds",
                            _stream_ttft_box[0],
                            labels={"model_id": self.model_label},
                        )
                    except Exception:
                        logger.debug(
                            "streaming TTFT prometheus recording failed", exc_info=True
                        )
                    _stream_ttft_recorded[0] = False  # only observe once
                # Attach logprobs to output if computed for this token
                _lp_list = None
                if _lp_entry is not None:
                    # Decode token string for the logprob entry
                    try:
                        _lp_entry["token"] = tokenizer.decode([_lp_entry["token_id"]])
                        from .text_utils import token_id_to_bytes  # raw bytes

                        _lp_entry["bytes"] = token_id_to_bytes(
                            tokenizer, _lp_entry["token_id"], _lp_entry["token"]
                        )
                    except Exception:
                        logger.debug(
                            "logprob token decode failed in streaming", exc_info=True
                        )
                        _lp_entry["token"] = ""
                        _lp_entry["bytes"] = []
                    _lp_list = [_lp_entry]
                yield GenerationOutput(
                    text=_clean_special_tokens(accumulated),
                    new_text=_clean_special_tokens(new_text),
                    prompt_tokens=prompt_tokens,
                    completion_tokens=n_tok,
                    finished=done,
                    finish_reason=finish_reason,
                    reasoning_tokens=_reasoning_tokens,
                    logprobs=_lp_list,
                    cached_tokens=_cached_tokens_box[0],
                    ttft_ms=round(_stream_ttft_box[0] * 1000, 1)
                    if _stream_ttft_box[0] > 0
                    else 0.0,
                    current_state=_cur_state,
                )
                if done:
                    break
        finally:
            # LoRA release+restore happens inside _run_with_lora on the executor .
            # Record in ServerMetrics for streaming fast path (consistency)
            if n_tok > 0:
                try:
                    from .server_metrics import get_server_metrics

                    _sm = get_server_metrics()
                    _sm.record_request_complete(
                        prompt_tokens=prompt_tokens,
                        completion_tokens=n_tok,
                        cached_tokens=_cached_tokens_box[0],
                        prefill_duration=_stream_ttft_box[0],
                        generation_duration=(time.perf_counter() - _stream_gen_t0),
                        model_id=self.model_name,
                    )
                    # Record ITL samples from streaming path
                    if _stream_itl_samples:
                        for _itl in _stream_itl_samples:
                            _sm.record_itl(_itl)
                        # Record average ITL in Prometheus
                        try:
                            from yunshu_gateway.middleware.prometheus_exporter import (
                                get_prometheus_metrics,
                            )

                            pm = get_prometheus_metrics()
                            avg_itl = sum(_stream_itl_samples) / len(
                                _stream_itl_samples
                            )
                            pm.observe_histogram(
                                "itl_seconds",
                                avg_itl,
                                labels={"model_id": self.model_label},
                            )
                        except Exception:
                            logger.debug(
                                "streaming ITL prometheus recording failed",
                                exc_info=True,
                            )
                except Exception:
                    logger.debug(
                        "ServerMetrics recording failed in streaming fast path",
                        exc_info=True,
                    )
            # Stop pipeline and log stats
            if _pipeline is not None:
                _pipeline.stop()
                logger.info(
                    "TokenPipeline stats: %s",
                    _pipeline.get_stats(),
                )
            if not future.done():
                future.cancel()
                with suppress(asyncio.CancelledError, Exception):
                    await future
            # Drain remaining queue items to unblock the executor thread's
            # call_soon_threadsafe calls, preventing GPU work from continuing
            # after the consumer has stopped iterating.
            while not _q.empty():
                try:
                    _q.get_nowait()
                except asyncio.QueueEmpty:
                    break
            # Decrement active fast path count
            _fp_lock = getattr(self, "_fast_path_lock", None)
            if _fp_lock is not None:
                with _fp_lock:
                    self._active_fast_path_count -= 1

    def _mtp_prompt_tokens(self, result: dict, messages: list, enable_thinking) -> int:
        """Best-effort prompt-token count for the mlx-vlm MTP backend.

        The backend's result rarely carries prompt_tokens; the old code hardcoded
        0, under-reporting every prompt for billing. Prefer the backend's value,
        else estimate by encoding the chat-templated prompt with our tokenizer."""
        pt = int(result.get("prompt_tokens", 0) or 0)
        if pt > 0:
            return pt
        try:
            if self._tokenizer is not None:
                prompt = self._apply_chat_template(messages, enable_thinking)
                return len(self._tokenizer.encode(prompt))
        except Exception:
            logger.debug("MTP prompt_tokens estimate failed", exc_info=True)
        return 0

    def _warn_mtp_dropped_params(
        self,
        top_p,
        top_k,
        min_p,
        repetition_penalty,
        frequency_penalty,
        presence_penalty,
        logit_bias,
        json_schema,
    ) -> None:
        """Warn once when shaping/constraint params are silently dropped by the
        mlx-vlm MTP backend, which honors only temperature."""
        dropped = []
        if top_p is not None and top_p < 1.0:
            dropped.append("top_p")
        if top_k:
            dropped.append("top_k")
        if min_p:
            dropped.append("min_p")
        if repetition_penalty and repetition_penalty != 1.0:
            dropped.append("repetition_penalty")
        if frequency_penalty:
            dropped.append("frequency_penalty")
        if presence_penalty:
            dropped.append("presence_penalty")
        if logit_bias:
            dropped.append("logit_bias")
        if json_schema is not None:
            dropped.append("json_schema/response_format")
        if dropped and not getattr(self, "_mtp_dropped_warned", False):
            self._mtp_dropped_warned = True
            logger.warning(
                "YUNSHU_MTP backend honors only temperature; these request params "
                "are NOT applied and were ignored: %s",
                ", ".join(dropped),
            )

    async def chat(
        self,
        messages: list[dict],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 1.0,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        logit_bias: dict[int, float] | None = None,
        stop: list[str] | None = None,
        enable_thinking: bool | None = None,
        **kwargs,
    ) -> GenerationOutput:
        """Non-streaming chat completion (messages → template → generate)."""
        # mlx-vlm MTP backend delegation (single-backend mode).
        # Greedy → lossless MTP speculative decode (~1.8x); sampling → plain gen
        # on the same mlx-vlm model. Runs on the MLX executor thread.
        if self._mlxvlm_mtp is not None:
            import asyncio as _asyncio

            from .mlx_executor import get_mlx_executor

            # (honesty): the MTP backend only honors temperature — warn
            # when shaping/constraint params are set but silently dropped, so a
            # caller isn't misled into thinking json_schema/top_p/penalties applied.
            self._warn_mtp_dropped_params(
                top_p,
                top_k,
                min_p,
                repetition_penalty,
                frequency_penalty,
                presence_penalty,
                logit_bias,
                kwargs.get("json_schema"),
            )
            _loop = _asyncio.get_running_loop()
            # build the prompt via the engine's _apply_chat_template
            # (role remap developer→system / function→tool, family adapter, BOS
            # guard) instead of letting the MTP backend call apply_chat_template
            # on raw messages — which crashed on a `developer`/`function` role and
            # double-BOS'd BOS-prepending drafters.
            _mtp_prompt = self._apply_chat_template(messages, enable_thinking)
            _r = await _loop.run_in_executor(
                get_mlx_executor(),
                lambda: self._mlxvlm_mtp.generate(
                    messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    prompt=_mtp_prompt,
                ),
            )
            _txt = _r.get("text", "")
            # the MTP backend ignores `stop`; honor it post-hoc.
            if stop:
                for _s in stop:
                    if _s and _s in _txt:
                        _txt = _txt[: _txt.find(_s)]
                        break
            _ct = _r.get("completion_tokens", 0)
            return GenerationOutput(
                text=_txt,
                new_text=_txt,
                # report real prompt_tokens (was hardcoded 0 → under-billing).
                prompt_tokens=self._mtp_prompt_tokens(_r, messages, enable_thinking),
                completion_tokens=_ct,
                finished=True,
                finish_reason="length" if _ct >= max_tokens else "stop",
            )

        prompt = self._apply_chat_template(messages, enable_thinking)
        return await self.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            logit_bias=logit_bias,
            stop=stop,
            **kwargs,
        )

    async def stream_chat(
        self,
        messages: list[dict],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 1.0,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        logit_bias: dict[int, float] | None = None,
        stop: list[str] | None = None,
        enable_thinking: bool | None = None,
        **kwargs,
    ) -> AsyncIterator[GenerationOutput]:
        """Streaming chat completion (messages → template → stream_generate)."""
        # under the mlx-vlm MTP backend (YUNSHU_MTP=1) the mlx-lm fast path is NOT
        # loaded (self._model is None), so stream_generate would crash on `model = self._model`.
        # Generate via the MTP backend and emit the result as a single chunk (the backend
        # doesn't token-stream). Honor `stop` post-hoc since the backend ignores it.
        if self._mlxvlm_mtp is not None:
            from .mlx_executor import get_mlx_executor

            self._warn_mtp_dropped_params(
                top_p,
                top_k,
                min_p,
                repetition_penalty,
                frequency_penalty,
                presence_penalty,
                logit_bias,
                kwargs.get("json_schema"),
            )
            _loop = asyncio.get_running_loop()
            # see chat() — template via the engine (role remap + BOS guard).
            _mtp_prompt = self._apply_chat_template(messages, enable_thinking)
            _r = await _loop.run_in_executor(
                get_mlx_executor(),
                lambda: self._mlxvlm_mtp.generate(
                    messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    prompt=_mtp_prompt,
                ),
            )
            _txt = _r.get("text", "")
            if stop:
                for _s in stop:
                    if _s and _s in _txt:
                        _txt = _txt[: _txt.find(_s)]
                        break
            _ct = _r.get("completion_tokens", 0)
            yield GenerationOutput(
                text=_txt,
                new_text=_txt,
                # real prompt_tokens (was hardcoded 0 → under-billing).
                prompt_tokens=self._mtp_prompt_tokens(_r, messages, enable_thinking),
                completion_tokens=_ct,
                finished=True,
                finish_reason="length" if _ct >= max_tokens else "stop",
            )
            return
        prompt = self._apply_chat_template(messages, enable_thinking)
        async for output in self.stream_generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            logit_bias=logit_bias,
            stop=stop,
            **kwargs,
        ):
            yield output

    async def _warm_prompt_prefill(self) -> None:
        """Prefill KV cache with warm prompts on startup.

        Delegates to ModelWarmupManager.warm_prompt_prefill() which:
        - Reads prompts from YUNSHU_WARM_PROMPTS env var (||-separated)
        - Tokenizes and prefills each into KV prefix cache
        - Future requests with matching prefixes get instant cache hits

        Provides 1.3-2.25x TTFT improvement on first real request with
        matching prefix (vllm-mlx pattern).
        """
        from .model_optimizations import ModelWarmupManager

        mgr = ModelWarmupManager()

        # Quick check: skip entirely if no warm prompts configured
        if not mgr.resolve_warm_prompts():
            return

        from .mlx_executor import get_mlx_executor

        executor = get_mlx_executor()
        loop = asyncio.get_running_loop()

        def _prefill_all():
            return mgr.warm_prompt_prefill(
                model=self._model,
                tokenizer=self._tokenizer,
                kv_prefix_cache=self._kv_prefix_cache,
                max_tokens=1,
                mem_pressure_threshold=self._mem_pressure_threshold,
            )

        try:
            result = await loop.run_in_executor(executor, _prefill_all)
            # Store result in engine stats
            self._warm_prompt_stats = {
                "prompts_loaded": result.prompts_loaded,
                "prompts_prefilled": result.prompts_prefilled,
                "prompts_skipped_cached": result.prompts_skipped_cached,
                "prompts_failed": result.prompts_failed,
                "total_tokens_prefilled": result.total_tokens_prefilled,
                "prefill_time_s": result.prefill_time_s,
                "source": result.source,
            }
            if result.prompts_prefilled > 0:
                logger.info(
                    f"Warm prompt prefill: {result.prompts_prefilled} prefilled, "
                    f"{result.prompts_skipped_cached} cached, "
                    f"{result.total_tokens_prefilled} tokens, "
                    f"{result.prefill_time_s:.3f}s"
                )
        except Exception as e:
            logger.warning(f"Warm prompt prefill failed: {e}")

    def _init_spec_decode(self) -> None:
        """Initialize speculative decoding if the model supports it.

        Called during start() after model loading. Checks for spec heads in the
        model config and creates a SpeculativeDecoder if detected.
        Also initializes N-gram proposer as a model-free fallback.
        """
        # Always initialize N-gram proposer (model-free, zero overhead when idle)
        import os

        if os.environ.get("YUNSHU_NGRAM_SPEC", "").strip() not in ("0", "false", "no"):
            from .ngram_proposer import NgramConfig, NgramProposer

            max_n = int(os.environ.get("YUNSHU_NGRAM_MAX_N", "5"))
            k = int(os.environ.get("YUNSHU_NGRAM_K", "5"))
            mode = os.environ.get("YUNSHU_NGRAM_MODE", "lps").strip()
            self._ngram_proposer = NgramProposer(
                NgramConfig(max_n=max_n, k=k, mode=mode)
            )
            # Default-on for greedy requests (lossless, never slower — see the
            # routing gate in generate()). Opt out with YUNSHU_NGRAM_DEFAULT=0.
            self._ngram_greedy_default = os.environ.get(
                "YUNSHU_NGRAM_DEFAULT", "1"
            ).strip().lower() not in ("0", "false", "no")
            logger.info(
                "N-gram proposer initialized: max_n=%d, k=%d, mode=%s, greedy_default=%s",
                max_n,
                k,
                mode,
                self._ngram_greedy_default,
            )

        # Adaptive spec controller (requires N-gram proposer active)
        if self._ngram_proposer is not None:
            from .adaptive_spec import AdaptiveSpecController

            self._adaptive_spec = AdaptiveSpecController.from_env()

        # SpecPrefill: sparse (lossy) prefill for long prompts — keeps only the
        # most "important" prompt tokens (scored by a small draft model) to cut
        # TTFT on very long contexts. Opt-in via YUNSHU_SPEC_PREFILL=1 AND a draft
        # model (YUNSHU_SPEC_PREFILL_DRAFT_MODEL, falling back to YUNSHU_DRAFT_MODEL).
        # The fast-path block that consumes this REQUIRES _spec_prefill_draft_model
        # to be non-None — without a draft model the flag would silently do nothing,
        # so we either wire a real draft here or warn loudly that it stays inactive.
        if os.environ.get("YUNSHU_SPEC_PREFILL", "").strip() in ("1", "true", "yes"):
            self._spec_prefill_threshold = int(
                os.environ.get("YUNSHU_SPEC_PREFILL_THRESHOLD", "8192")
            )
            self._spec_prefill_keep_rate = float(
                os.environ.get("YUNSHU_SPEC_PREFILL_KEEP_RATE", "0.20")
            )
            sp_draft_path = (
                os.environ.get("YUNSHU_SPEC_PREFILL_DRAFT_MODEL", "").strip()
                or os.environ.get("YUNSHU_DRAFT_MODEL", "").strip()
            )
            if sp_draft_path:
                try:
                    from mlx_lm.utils import load as load_model

                    self._spec_prefill_draft_model, _ = load_model(sp_draft_path)
                    self._spec_prefill_enabled = True
                    logger.info(
                        "SpecPrefill active: draft=%s threshold=%d keep_rate=%.2f",
                        sp_draft_path,
                        self._spec_prefill_threshold,
                        self._spec_prefill_keep_rate,
                    )
                except Exception as e:  # noqa: BLE001
                    logger.warning(
                        "SpecPrefill requested but draft model %r failed to load "
                        "(%s) — SpecPrefill stays INACTIVE.",
                        sp_draft_path,
                        e,
                    )
            else:
                logger.warning(
                    "YUNSHU_SPEC_PREFILL=1 but no draft model set "
                    "(YUNSHU_SPEC_PREFILL_DRAFT_MODEL / YUNSHU_DRAFT_MODEL) — "
                    "SpecPrefill needs one to score token importance and stays "
                    "INACTIVE. The fast path runs normal full prefill."
                )

        # Gemma-4 assistant drafter (external EAGLE-style drafter, KV-shared with
        # target). Independent of target spec heads, so initialize before the
        # head-detection early return below.
        self._init_gemma4_assistant_spec()

        from .speculative_decoder import auto_configure_speculative, detect_spec_heads

        model_config = {}
        config_obj = getattr(self._model, "config", None) or getattr(
            self._model, "args", None
        )
        if config_obj is not None:
            if hasattr(config_obj, "to_dict"):
                model_config = config_obj.to_dict()
            elif hasattr(config_obj, "__dict__"):
                model_config = {
                    k: v
                    for k, v in config_obj.__dict__.items()
                    if not k.startswith("_")
                }

        head_info = detect_spec_heads(model_config)

        # : an EXTERNAL cross-model draft (YUNSHU_DRAFT_MODEL /
        # config draft_model_path) is independent of the target's native spec
        # heads — it works for ANY model. Load it BEFORE the no-native-heads
        # early return below, otherwise an explicitly-configured draft model was
        # silently ignored for models without native heads (e.g. plain Qwen2.5),
        # and a `spec_decode: true` request fell through to the n-gram path.
        draft_path = os.environ.get("YUNSHU_DRAFT_MODEL", "").strip()
        if not draft_path and model_config:
            draft_path = model_config.get("draft_model_path", "")
        if draft_path:
            try:
                from mlx_lm.utils import load as load_model

                draft_model, _ = load_model(draft_path)
                from .speculative_decoder import SpeculativeDecoder

                self._spec_decoder = SpeculativeDecoder(
                    self._model,
                    draft_model,
                    self._tokenizer,
                    lookahead=self._lookahead_reasoning,
                )
                self._spec_enabled = True
                logger.info(
                    f"Draft model loaded from {draft_path}: "
                    f"speculative decoding ACTIVE (type={head_info.head_type})"
                )
            except Exception as e:
                logger.warning(
                    f"Draft model load failed ({e}), speculative decoding disabled"
                )

        if head_info.head_type == "none":
            logger.debug("No speculative decoding heads detected (native)")
            return  # external draft, if any, already loaded above

        spec_config = auto_configure_speculative(model_config)
        if spec_config.draft_length == 0:
            return

        logger.info(
            f"Speculative decoding available: type={head_info.head_type}, "
            f"draft_length={spec_config.draft_length}"
        )

        # the home-grown MTP decoder path is DEPRECATED — it
        # lacked mlx-vlm's GatedDeltaNet intermediate-state capture (garbage on
        # 27B, ~0.9x on 9B). Use YUNSHU_MTP=1 for the supported mlx-vlm MTP backend.
        # HONESTY: mlx-vlm MTP is EXPERIMENTAL (the "1.8x" figure is proof-script
        # only, not served/gated; the path drops sampling params + is non-streaming) — not a
        # shipped prod win.
        if head_info.head_type == "mtp" and os.environ.get(
            "YUNSHU_LEGACY_MTP", "0"
        ).strip() not in ("1", "true", "yes"):
            logger.info(
                "Native MTP head detected on %s, but the home-grown MTP path is "
                "deprecated — set YUNSHU_MTP=1 for the (experimental) mlx-vlm MTP backend "
                "or YUNSHU_LEGACY_MTP=1 for the old path.",
                self.model_name,
            )
        elif head_info.head_type == "mtp" and self._spec_decoder is None:
            try:
                # Load MTP head weights if available
                inner = getattr(self._model, "language_model", self._model)
                if not hasattr(inner, "mtp"):
                    try:
                        from .mtp_patch import load_model_with_mtp

                        model_name_or_path = model_config.get(
                            "_name_or_path", self.model_name
                        )
                        self._model = load_model_with_mtp(model_name_or_path)
                        logger.info("MTP head weights loaded from model directory")
                    except FileNotFoundError as e:
                        logger.info(
                            f"MTP weights not found ({e}), using backbone-only MTP"
                        )
                    except Exception as e:
                        logger.warning(
                            f"MTP weights load failed ({e}), using backbone-only MTP"
                        )

                from .mtp_decoder import MTPConfig, MTPDecoder

                mtp_config = MTPConfig(
                    max_tokens=256,
                    cooldown_on_reject=os.environ.get("YUNSHU_MTP_COOLDOWN", "").strip()
                    in ("1", "true", "yes"),
                    fastmtp_top_k=int(os.environ.get("YUNSHU_MTP_FASTMTP_TOP_K", "0")),
                    use_n_confirmed=True,
                )
                self._mtp_decoder = MTPDecoder(
                    self._model,
                    self._tokenizer,
                    mtp_config,
                )
                # (: the _mtp_strategy wrapper was dead — routing uses
                # _mtp_decoder directly; removed the unread MTPStrategy build.)
                self._spec_enabled = True
                logger.info(
                    f"MTP decoder initialized: heads={head_info.num_heads}, "
                    f"draft_length={head_info.draft_length}, "
                    f"n_confirmed=True, config={head_info.head_config}"
                )
            except Exception as e:
                logger.warning(f"MTP decoder init failed ({e})")

        # NOTE : Medusa, and the unified SpecStrategyFactory strategies
        # (GPUNgram/LLM/Suffix/DFlash/Composite), were built into engine state but
        # NEVER consulted by generate()/generate_stream() routing — only by the
        # now-deleted _get_spec_strategy() (zero serving callers). They were dead
        # code claiming to be working spec strategies. Removed to keep the spec
        # surface honest. The real, reachable spec paths are: N-gram (default),
        # MTP (model with prediction heads), cross-model (YUNSHU_DRAFT_MODEL), and
        # the Gemma-4 assistant drafter (YUNSHU_GEMMA4_ASSISTANT). Medusa/the
        # factory strategies remain available as research modules in their own
        # files but are not wired into serving.

        # Store config for on-demand decoder creation
        self._spec_config = spec_config
        self._spec_head_info = head_info
        self._spec_enabled = True

    def _init_gemma4_assistant_spec(self) -> None:
        """Build the Gemma-4 dual-load assistant drafter when configured.

        Gated by ``YUNSHU_GEMMA4_ASSISTANT`` (path to the assistant-drafter dir).
        The drafter is an external EAGLE-style proposer that shares the target's
        KV cache; it consumes the target's token embedding + last hidden state
        (not an autonomous LM), so it is driven via ``spec_decode_generate``
        rather than the scheduler's generic draft-model path. Validated ~1.27×
        (greedy-only). Graceful no-op when unset or on any load failure.
        """
        import os

        drafter_dir = os.environ.get("YUNSHU_GEMMA4_ASSISTANT", "").strip()
        if not drafter_dir:
            return
        try:
            inner = getattr(self._model, "language_model", self._model)
            tm = getattr(inner, "model", inner)
            embed = tm.embed_tokens
            embed_scale = float(getattr(tm, "embed_scale", 1.0))
            # Read config.json directly: the loaded mlx_lm config object flattens
            # away nested keys (text_config.layer_types, num_kv_shared_layers) that
            # the drafter's KV-share resolution needs.
            import json as _json
            from pathlib import Path

            mp = Path(self.model_name)
            if not (mp / "config.json").exists():
                from mlx_lm.utils import hf_repo_to_path

                mp = Path(hf_repo_to_path(self.model_name))
            tcfg = _json.loads((mp / "config.json").read_text())
            from .gemma4_assistant import Gemma4AssistantProposer

            self._gemma4_assistant_proposer = Gemma4AssistantProposer.from_paths(
                drafter_dir, embed.weight, embed_scale, tcfg
            )
            self._spec_enabled = True
            logger.info(
                f"Gemma-4 assistant drafter loaded from {drafter_dir}: "
                f"spec decode primitive ACTIVE (sliding_kv={self._gemma4_assistant_proposer.sliding_kv_layer}, "
                f"full_kv={self._gemma4_assistant_proposer.full_kv_layer})"
            )
        except Exception as e:
            logger.warning(f"Gemma-4 assistant drafter init failed ({e}), skipping")
            self._gemma4_assistant_proposer = None

    def gemma4_spec_generate(
        self,
        prompt_ids: list[int],
        max_tokens: int = 256,
        k: int = 4,
        temperature: float = 0.0,
        seed: int | None = None,
    ) -> list[int]:
        """Single-sequence generation via the Gemma-4 assistant drafter.

        Reachable serving path for the dual-load spec-decode primitive (validated
        2.08×): builds the target KV cache and drives ``spec_decode_generate``.
        ``temperature == 0`` → greedy (target's greedy sequence); ``> 0`` →
        distribution-correct speculative sampling. Requires
        ``YUNSHU_GEMMA4_ASSISTANT`` to have been set so the proposer was built at
        start(). Raises RuntimeError if the proposer is unavailable.
        """
        if self._gemma4_assistant_proposer is None:
            raise RuntimeError(
                "Gemma-4 assistant drafter not initialized "
                "(set YUNSHU_GEMMA4_ASSISTANT to the drafter dir before start())"
            )
        from mlx_lm.models.cache import make_prompt_cache

        inner = getattr(self._model, "language_model", self._model)
        tm = getattr(inner, "model", inner)
        eos: set[int] = set()
        eid = getattr(self._tokenizer, "eos_token_id", None)
        if isinstance(eid, (list, tuple)):
            eos.update(eid)
        elif eid is not None:
            eos.add(eid)
        cache = make_prompt_cache(self._model)
        # apply Gemma's final_logit_softcapping to the target
        # lm_head. Model.__call__ does tanh(logits/cap)*cap (cap=30), but the spec path
        # samples straight from embed_tokens.as_linear (raw logits) → softcap bypassed.
        # Greedy (temp=0) is unaffected (softcap is monotonic), but for temp>0 the raw
        # logit spread is wider, so the spec sampling distribution diverged from normal
        # generation. Wrapping restores parity; verify acceptance (greedy-exact) is
        # unchanged since argmax is preserved.
        _softcap = getattr(self._model, "_yunshu_final_softcap", None)
        _base_head = tm.embed_tokens.as_linear
        if _softcap:
            import mlx.core as _mx_sc

            _cap = float(_softcap)

            def _lm_head(_h, _f=_base_head, _c=_cap, _mx=_mx_sc):
                return _mx.tanh(_f(_h) / _c) * _c
        else:
            _lm_head = _base_head
        return self._gemma4_assistant_proposer.spec_decode_generate(
            tm,
            _lm_head,
            cache,
            prompt_ids,
            max_tokens=max_tokens,
            k=k,
            eos_ids=eos,
            temperature=temperature,
            seed=seed,
        )

    def _gemma4_spec_eligible(
        self,
        *,
        logprobs,
        json_schema,
        logits_processors,
        logit_bias,
        top_p,
        top_k,
        min_p,
        repetition_penalty,
        frequency_penalty,
        presence_penalty,
        xtc_probability,
        lora_adapter=None,
    ) -> bool:
        """Whether an n=1 request can be served by the Gemma-4 spec primitive.

        Only routes when the spec output would be IDENTICAL to normal generation:
        the primitive supports greedy / pure-temperature sampling but not
        top_p/top_k/min_p/penalties/grammar/logprobs/logit_bias/xtc. Gating those
        out guarantees no silent behavior change. Active only when the assistant
        drafter was loaded (``YUNSHU_GEMMA4_ASSISTANT``).

        ALSO gate out lora_adapter. `_generate_gemma4_
        assistant_spec` runs the base-weight drafter primitive and does NOT apply
        any adapter, so routing a LoRA request here would silently generate from
        the BASE model — violating the "identical to normal generation" contract.
        LoRA requests must fall through to `_generate_fast` (which applies the
        adapter atomically on the executor via the keystone).
        """
        return (
            getattr(self, "_gemma4_assistant_proposer", None) is not None
            and not logprobs
            and json_schema is None
            and not logits_processors
            and not logit_bias
            and not lora_adapter
            and (top_p is None or top_p >= 1.0)
            and (not top_k or top_k <= 0)
            and (not min_p or min_p <= 0.0)
            and repetition_penalty == 1.0
            and frequency_penalty == 0.0
            and presence_penalty == 0.0
            and (not xtc_probability or xtc_probability <= 0.0)
        )

    async def _generate_gemma4_assistant_spec(
        self,
        prompt: str | list[dict],
        max_tokens: int = 256,
        temperature: float = 0.0,
        seed: int | None = None,
        enable_thinking: bool | None = None,
        stop_token_ids: list[int] | None = None,
        k: int = 4,
    ) -> GenerationOutput:
        """n=1 serving via the Gemma-4 assistant drafter (validated ~2.5x, lossless).

        Builds prompt_ids, runs the proven ``spec_decode_generate`` primitive on
        the single MLX executor thread, and returns a GenerationOutput. The
        primitive is eos-terminated; stop-STRING truncation is applied by the
        gateway post-hoc (same contract as the fast path). Eligibility is gated by
        ``_gemma4_spec_eligible`` so this is only reached when the result is
        identical to normal generation.
        """
        tokenizer = self._tokenizer
        if isinstance(prompt, str):
            text = prompt
            input_ids = tokenizer.encode(text)
        elif isinstance(prompt, list) and prompt and isinstance(prompt[0], dict):
            # : route through the engine's _apply_chat_template + _encode_prompt
            # (NOT raw tokenizer calls) so this spec path matches the default fast path
            # exactly — role remap (developer→system, function→tool), the gemma-4 family
            # message adapter (system-message merge), dangling-<think> close, tool_calls
            # arguments str→dict normalization, AND the double-BOS guard. Bypassing them
            # diverged tool/multi-role prompts from normal generation (violating this
            # route's "identical to normal generation" contract).
            text = self._apply_chat_template(prompt, enable_thinking=enable_thinking)
            input_ids = self._encode_prompt(tokenizer, text)
        else:
            text = str(prompt)
            input_ids = tokenizer.encode(text)
        prompt_tokens = len(input_ids)

        eos_ids: set[int] = set()
        eid = getattr(tokenizer, "eos_token_id", None)
        if isinstance(eid, (list, tuple)):
            eos_ids.update(eid)
        elif eid is not None:
            eos_ids.add(eid)
        if stop_token_ids:
            eos_ids.update(stop_token_ids)

        from .mlx_executor import get_mlx_executor

        executor = get_mlx_executor()
        loop = asyncio.get_running_loop()
        t0 = time.monotonic()

        def _run():
            gen_ids = self.gemma4_spec_generate(
                input_ids,
                max_tokens=max_tokens,
                k=k,
                temperature=temperature or 0.0,
                seed=seed,
            )
            finished_by_eos = bool(gen_ids) and gen_ids[-1] in eos_ids
            visible_ids = gen_ids[:-1] if finished_by_eos else gen_ids
            # Detokenize via the per-request detokenizer, skipping special tokens
            # (e.g. Gemma-4's reserved id 100 -> "<|channel>") so the rendered text
            # is identical to normal generation. The raw detokenizer does NOT skip
            # them, so filter all_special_ids first. completion_tokens still counts
            # every generated token. Runs on the single executor thread.
            _special = set(getattr(tokenizer, "all_special_ids", None) or [])
            detok = tokenizer.detokenizer
            detok.reset()
            for tok in visible_ids:
                if tok in _special:
                    continue
                detok.add_token(tok)
            detok.finalize()
            return visible_ids, detok.text, finished_by_eos

        visible_ids, out_text, finished_by_eos = await loop.run_in_executor(
            executor, _run
        )
        ttft_ms = (time.monotonic() - t0) * 1000.0
        out_text = _clean_special_tokens(out_text)
        # (self-audit): record ServerMetrics. This is the ONE reachable
        # non-streaming spec route (YUNSHU_GEMMA4_ASSISTANT); the gateway stopped
        # double-recording (37568f5), so without this the request is UNCOUNTED —
        # same defect class as the VLM non-streaming metrics loss.
        try:
            from .server_metrics import get_server_metrics

            get_server_metrics().record_request_complete(
                prompt_tokens=prompt_tokens,
                completion_tokens=len(visible_ids),
                prefill_duration=ttft_ms / 1000.0,
                model_id=self.model_name,
            )
        except Exception:
            logger.debug(
                "ServerMetrics record failed (gemma4 assistant spec)", exc_info=True
            )
        return GenerationOutput(
            text=out_text,
            new_text=out_text,
            prompt_tokens=prompt_tokens,
            completion_tokens=len(visible_ids),
            finished=True,
            finish_reason="stop" if finished_by_eos else "length",
            ttft_ms=ttft_ms,
            cached_tokens=0,
        )

    async def _generate_speculative(
        self,
        prompt: str | list[dict],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 1.0,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        logit_bias: dict[int, float] | None = None,
        stop: list[str] | None = None,
        stop_token_ids: list[int] | None = None,
        seed: int | None = None,
        enable_thinking: bool | None = None,
        logprobs: bool = False,
        top_logprobs: int | None = None,
        thinking_budget: int | None = None,
        xtc_probability: float = 0.0,
        xtc_threshold: float = 0.0,
        json_schema: dict | str | None = None,
        cancel_event: asyncio.Event | None = None,
        logits_processors: list | None = None,
        timeout_seconds: float = 300.0,
        lora_adapter: str | None = None,
    ) -> GenerationOutput:
        """Generate using speculative decoding (single-request EAGLE-3 path).

        This path bypasses the continuous batching scheduler and runs the
        SpeculativeDecoder directly. Best for single-request scenarios where
        the draft model can propose K tokens for the target to verify.
        """
        if self._spec_decoder is None:
            # Fall back to standard generation if no decoder
            return await self._generate_fast(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                repetition_penalty=repetition_penalty,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                logit_bias=logit_bias,
                stop=stop,
                stop_token_ids=stop_token_ids,
                seed=seed,
                enable_thinking=enable_thinking,
                logprobs=logprobs,
                top_logprobs=top_logprobs,
                thinking_budget=thinking_budget,
                xtc_probability=xtc_probability,
                xtc_threshold=xtc_threshold,
                json_schema=json_schema,
                cancel_event=cancel_event,
                logits_processors=logits_processors,
                timeout_seconds=timeout_seconds,
                lora_adapter=lora_adapter,
            )

        from .mlx_executor import get_mlx_executor

        executor = get_mlx_executor()
        loop = asyncio.get_running_loop()

        # Tokenize prompt — handle messages-format (list of dicts) like _generate_fast.
        # route through _apply_chat_template + _encode_prompt so this matches the
        # fast path AND applies the double-BOS guard (raw apply_chat_template+encode prepends
        # BOS twice for Gemma/Llama-3/Mistral, corrupting the first-token distribution).
        if isinstance(prompt, list) and prompt and isinstance(prompt[0], dict):
            text = self._apply_chat_template(prompt, enable_thinking=enable_thinking)
            input_ids = self._encode_prompt(self._tokenizer, text)
        else:
            text = prompt if isinstance(prompt, str) else str(prompt)
            input_ids = self._tokenizer.encode(text)
        import mlx.core as mx

        if seed is not None:
            mx.random.seed(seed)

        input_array = mx.array(input_ids).reshape(1, -1)

        # Build EOS + stop token sets
        eos_ids = set()
        if hasattr(self._tokenizer, "eos_token_id"):
            eid = self._tokenizer.eos_token_id
            if isinstance(eid, (list, tuple)):
                eos_ids.update(eid)
            elif eid is not None:
                eos_ids.add(eid)
        if stop_token_ids:
            eos_ids.update(stop_token_ids)
        if stop:
            for s in stop:
                try:
                    ids = self._tokenizer.encode(s)
                    if len(ids) == 1:
                        eos_ids.add(ids[0])
                except Exception:
                    logger.debug(
                        f"failed to encode stop sequence: {s!r}", exc_info=True
                    )

        # Run speculative generation on the MLX executor thread
        # Use incremental detokenizer for correct multi-byte UTF-8
        detokenizer = self._tokenizer.detokenizer
        detokenizer.reset()

        # Wire grammar constraint into spec decoder for structured output.
        # route regex/choice/cfg correctly (was always JsonSchemaConstraint →
        # silently dropped non-JSON grammars).
        _spec_constraint = None
        if json_schema is not None:
            try:
                _spec_constraint = _build_grammar_constraint(
                    json_schema, self._tokenizer
                )
            except Exception:
                logger.warning(
                    "Grammar constraint setup failed for spec decode", exc_info=True
                )
        _prev_constraint = self._spec_decoder.constraint
        self._spec_decoder.constraint = _spec_constraint

        # (LoRA concurrency keystone): the adapter lifecycle runs inside
        # _run_spec_with_lora on the executor thread (serialized with generation), not here
        # on the event loop. _lora_applied stays False so the legacy event-loop release
        # blocks below are inert no-ops.
        _lora_applied = False

        def _run_spec():
            token_ids = self._spec_decoder.generate(
                input_ids=input_array,
                max_tokens=max_tokens,
                temperature=temperature,
                cancel_event=cancel_event,
                repetition_penalty=repetition_penalty,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                logit_bias=logit_bias,
            )
            # Apply stop token truncation (exclude stop token from output)
            for i, tid in enumerate(token_ids):
                if tid in eos_ids:
                    # Add tokens before the stop token to detokenizer
                    for j in range(i):
                        detokenizer.add_token(token_ids[j])
                    return token_ids[:i], True
            # No stop token found — add all tokens to detokenizer
            for tid in token_ids:
                detokenizer.add_token(tid)
            return token_ids, False

        def _run_spec_with_lora():
            # adapter acquire+apply / release+restore on the executor thread.
            _applied = False
            if lora_adapter and getattr(self, "_lora_manager", None) is not None:
                try:
                    _applied = self._lora_manager.acquire_adapter(lora_adapter)
                except Exception as _le:  # fail loud, don't serve base
                    raise RuntimeError(
                        f"LoRA adapter '{lora_adapter}' could not be applied"
                    ) from _le
                if not _applied:
                    raise RuntimeError(
                        f"LoRA adapter '{lora_adapter}' could not be applied"
                    )
            try:
                return _run_spec()
            finally:
                if _applied and getattr(self, "_lora_manager", None) is not None:
                    try:
                        self._lora_manager.release_adapter(lora_adapter)
                    except Exception:
                        logger.debug(
                            "LoRA release failed (spec executor)", exc_info=True
                        )

        _spec_gen_t0 = time.perf_counter()
        try:
            token_ids, hit_stop = await asyncio.wait_for(
                loop.run_in_executor(executor, _run_spec_with_lora),
                timeout=timeout_seconds,
            )
        except TimeoutError:
            logger.warning(f"Speculative generation timed out after {timeout_seconds}s")
            try:
                import mlx.core as _mx

                await loop.run_in_executor(
                    executor, lambda: (_mx.synchronize(), _mx.clear_cache())
                )
            except Exception:
                logger.debug(
                    "GPU cache cleanup failed after spec decode timeout", exc_info=True
                )
            self._spec_decoder.constraint = _prev_constraint
            if (
                _lora_applied
                and hasattr(self, "_lora_manager")
                and self._lora_manager is not None
            ):
                try:
                    self._lora_manager.release_adapter(lora_adapter)
                except Exception:
                    logger.warning(
                        "LoRA release failed after spec decode timeout", exc_info=True
                    )
            return GenerationOutput(
                finished=True,
                finish_reason="error",
                prompt_tokens=len(input_ids),
                completion_tokens=0,
                error=f"Speculative generation timed out after {timeout_seconds}s",
                ttft_ms=0.0,
                cached_tokens=0,
            )
        except MemoryError:
            logger.warning(
                "OOM during speculative generation — returning memory_limit finish reason"
            )
            try:
                import mlx.core as _mx

                await loop.run_in_executor(
                    executor, lambda: (_mx.synchronize(), _mx.clear_cache())
                )
            except Exception:
                logger.debug(
                    "GPU cache cleanup failed after spec decode OOM", exc_info=True
                )
            self._spec_decoder.constraint = _prev_constraint
            if (
                _lora_applied
                and hasattr(self, "_lora_manager")
                and self._lora_manager is not None
            ):
                try:
                    self._lora_manager.release_adapter(lora_adapter)
                except Exception:
                    logger.warning(
                        "LoRA release failed after spec decode OOM", exc_info=True
                    )
            return GenerationOutput(
                finished=True,
                finish_reason="memory_limit",
                prompt_tokens=len(input_ids),
                completion_tokens=0,
                error="OOM during speculative generation",
                ttft_ms=0.0,
                cached_tokens=0,
            )
        except RuntimeError as e:
            if "memory" in str(e).lower() or "out of" in str(e).lower():
                logger.warning(f"MLX OOM during speculative generation: {e}")
                try:
                    import mlx.core as _mx

                    await loop.run_in_executor(
                        executor, lambda: (_mx.synchronize(), _mx.clear_cache())
                    )
                except Exception:
                    logger.debug(
                        "GPU cache cleanup failed after spec decode OOM (RuntimeError)",
                        exc_info=True,
                    )
                self._spec_decoder.constraint = _prev_constraint
                if (
                    _lora_applied
                    and hasattr(self, "_lora_manager")
                    and self._lora_manager is not None
                ):
                    try:
                        self._lora_manager.release_adapter(lora_adapter)
                    except Exception:
                        logger.warning(
                            "LoRA release failed after spec decode OOM (RuntimeError)",
                            exc_info=True,
                        )
                return GenerationOutput(
                    finished=True,
                    finish_reason="memory_limit",
                    prompt_tokens=len(input_ids),
                    completion_tokens=0,
                    error=str(e),
                    ttft_ms=0.0,
                    cached_tokens=0,
                )
            self._spec_decoder.constraint = _prev_constraint
            if (
                _lora_applied
                and hasattr(self, "_lora_manager")
                and self._lora_manager is not None
            ):
                try:
                    self._lora_manager.release_adapter(lora_adapter)
                except Exception:
                    logger.warning(
                        "LoRA release failed after spec decode RuntimeError",
                        exc_info=True,
                    )
            # Return error output for non-OOM RuntimeError instead of
            # propagating to caller (which expects GenerationOutput).
            return GenerationOutput(
                finished=True,
                finish_reason="error",
                prompt_tokens=len(input_ids),
                completion_tokens=0,
                error=f"RuntimeError during speculative generation: {e}",
                ttft_ms=0.0,
                cached_tokens=0,
            )
        except Exception as e:
            logger.error(
                f"Unexpected error during speculative generation: {e}", exc_info=True
            )
            self._spec_decoder.constraint = _prev_constraint
            if (
                _lora_applied
                and hasattr(self, "_lora_manager")
                and self._lora_manager is not None
            ):
                try:
                    self._lora_manager.release_adapter(lora_adapter)
                except Exception:
                    logger.warning(
                        "LoRA release failed after spec decode unexpected error",
                        exc_info=True,
                    )
            # Return error output instead of propagating exception to caller.
            return GenerationOutput(
                finished=True,
                finish_reason="error",
                prompt_tokens=len(input_ids),
                completion_tokens=0,
                error=f"Speculative generation failed: {e}",
                ttft_ms=0.0,
                cached_tokens=0,
            )
        _spec_ttft_s = time.perf_counter() - _spec_gen_t0
        detokenizer.finalize()

        text = _clean_special_tokens(detokenizer.text)

        # Build logprobs from individual tokens — speculative decode black-box
        # generate() does not expose logits, so we cannot compute real logprobs.
        # Return None instead of fake 0.0 to avoid misleading consumers.
        _logprobs = None
        if logprobs and token_ids:
            _logprobs = None  # Real logprobs unavailable from spec decode black-box

        # Record TTFT in Prometheus for spec decode path
        if _spec_ttft_s > 0:
            try:
                from yunshu_gateway.middleware.prometheus_exporter import (
                    get_prometheus_metrics,
                )

                pm = get_prometheus_metrics()
                pm.observe_histogram(
                    "ttft_seconds", _spec_ttft_s, labels={"model_id": self.model_label}
                )
            except Exception:
                logger.debug(
                    "TTFT prometheus recording failed in spec decode path",
                    exc_info=True,
                )

        # Determine finish_reason with cancel awareness
        _cancelled = _is_cancelled(cancel_event)
        if _cancelled or hit_stop or len(token_ids) < max_tokens:
            _finish_reason = "stop"
        else:
            _finish_reason = "length"

        # Reasoning parser: extract thinking tokens from spec decode output.
        # Speculative decode uses a black-box generate() that does not track
        # thinking tokens, so we parse the output text for reasoning content.
        _spec_reasoning_tok = 0
        if text:
            try:
                from .reasoning_parser import get_reasoning_parser

                rp = get_reasoning_parser(self.model_name)
                rp_out = rp.parse(text)
                if rp_out.reasoning and rp_out.reasoning_tokens > 0:
                    _spec_reasoning_tok = rp_out.reasoning_tokens
                    if rp_out.content != text:
                        text = rp_out.content
            except Exception:
                logger.debug(
                    "reasoning_parser failed in spec decode path", exc_info=True
                )

        self._spec_decoder.constraint = _prev_constraint
        if (
            _lora_applied
            and hasattr(self, "_lora_manager")
            and self._lora_manager is not None
        ):
            try:
                self._lora_manager.release_adapter(lora_adapter)
            except Exception:
                logger.warning(
                    "LoRA release failed after spec decode normal completion",
                    exc_info=True,
                )
        return GenerationOutput(
            text=text,
            new_text=text,
            prompt_tokens=len(input_ids),
            completion_tokens=len(token_ids),
            finished=True,
            finish_reason=_finish_reason,
            reasoning_tokens=_spec_reasoning_tok,
            cached_tokens=0,
            logprobs=_logprobs,
            ttft_ms=round(_spec_ttft_s * 1000, 1),
        )

    async def _stream_generate_speculative(
        self,
        prompt: str | list[dict],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 1.0,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        logit_bias: dict[int, float] | None = None,
        logprobs: bool = False,
        top_logprobs: int | None = None,
        stop: list[str] | None = None,
        stop_token_ids: list[int] | None = None,
        seed: int | None = None,
        enable_thinking: bool | None = None,
        thinking_budget: int | None = None,
        xtc_probability: float = 0.0,
        xtc_threshold: float = 0.0,
        json_schema: dict | str | None = None,
        cancel_event: asyncio.Event | None = None,
        logits_processors: list | None = None,
        lora_adapter: str | None = None,
    ) -> AsyncIterator[GenerationOutput]:
        """Stream generate using speculative decoding (single-request path).

        Yields chunks as they are verified by the target model.
        Each yield contains the accepted tokens from one verify step.

        UNREACHABLE in the streaming flow (honesty annotation, mirrors
        _stream_generate_mtp): stream_generate() forces ``spec_decode = False`` at
        the top (batched_engine.py ~4381 — fast-path spec routes are not lossless and give
        no speedup on Apple Silicon), so the only caller (the `if spec_decode and ...`
        block ~4456) never fires. The documented "spec-stream multi-token stop-prefix
        leak" therefore is NOT a live bug — it lives only here in dead code. The LIVE
        streaming path is _stream_generate_fast, which routes deltas through
        StopHoldbackBuffer so no stop prefix leaks. Retrofitting hold-back into
        this complex verify-loop would be risky churn on an unreachable path; if spec
        streaming is ever re-enabled, wrap the per-step emit through StopHoldbackBuffer
        (the _stream_generate_fast _hb pattern) BEFORE shipping.
        """
        if self._spec_decoder is None:
            # Fall back to fast path streaming (avoid recursive dispatch)
            async for output in self._stream_generate_fast(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                repetition_penalty=repetition_penalty,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                logit_bias=logit_bias,
                stop=stop,
                stop_token_ids=stop_token_ids,
                seed=seed,
                enable_thinking=enable_thinking,
                thinking_budget=thinking_budget,
                xtc_probability=xtc_probability,
                xtc_threshold=xtc_threshold,
                json_schema=json_schema,
                cancel_event=cancel_event,
                logprobs=bool(logprobs),
                top_logprobs=top_logprobs,
                logits_processors=logits_processors,
                lora_adapter=lora_adapter,
            ):
                yield output
            return

        from .mlx_executor import get_mlx_executor

        executor = get_mlx_executor()
        loop = asyncio.get_running_loop()

        # Tokenize prompt — handle messages-format (list of dicts) like _generate_fast.
        # route through _apply_chat_template + _encode_prompt so this matches the
        # fast path AND applies the double-BOS guard (raw apply_chat_template+encode prepends
        # BOS twice for Gemma/Llama-3/Mistral, corrupting the first-token distribution).
        if isinstance(prompt, list) and prompt and isinstance(prompt[0], dict):
            text = self._apply_chat_template(prompt, enable_thinking=enable_thinking)
            input_ids = self._encode_prompt(self._tokenizer, text)
        else:
            text = prompt if isinstance(prompt, str) else str(prompt)
            input_ids = self._tokenizer.encode(text)
        import mlx.core as mx

        if seed is not None:
            mx.random.seed(seed)

        input_array = mx.array(input_ids).reshape(1, -1)

        # Get EOS IDs + stop tokens
        eos_ids = set()
        stop_suffixes = []
        if hasattr(self._tokenizer, "eos_token_id"):
            eid = self._tokenizer.eos_token_id
            if isinstance(eid, (list, tuple)):
                eos_ids.update(eid)
            elif eid is not None:
                eos_ids.add(eid)
        if stop_token_ids:
            eos_ids.update(stop_token_ids)
        if stop:
            for s in stop:
                if not s:
                    continue
                try:
                    ids = self._tokenizer.encode(s)
                    if len(ids) == 1:
                        eos_ids.add(ids[0])
                    # : always string-match (bare-stop token id
                    # rarely matches the space-prefixed emitted token).
                    stop_suffixes.append(s)
                except Exception:
                    logger.debug(
                        f"failed to encode stop sequence: {s!r}", exc_info=True
                    )

        _eids = getattr(self._tokenizer, "eos_token_ids", None)
        if _eids is not None:  # may be a bare int (Qwen3.6-27B), not iterable
            eos_ids.update(_eids if isinstance(_eids, (list, tuple, set)) else (_eids,))

        # Run speculative steps on executor thread, yielding after each step
        from mlx_lm.models.cache import make_prompt_cache

        target_cache = make_prompt_cache(self._spec_decoder.target)
        draft_cache = make_prompt_cache(self._spec_decoder.draft)

        # Wire grammar constraint into spec decoder for streaming structured output.
        # Save the previous constraint to restore after this request completes,
        # preventing cross-request constraint mixing on the shared decoder instance.
        # route regex/choice/cfg correctly (was always JsonSchemaConstraint →
        # silently dropped non-JSON grammars).
        _spec_constraint = None
        if json_schema is not None:
            try:
                _spec_constraint = _build_grammar_constraint(
                    json_schema, self._tokenizer
                )
            except Exception:
                logger.warning(
                    "Grammar constraint setup failed for spec streaming", exc_info=True
                )
        _prev_constraint = self._spec_decoder.constraint
        self._spec_decoder.constraint = _spec_constraint

        # LoRA acquire+apply / release MUST happen on the executor thread (inside
        # _prefill / the release closure below), never here on the event loop:
        # acquire_adapter loads+activates the adapter by MUTATING the shared model,
        # and doing that off the executor races a concurrent request's generate_step
        # (wrong-adapter bleed). _lora_state carries the applied flag back so the
        # finally can release on the executor. (Same keystone as _generate_fast.)
        _lora_state = {"applied": False}

        # Thinking budget enforcement — detect <think/</think via single-token IDs
        _spec_think_start_token = None
        _spec_think_end_token = None
        if thinking_budget is not None or enable_thinking:
            # bracketed-form helper (bare "</think" tokenized to 2 → guard failed).
            _spec_think_start_token, _spec_think_end_token = _resolve_think_token_ids(
                self._tokenizer
            )
        _spec_in_thinking = False
        _spec_thinking_tokens_used = 0

        # Prefill both models
        def _prefill():
            # Acquire+apply the LoRA adapter HERE on the executor thread, before any
            # forward pass, serialized with every other request's generate_step.
            if lora_adapter and getattr(self, "_lora_manager", None) is not None:
                try:
                    _lora_state["applied"] = self._lora_manager.acquire_adapter(
                        lora_adapter
                    )
                except Exception as _le:  # fail loud, don't serve base
                    raise RuntimeError(
                        f"LoRA adapter '{lora_adapter}' could not be applied"
                    ) from _le
                if not _lora_state["applied"]:
                    raise RuntimeError(
                        f"LoRA adapter '{lora_adapter}' could not be applied"
                    )
            self._spec_decoder.target(input_array, cache=target_cache)
            self._spec_decoder.draft(input_array, cache=draft_cache)

            # Roll back both caches by 1 position so the last prompt token is
            # NOT in the cache. This prevents double-feeding:
            # - generate_draft will feed current_ids (last prompt token) into
            # the draft cache — correct since it was rolled back.
            # - verify_draft will feed [current_ids, d0, ...] into the target
            # cache — correct since current_ids was rolled back.
            # Without this rollback, both methods would double-feed the last
            # prompt token (it's already in both caches from prefill).
            try:
                from mlx_lm.models.cache import trim_prompt_cache

                trim_prompt_cache(target_cache, 1)
                trim_prompt_cache(draft_cache, 1)
            except Exception:
                logger.debug(
                    "trim_prompt_cache failed, falling back to per-layer trim",
                    exc_info=True,
                )
                for c in target_cache:
                    if hasattr(c, "trim"):
                        c.trim(1)
                for c in draft_cache:
                    if hasattr(c, "trim"):
                        c.trim(1)

        # After prefill + rollback, both caches have prompt[:-1].
        # current_ids is the last prompt token, which will be fed into both
        # caches by generate_draft and verify_draft respectively — no double-feed.
        current_ids = input_array[:, -1:]  # [1, 1] last prompt token

        generated_tokens: list[int] = []
        prompt_tokens = len(input_ids)
        detokenizer = self._tokenizer.detokenizer
        detokenizer.reset()

        try:
            await loop.run_in_executor(executor, _prefill)
        except Exception as e:
            logger.error(f"Spec decode streaming prefill failed: {e}", exc_info=True)
            self._spec_decoder.constraint = _prev_constraint
            if (
                _lora_state["applied"]
                and getattr(self, "_lora_manager", None) is not None
            ):
                try:
                    self._lora_manager.release_adapter(lora_adapter)
                except Exception:
                    logger.warning(
                        "LoRA release failed after spec decode prefill error",
                        exc_info=True,
                    )
            raise
        _spec_gen_t0 = time.perf_counter()  # TTFT timing starts after prefill
        try:
            _spec_ttft_ms_val = 0.0
            _spec_ttft_recorded = False
            while len(generated_tokens) < max_tokens:
                if _is_cancelled(cancel_event):
                    logger.debug("Cancel event triggered during spec decode streaming")
                    # Yield terminal stop chunk so consumer sees finished=True
                    if generated_tokens:
                        detokenizer.finalize()
                        _final_text = _clean_special_tokens(detokenizer.text)
                    else:
                        _final_text = ""
                    yield GenerationOutput(
                        text=_final_text,
                        new_text="",
                        prompt_tokens=prompt_tokens,
                        completion_tokens=len(generated_tokens),
                        finished=True,
                        finish_reason="stop",
                        ttft_ms=_spec_ttft_ms_val,
                        reasoning_tokens=0,
                        cached_tokens=0,
                        logprobs=None,
                    )
                    break

                # Snapshot draft cache for rollback on rejection
                draft_snap = self._spec_decoder._snapshot_cache(draft_cache)

                # Cap draft length to remaining thinking budget
                _effective_spec_K = None
                if thinking_budget is not None and _spec_in_thinking:
                    remaining = thinking_budget - _spec_thinking_tokens_used
                    if remaining <= 0:
                        if _spec_think_end_token is not None:
                            _spec_in_thinking = False
                            generated_tokens.append(_spec_think_end_token)
                            detokenizer.add_token(_spec_think_end_token)
                        break
                    _effective_spec_K = max(1, remaining - 1)

                _spec_target_trimmed = (
                    False  # Set by _spec_step if SP-PEN trims target cache
                )

                def _spec_step():
                    # First iteration: last prompt token (already in cache from prefill,
                    # so forward pass advances the cache by 1 and returns logits).
                    # Subsequent iterations: last accepted/bonus token.
                    draft_result = self._spec_decoder.generate_draft(
                        current_ids, draft_cache, max_draft_tokens=_effective_spec_K
                    )
                    # verify_draft: includes current_ids as alignment token so logits
                    # are correctly positioned. Returns accepted tokens + bonus.
                    # NOTE: verify_draft feeds [current_ids, draft_tokens] to target,
                    # populating target_cache with K+1 tokens. We must NOT re-feed
                    # accepted tokens to target (that would double-populate the cache).
                    verify_result = self._spec_decoder.verify_draft(
                        draft_result,
                        current_ids,
                        target_cache,
                    )

                    # SP-PEN: Apply penalty/bias to bonus token logits.
                    # After verify_draft, target_cache has K+1 entries but only the
                    # first (accepted_count + 1) are valid. Trim the rejected entries
                    # so the cache is consistent, then do a single forward pass for
                    # the bonus position to get fresh logits with penalties applied.
                    _has_penalties = (
                        repetition_penalty != 1.0
                        or frequency_penalty != 0.0
                        or presence_penalty != 0.0
                        or (logit_bias is not None and len(logit_bias) > 0)
                    )
                    if _has_penalties:
                        import mlx.core as _sp_mx

                        K_local = len(draft_result.token_ids)
                        ac = verify_result.accepted_count
                        _already_trimmed = K_local > 0 and ac < K_local
                        if _already_trimmed:
                            from mlx_lm.models.cache import trim_prompt_cache

                            _trim_n = K_local - ac
                            try:
                                trim_prompt_cache(target_cache, _trim_n)
                            except Exception:
                                logger.debug(
                                    "trim_prompt_cache (SP-PEN partial) failed, falling back",
                                    exc_info=True,
                                )
                                for _c in target_cache:
                                    if hasattr(_c, "trim"):
                                        _c.trim(_trim_n)
                            nonlocal _spec_target_trimmed
                            _spec_target_trimmed = True
                        # Feed last accepted token (or current_ids if none accepted) to get bonus logits
                        _bonus_input = (
                            mx.array([[verify_result.accepted_ids[-1]]])
                            if verify_result.accepted_ids
                            else current_ids
                        )
                        _bonus_out = self._spec_decoder.target(
                            _bonus_input, cache=target_cache
                        )
                        _bonus_logits = (
                            _bonus_out.logits
                            if hasattr(_bonus_out, "logits")
                            else _bonus_out
                        )
                        _bonus_logits = _bonus_logits[0, -1, :]
                        # Undo the bonus forward — next iteration's verify_draft will feed
                        # current_ids (= bonus) into target_cache, and we must not have
                        # the SP-PEN probe already in the cache or it double-populates.
                        try:
                            from mlx_lm.models.cache import trim_prompt_cache

                            trim_prompt_cache(target_cache, 1)
                        except Exception:
                            logger.debug(
                                "trim_prompt_cache (SP-PEN bonus undo) failed, falling back",
                                exc_info=True,
                            )
                            for _c in target_cache:
                                if hasattr(_c, "trim"):
                                    _c.trim(1)
                        # Build token history: prompt + all generated so far + accepted drafts
                        _token_hist = (
                            list(input_ids)
                            + generated_tokens
                            + verify_result.accepted_ids
                        )
                        _bonus_logits = _apply_spec_bonus_penalties(
                            _bonus_logits,
                            _token_hist,
                            len(input_ids),
                            repetition_penalty=repetition_penalty,
                            frequency_penalty=frequency_penalty,
                            presence_penalty=presence_penalty,
                            logit_bias=logit_bias,
                        )
                        # Re-sample bonus token from penalized logits using configured sampler
                        from mlx_lm.sample_utils import make_sampler as _make_sp_sampler

                        # mlx-lm make_sampler's first param is `temp`,
                        # not `temperature` — the old kwarg raised TypeError and failed the
                        # whole request (no fallback at the _spec_step call site) whenever
                        # cross-model spec decode ran with any penalty/bias set.
                        _sp_sampler = _make_sp_sampler(
                            temp=temperature,
                            top_p=top_p,
                            top_k=top_k if top_k and top_k > 0 else -1,
                            min_p=min_p if min_p and min_p > 0 else 0.0,
                        )
                        _bonus_id = int(
                            _sp_sampler(
                                _sp_mx.expand_dims(_bonus_logits, axis=(0, 1))
                            ).item()
                        )
                        # Overwrite bonus token in verify_result (simple reconstruction)
                        verify_result = type(verify_result)(
                            accepted_count=verify_result.accepted_count,
                            accepted_ids=verify_result.accepted_ids,
                            rejected_at=verify_result.rejected_at,
                            bonus_token_id=_bonus_id,
                            target_logprobs=verify_result.target_logprobs,
                        )

                    return draft_result, verify_result

                draft_result, verify_result = await loop.run_in_executor(
                    executor, _spec_step
                )

                K = len(draft_result.token_ids)
                accepted_count = verify_result.accepted_count
                new_tokens = verify_result.accepted_ids[:]
                if (
                    verify_result.bonus_token_id is not None
                    and verify_result.bonus_token_id >= 0
                ):
                    new_tokens.append(verify_result.bonus_token_id)

                self._spec_decoder._stats["total_draft_tokens"] += K
                self._spec_decoder._stats["total_accepted_tokens"] += accepted_count
                self._spec_decoder._stats["total_bonus_tokens"] += 1
                self._spec_decoder._stats["total_steps"] += 1
                if self._lookahead_reasoning is not None:
                    self._lookahead_reasoning.record_accept(accepted_count)

                hit_eos = False
                _hit_suffix = False
                _yielded_token_count = (
                    0  # Track how many tokens were actually yielded before break
                )
                for token_id in new_tokens:
                    if token_id in eos_ids:
                        hit_eos = True
                        break
                    # Thinking budget tracking — detect <think/</think via token IDs
                    if _spec_think_start_token is not None:
                        if (
                            not _spec_in_thinking
                            and token_id == _spec_think_start_token
                        ):
                            _spec_in_thinking = True
                        elif _spec_in_thinking:
                            _spec_thinking_tokens_used += 1
                            if token_id == _spec_think_end_token:
                                _spec_in_thinking = False
                    generated_tokens.append(token_id)
                    detokenizer.add_token(token_id)
                    _yielded_token_count += 1
                    # Thinking budget enforcement: force </think when budget exceeded
                    if (
                        thinking_budget is not None
                        and _spec_in_thinking
                        and _spec_thinking_tokens_used >= thinking_budget
                        and _spec_think_end_token is not None
                    ):
                        _spec_in_thinking = False
                        # Force-insert </think token
                        generated_tokens.append(_spec_think_end_token)
                        detokenizer.add_token(_spec_think_end_token)
                        _yielded_token_count += 1
                        # Treat as stop — no more tokens after budget hit
                        hit_eos = True
                        break
                    if stop_suffixes and any(
                        detokenizer.text.endswith(s) for s in stop_suffixes
                    ):
                        _hit_suffix = True
                        # Remove the suffix-triggering token — it should not
                        # appear in the output, matching the non-spec pattern.
                        generated_tokens.pop()
                        _yielded_token_count -= 1
                        # Reset detokenizer to state before the suffix token was
                        # added. Simply popping from detokenizer.tokens is not
                        # sufficient because NaiveStreamingDetokenizer computes
                        # .text from _current_tokens (an internal list), not from
                        # the .tokens attribute. Re-decode all remaining tokens
                        # to produce clean text without the suffix.
                        _kept_tokens = (
                            list(detokenizer.tokens[:-1]) if detokenizer.tokens else []
                        )
                        detokenizer.reset()
                        for _t in _kept_tokens:
                            detokenizer.add_token(_t)
                        break

                # Compute TTFT before first yield
                if not _spec_ttft_recorded:
                    _spec_ttft_recorded = True
                    _spec_ttft_s = time.perf_counter() - _spec_gen_t0
                    _spec_ttft_ms_val = round(_spec_ttft_s * 1000, 1)
                    try:
                        from yunshu_gateway.middleware.prometheus_exporter import (
                            get_prometheus_metrics,
                        )

                        pm = get_prometheus_metrics()
                        pm.observe_histogram(
                            "ttft_seconds",
                            _spec_ttft_s,
                            labels={"model_id": self.model_label},
                        )
                    except Exception:
                        logger.debug(
                            "spec streaming TTFT prometheus recording failed",
                            exc_info=True,
                        )

                # Yield accepted text via incremental detokenizer
                chunk_text = _clean_special_tokens(detokenizer.last_segment)
                finish_reason = None
                if hit_eos or _hit_suffix:
                    finish_reason = "stop"
                elif len(generated_tokens) >= max_tokens:
                    finish_reason = "length"

                # Trim stop suffix from chunk text when matched
                if _hit_suffix and stop_suffixes:
                    for s in stop_suffixes:
                        if chunk_text.endswith(s):
                            chunk_text = chunk_text[: -len(s)]
                            break

                # Build logprobs from target model verification
                # Only include logprobs for tokens that were actually yielded
                # (before hit_eos or _hit_suffix broke the loop). Without this
                # guard, logprobs included EOS and post-break tokens that were
                # never added to generated_tokens / detokenizer.
                _chunk_logprobs = None
                if logprobs and new_tokens:
                    _chunk_logprobs = []
                    _lp_count = min(_yielded_token_count, len(new_tokens))
                    for i in range(_lp_count):
                        tid = new_tokens[i]
                        tok_text = _clean_special_tokens(self._tokenizer.decode([tid]))
                        lp = (
                            verify_result.target_logprobs[i]
                            if i < len(verify_result.target_logprobs)
                            else 0.0
                        )
                        _chunk_logprobs.append(
                            {
                                "token": tok_text,
                                "logprob": float(lp),
                                "top_logprobs": [
                                    {"token": tok_text, "logprob": float(lp)}
                                ],
                            }
                        )

                yield GenerationOutput(
                    text=_clean_special_tokens(detokenizer.text),
                    new_text=chunk_text,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=len(generated_tokens),
                    finished=finish_reason is not None,
                    finish_reason=finish_reason,
                    reasoning_tokens=_spec_thinking_tokens_used
                    if _spec_think_start_token is not None
                    else 0,
                    cached_tokens=0,
                    logprobs=_chunk_logprobs,
                    ttft_ms=_spec_ttft_ms_val,
                )

                if finish_reason is not None:
                    detokenizer.finalize()
                    break

                # Update caches for next iteration:
                # - Target cache: verify_draft already fed [last_tok, d0..dK-1].
                # If all accepted, target has exactly the right state (last_tok + K drafts).
                # If partially accepted, we need to trim rejected tokens from target cache.
                # - Draft cache: if all accepted, draft already has K tokens from generate_draft.
                # If partially accepted, restore snapshot and refeed accepted + bonus.
                def _update_caches():
                    nonlocal draft_snap

                    if accepted_count < K:
                        if not _spec_target_trimmed:
                            # Partial acceptance: trim target cache to remove rejected entries.
                            # verify_draft fed K+1 tokens (last_tok + K drafts).
                            # We want to keep: last_tok + accepted_count drafts = accepted_count + 1
                            # Trim: (K+1) - (accepted_count + 1) = K - accepted_count entries.
                            from mlx_lm.models.cache import trim_prompt_cache

                            trim_count = K - accepted_count
                            try:
                                trim_prompt_cache(target_cache, trim_count)
                            except Exception:
                                logger.debug(
                                    "trim_prompt_cache (partial acceptance) failed, falling back",
                                    exc_info=True,
                                )
                                for c in target_cache:
                                    if hasattr(c, "trim"):
                                        c.trim(trim_count)

                        # Restore draft cache to pre-draft state and refeed accepted tokens.
                        # This MUST run even when SP-PEN already trimmed the target cache,
                        # otherwise the draft cache accumulates incorrect state from rejected
                        # tokens, corrupting subsequent draft generation.
                        # Do NOT refeed the bonus token here — generate_draft will feed
                        # current_ids (= bonus token) at the start of the next iteration,
                        # so including it now would cause a double-feed.
                        self._spec_decoder._restore_cache(draft_cache, draft_snap)
                        for tok in verify_result.accepted_ids:
                            self._spec_decoder.draft(
                                mx.array([[tok]]), cache=draft_cache
                            )

                    # Feed bonus token to both caches (target already has it from verify_draft
                    # when all accepted; when partial, we trimmed and need to re-add).
                    # For draft: bonus token needs to be fed in both cases.
                    # For target: when all accepted, bonus is the last token from verify_draft
                    # logits[K] — already in cache. When partial, we trimmed and re-added
                    # accepted+bonus above, so target is current.

                    # Update current_ids for next iteration
                    return mx.array([[verify_result.bonus_token_id]])

                current_ids = await loop.run_in_executor(executor, _update_caches)
        except GeneratorExit:
            logger.debug("Client disconnected during spec decode streaming")
        except Exception as e:
            logger.error(f"Spec decode streaming error: {e}", exc_info=True)
            # Yield a terminal error output so the consumer sees finished=True
            # instead of a broken stream (exception without terminal output).
            try:
                detokenizer.finalize()
                _err_final_text = _clean_special_tokens(detokenizer.text)
            except Exception:
                _err_final_text = ""
            yield GenerationOutput(
                text=_err_final_text,
                new_text="",
                prompt_tokens=prompt_tokens,
                completion_tokens=len(generated_tokens),
                finished=True,
                finish_reason="error",
                ttft_ms=_spec_ttft_ms_val,
                reasoning_tokens=0,
                cached_tokens=0,
                logprobs=None,
            )
            raise
        finally:
            self._spec_decoder.constraint = _prev_constraint
            if (
                _lora_state["applied"]
                and getattr(self, "_lora_manager", None) is not None
            ):
                try:
                    self._lora_manager.release_adapter(lora_adapter)
                except Exception:
                    logger.warning(
                        "LoRA release failed in spec decode streaming finally",
                        exc_info=True,
                    )

    async def _generate_ngram_spec(
        self,
        prompt: str | list[dict],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 1.0,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        logit_bias: dict[int, float] | None = None,
        stop: list[str] | None = None,
        stop_token_ids: list[int] | None = None,
        seed: int | None = None,
        logprobs: bool = False,
        top_logprobs: int | None = None,
        json_schema: dict | str | None = None,
        xtc_probability: float = 0.0,
        xtc_threshold: float = 0.0,
        enable_thinking: bool | None = None,
        thinking_budget: int | None = None,
        cancel_event: asyncio.Event | None = None,
        logits_processors: list | None = None,
        timeout_seconds: float = 300.0,
        lora_adapter: str | None = None,
    ) -> GenerationOutput:
        """Generate using N-gram speculative decoding (model-free).

        Uses the N-gram proposer to predict K draft tokens, then verifies
        each by running the target model forward. Accepts matching tokens,
        resamples on mismatch.
        """
        import mlx.core as mx
        from mlx_lm.generate import generate_step
        from mlx_lm.models.cache import make_prompt_cache
        from mlx_lm.sample_utils import make_sampler

        from .mlx_executor import get_mlx_executor

        tokenizer = self._tokenizer
        model = self._model

        # SAFETY GUARD: speculative verification rolls the KV cache back
        # to discard rejected drafts, which REQUIRES a trimmable cache. Qwen3.5's
        # hybrid attention uses 18 non-trimmable ArraysCache (linear/gated-attention
        # recurrent state) + 6 KVCache, so can_trim is False — spec there can't roll
        # back rejected drafts (degenerate ".txt.txt..." repetition), and the
        # recurrent state must be driven by ONE continuous generate_step, so we
        # delegate the WHOLE request to the plain fast path. mlx-lm's
        # speculative_generate_step likewise RAISES on a non-trimmable cache.
        try:
            from mlx_lm.models.cache import (
                can_trim_prompt_cache as _can_trim,
            )
            from mlx_lm.models.cache import (
                make_prompt_cache as _mk_cache,
            )

            _spec_cache_supported = _can_trim(_mk_cache(model))
        except Exception:
            _spec_cache_supported = False
        if not _spec_cache_supported:
            logger.info(
                "N-gram spec disabled: model cache is not trimmable — "
                "using the plain fast path."
            )
            return await self._generate_fast(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                repetition_penalty=repetition_penalty,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                logit_bias=logit_bias,
                stop=stop,
                stop_token_ids=stop_token_ids,
                seed=seed,
                enable_thinking=enable_thinking,
                logprobs=logprobs,
                top_logprobs=top_logprobs,
                thinking_budget=thinking_budget,
                xtc_probability=xtc_probability,
                xtc_threshold=xtc_threshold,
                json_schema=json_schema,
                cancel_event=cancel_event,
                logits_processors=logits_processors,
                timeout_seconds=timeout_seconds,
                lora_adapter=lora_adapter,
            )
        # Create a per-request NgramProposer to avoid race conditions
        # when concurrent requests call reset()/propose() on a shared instance.
        from .ngram_proposer import NgramConfig as _NgramConfig
        from .ngram_proposer import NgramProposer as _NgramProposer

        proposer = _NgramProposer(
            _NgramConfig(
                max_n=self._ngram_proposer.config.max_n,
                k=self._ngram_proposer.config.k,
                mode=self._ngram_proposer.config.mode,
            )
        )

        # Handle messages-format prompts (list of dicts) — apply chat template.
        # route through _apply_chat_template + _encode_prompt (NOT raw tokenizer
        # calls) so this spec path matches the default fast path — role remap, family
        # adapters, AND the double-BOS guard. The raw apply_chat_template(tokenize=False)
        # emits a string already opening with the literal bos_token; a following
        # tokenizer.encode (add_special_tokens=True) prepended BOS a SECOND time for
        # Gemma/Llama-3/Mistral → [BOS, BOS, …], corrupting the first-token distribution.
        if isinstance(prompt, list) and prompt and isinstance(prompt[0], dict):
            text = self._apply_chat_template(prompt, enable_thinking=enable_thinking)
            input_ids = self._encode_prompt(tokenizer, text)
        else:
            text = prompt if isinstance(prompt, str) else str(prompt)
            input_ids = tokenizer.encode(text)
        prompt_tokens = len(input_ids)

        # Build stop token sets
        stop_ids: set[int] = set()
        stop_suffixes = []
        if hasattr(tokenizer, "eos_token_id"):
            eid = tokenizer.eos_token_id
            if isinstance(eid, (list, tuple)):
                stop_ids.update(eid)
            elif eid is not None:
                stop_ids.add(eid)
        _eids = getattr(tokenizer, "eos_token_ids", None)
        if _eids is not None:  # may be a bare int (Qwen3.6-27B), not iterable
            stop_ids.update(
                _eids if isinstance(_eids, (list, tuple, set)) else (_eids,)
            )
        if stop_token_ids:
            stop_ids.update(stop_token_ids)
        if stop:
            for s in stop:
                try:
                    ids = tokenizer.encode(s)
                    if len(ids) == 1:
                        stop_ids.add(ids[0])
                    elif len(ids) > 1:
                        stop_suffixes.append(s)
                except Exception:
                    logger.debug(
                        f"failed to encode stop sequence: {s!r}", exc_info=True
                    )

        # route temp>0 off mlx-lm's PRNG-trapped make_sampler (its
        # categorical_sampling @mx.compile cache traps the global PRNG state, so
        # sequential/concurrent temp>0 spec requests collapse + seed is a no-op).
        if temperature is not None and temperature > 1e-6:
            sampler = _build_temp_sampler(
                temperature=temperature,
                top_p=top_p,
                top_k=top_k if top_k > 0 else 0,
                min_p=min_p,
                seed=seed,
                xtc_probability=xtc_probability,
                xtc_threshold=xtc_threshold,
            )
        else:
            sampler = make_sampler(
                temp=temperature,
                top_p=top_p,
                top_k=top_k if top_k > 0 else 0,
                min_p=min_p,
                xtc_probability=xtc_probability,
                xtc_threshold=xtc_threshold,
            )

        # Grammar constraint: pre-validate draft tokens against allowed set
        _grammar_constraint = None
        if json_schema is not None:
            try:
                sampler = _build_constrained_sampler(sampler, json_schema, tokenizer)
                _grammar_constraint = (
                    sampler.constraint if hasattr(sampler, "constraint") else None
                )
            except Exception:
                logger.warning(
                    "Grammar constraint setup failed for n-gram spec", exc_info=True
                )

        def _grammar_filter_drafts(
            draft_ids: list[int],
            generated_ids: list[int],
        ) -> list[int]:
            """Filter draft tokens that violate grammar constraints.

            Returns the longest prefix of draft_ids where every token is
            in the grammar's allowed set at its position. This avoids
            wasting a batched forward pass on drafts that can't be accepted.
            """
            if _grammar_constraint is None:
                return draft_ids
            # rollback() is no-arg ONLY for JsonSchemaConstraint; Regex/
            # Choice/Cfg constraints' rollback(saved) REQUIRES the dict checkpoint()
            # returns, so the bare rollback() here raised TypeError for a {"type":
            # "regex"|"choice"|"cfg"} grammar on the n-gram spec path. Capture the
            # checkpoint and roll back signature-robustly.
            _saved = _grammar_constraint.checkpoint()

            def _gc_rollback():
                try:
                    _grammar_constraint.rollback(_saved)
                except TypeError:
                    try:
                        _grammar_constraint.rollback()
                    except Exception:
                        logger.debug(
                            "grammar rollback failed in n-gram filter", exc_info=True
                        )
                except Exception:
                    logger.debug(
                        "grammar rollback failed in n-gram filter", exc_info=True
                    )

            allowed = _grammar_constraint.get_allowed_tokens(tokenizer, generated_ids)
            if not allowed:
                _gc_rollback()
                return draft_ids
            allowed_set = set(allowed)
            filtered = []
            for tid in draft_ids:
                if tid in allowed_set:
                    filtered.append(tid)
                    try:
                        tok_text = tokenizer.decode([tid])
                        _grammar_constraint.advance(tok_text)
                    except Exception:
                        logger.debug("grammar constraint advance failed", exc_info=True)
                        break
                    allowed = _grammar_constraint.get_allowed_tokens(
                        tokenizer, generated_ids + filtered
                    )
                    if allowed:
                        allowed_set = set(allowed)
                    else:
                        break
                else:
                    break
            _gc_rollback()
            return filtered

        # Inflight prefix sharing: defined before _run so cleanup is accessible
        # in exception handlers. Use timestamp instead of id(_run) since _run
        # is not yet defined at this point.
        _ng_inflight_req_id = f"ng-{int(time.monotonic() * 1e6)}"

        def _unregister_inflight():
            try:
                from .inflight_prefix_sharing import get_inflight_tracker

                get_inflight_tracker().unregister(_ng_inflight_req_id)
            except Exception:
                logger.debug(
                    "inflight prefix unregister failed in n-gram spec", exc_info=True
                )

        def _run():
            if seed is not None:
                mx.random.seed(seed)
            ids = mx.array(input_ids)
            tokens = []
            ttft_s = 0.0
            _stopped_by_stop_id = False
            _stopped_by_suffix = False

            # Prefill with KV prefix cache
            prefix_cache = self._kv_prefix_cache
            # bypass the KV prefix cache when a LoRA adapter is active — it is keyed
            # on token ids + model name but NOT the adapter, so reusing/storing KV here would
            # decode on a different adapter's KV (the keystone, un-propagated to this
            # n-gram spec path; the two live fast paths bypass it at 3571 / 5230). Setting it
            # to None disables BOTH the get and the post-generation add below. Latent today
            # (a LoRA request fails _gemma4_spec_eligible → spec_decode forced off → this path
            # isn't entered with an adapter), but fail-safe if that gate ever changes.
            if lora_adapter is not None:
                prefix_cache = None
            if prefix_cache is not None:
                prefix_cache.evict_under_pressure(self._mem_pressure_threshold)
            # Also evict from paged KV manager when enabled
            if self._kv_manager is not None:
                try:
                    self._kv_manager.memory_pressure_evict(
                        self._mem_pressure_threshold / 100.0
                    )
                except Exception:
                    logger.debug("paged KV pressure eviction failed", exc_info=True)
            try:
                cached_kv, _, matched = (
                    prefix_cache.get(ids)
                    if prefix_cache is not None
                    else (None, None, 0)
                )
            except Exception:
                logger.warning(
                    "KV prefix cache get failed in spec path — falling back to full prefill",
                    exc_info=True,
                )
                cached_kv, _, matched = None, None, 0
            cache = cached_kv if cached_kv is not None else make_prompt_cache(model)
            ids_to_prefill = ids[matched:] if cached_kv is not None else ids
            # generate_step requires at least one prompt token to start
            # decoding. If the prefix cache returned a full match, re-feed
            # the last cached token so we have a starting input.
            if len(ids_to_prefill) == 0 and len(ids) > 0:
                ids_to_prefill = ids[-1:]

            # Inflight prefix sharing: register for concurrent KV block sharing
            try:
                from .inflight_prefix_sharing import get_inflight_tracker

                get_inflight_tracker().register(
                    _ng_inflight_req_id,
                    [int(t) for t in ids],
                    cache,
                    self.model_name or "",
                )
            except Exception:
                logger.debug(
                    "inflight prefix register failed in n-gram spec", exc_info=True
                )

            try:
                gen_t0 = time.perf_counter()
                _timeout_deadline = gen_t0 + timeout_seconds
                _timeout_check_interval = 32
                detokenizer = tokenizer.detokenizer
                detokenizer.reset()
                all_token_ids = list(
                    input_ids
                )  # Track full history for N-gram matching

                with _wired_limit_ctx(model):
                    # Step 1: Prefill
                    first_logits = None
                    for token, logits in generate_step(  # noqa: B007 # token used after loop (line ~5650)
                        ids_to_prefill,
                        model,
                        max_tokens=1,
                        sampler=sampler,
                        prompt_cache=cache,
                    ):
                        first_logits = logits
                        break

                    if first_logits is None:
                        return (
                            tokens,
                            "",
                            [],
                            time.perf_counter() - gen_t0,
                            matched,
                            False,
                            False,
                        )

                    ttft_s = time.perf_counter() - gen_t0

                    # Get first token — use the sampler-applied token from generate_step,
                    # not argmax (which would ignore temperature/top_p/top_k settings).
                    first_token = int(token)
                    tokens.append(first_token)
                    all_token_ids.append(first_token)

                    # align with mlx-lm's speculative_generate_step — the
                    # last produced token stays OUT of the cache (it is the next
                    # token to feed). generate_step already yields first_token
                    # WITHOUT adding its KV, which is exactly the invariant
                    # verify_with_last_token now expects, so we do NOT feed it here.

                    # Step 2: Decode loop with N-gram lookahead
                    remaining = max_tokens - 1
                    while remaining > 0:
                        if _is_cancelled(cancel_event):
                            break
                        # Stop check: break outer loop if stop token was hit in
                        # a previous iteration's accepted/bonus tokens.
                        if _stopped_by_stop_id or _stopped_by_suffix:
                            break
                        # Request-level timeout: check every N tokens to bound
                        # generation time. Without this, a pathological N-gram
                        # proposal/accept cycle can loop indefinitely.
                        if len(tokens) % _timeout_check_interval == 0:
                            if time.perf_counter() > _timeout_deadline:
                                logger.warning(
                                    f"N-gram spec generation timed out after "
                                    f"{timeout_seconds}s ({len(tokens)} tokens)"
                                )
                                break
                        # Propose K draft tokens via N-gram
                        # Use adaptive K if controller is active, else use proposer default
                        _adaptive_k = (
                            self._adaptive_spec.get_draft_length()
                            if self._adaptive_spec
                            else None
                        )
                        draft_ids = proposer.propose(all_token_ids)[
                            : (_adaptive_k or len(all_token_ids))
                        ]
                        # Grammar-aware draft filtering: reject drafts that violate constraints
                        draft_ids = _grammar_filter_drafts(draft_ids, all_token_ids)
                        n_draft = min(len(draft_ids), remaining)

                        if n_draft == 0:
                            # No N-gram proposal — generate one token normally.
                            # tokens[-1] is NOT in the cache (the mlx-lm
                            # invariant), so generate_step feeds it cleanly and
                            # yields the next token (also left out of the cache). No
                            # pre-trim, no extra feed — consistent with the verify
                            # path, which now also keeps the last token out of cache.
                            step_input = mx.array(
                                [tokens[-1]]
                            )  # 1D — generate_step adds the batch dim
                            for token, _logits in generate_step(
                                step_input,
                                model,
                                max_tokens=1,
                                sampler=sampler,
                                prompt_cache=cache,
                            ):
                                token_id = int(token)
                                tokens.append(token_id)
                                all_token_ids.append(token_id)
                                remaining -= 1
                                if token_id in stop_ids:
                                    tokens.pop()
                                    _stopped_by_stop_id = True
                                    break
                                detokenizer.add_token(token_id)
                                if stop_suffixes and any(
                                    detokenizer.text.endswith(s) for s in stop_suffixes
                                ):
                                    tokens.pop()  # Exclude suffix-triggering token from count
                                    _stopped_by_suffix = True
                                    break
                            continue

                        # C10: Batch verify all K draft tokens via SpecDraftVerifier
                        self._ngram_stats["proposals"] += 1
                        self._ngram_stats["total_draft"] += n_draft

                        # Use SpecDraftVerifier for proper batch verification:
                        # 1. One forward pass populates KV cache for all K positions
                        # 2. Consecutive prefix match finds acceptance boundary
                        # 3. KV cache trimmed to remove rejected entries
                        # 4. Bonus token emitted from rejection point
                        result = self._spec_draft_verifier.verify_with_last_token(
                            model=model,
                            last_token_id=tokens[-1],
                            draft_ids=draft_ids[:n_draft],
                            prompt_cache=cache,
                            sampler=sampler,
                        )

                        # Emit accepted tokens
                        _stopped = False
                        for tid in result.accepted_tokens:
                            tokens.append(tid)
                            all_token_ids.append(tid)
                            remaining -= 1
                            if tid in stop_ids:
                                tokens.pop()
                                _stopped = True
                                _stopped_by_stop_id = True
                                break
                            detokenizer.add_token(tid)
                            if stop_suffixes and any(
                                detokenizer.text.endswith(s) for s in stop_suffixes
                            ):
                                tokens.pop()  # Exclude suffix-triggering token from count
                                _stopped = True
                                _stopped_by_suffix = True
                                break
                            # Advance sampler's grammar constraint to stay in sync
                            if _grammar_constraint is not None:
                                try:
                                    tok_text = tokenizer.decode([tid])
                                    _grammar_constraint.advance(tok_text)
                                except Exception:
                                    logger.debug(
                                        "Grammar constraint advance failed for token %d",
                                        tid,
                                        exc_info=True,
                                    )

                        # Emit bonus token (model's own prediction at rejection/last point)
                        if (
                            not _stopped
                            and result.bonus_token is not None
                            and remaining > 0
                        ):
                            bonus = result.bonus_token
                            tokens.append(bonus)
                            all_token_ids.append(bonus)
                            remaining -= 1
                            if bonus in stop_ids:
                                tokens.pop()
                                _stopped_by_stop_id = True
                            else:
                                detokenizer.add_token(bonus)
                                # Suffix check ONLY when bonus is not a stop_id.
                                # Previously this ran unconditionally, so when bonus was
                                # a stop_id the detokenizer text was stale and a coincidental
                                # suffix match from prior tokens would cause a spurious
                                # tokens.pop() (double-pop) and incorrect _stopped_by_suffix.
                                if stop_suffixes and any(
                                    detokenizer.text.endswith(s) for s in stop_suffixes
                                ):
                                    tokens.pop()  # Exclude suffix-triggering token from count
                                    _stopped_by_suffix = True
                                    _stopped = True
                            # Advance sampler's grammar constraint for bonus token
                            if _grammar_constraint is not None:
                                try:
                                    tok_text = tokenizer.decode([bonus])
                                    _grammar_constraint.advance(tok_text)
                                except Exception:
                                    logger.debug(
                                        "grammar advance failed for n-gram bonus token",
                                        exc_info=True,
                                    )
                            # do NOT feed the bonus token into the cache.
                            # Aligning with mlx-lm, the bonus is the next token to
                            # feed — it stays OUT of the cache and the next
                            # verify_with_last_token feeds [bonus, drafts] directly
                            # (no pre-trim). The previous "feed the bonus" + verify
                            # "trim 1" pair left the bonus's stale K/V in the buffer
                            # at the rolled-back position, which the next forward
                            # attended to → repeated/garbled output (".txt.txt…").

                        self._ngram_stats["accepted"] += result.accepted_count

                        # Feed back to adaptive spec controller
                        if self._adaptive_spec is not None:
                            self._adaptive_spec.record_step(
                                n_draft, result.accepted_count
                            )

                # Cache KV state
                if self._kv_quant_bits is not None:
                    _maybe_quantize_kv_cache(
                        cache,
                        self._kv_quant_start,
                        self._kv_quant_group_size,
                        self._kv_quant_bits,
                    )
                if prefix_cache is not None:
                    prefix_cache.add(mx.array(input_ids), cache)

                # Finalize detokenizer to flush partial UTF-8 bytes before
                # assembling output. Without this, multi-byte characters
                # at token boundaries can be truncated, causing incorrect
                # suffix detection via detokenizer.text.endswith() above.
                try:
                    detokenizer.finalize()
                except Exception:
                    logger.debug(
                        "detokenizer finalize failed in n-gram spec", exc_info=True
                    )

                # Use detokenizer text when suffix matching is active (same
                # pattern as _generate_fast) because tokenizer.decode(tokens)
                # may contain a partial suffix that leaked across boundaries.
                if _stopped_by_suffix and stop_suffixes:
                    output_text = detokenizer.text
                    for s in stop_suffixes:
                        if output_text.endswith(s):
                            output_text = output_text[: -len(s)]
                            break
                    output_text = _clean_special_tokens(output_text)
                else:
                    output_text = tokenizer.decode(tokens, skip_special_tokens=True)
                mx.synchronize()
                return (
                    tokens,
                    output_text,
                    [],
                    ttft_s,
                    matched,
                    _stopped_by_suffix,
                    _stopped_by_stop_id,
                )
            finally:
                _unregister_inflight()

        executor = get_mlx_executor()
        loop = asyncio.get_running_loop()
        # (self-audit fix B): keep _lora_applied permanently False so the
        # legacy event-loop release blocks below are inert no-ops; the adapter
        # lifecycle now runs INSIDE _run_with_lora on the executor thread, serialized
        # with generate_step (max_workers=1) — the same keystone as _generate_fast.
        # Acquiring/releasing on the event loop mutated the shared model while a
        # different request's generation read it (cross-thread race).
        _lora_applied = False

        def _run_with_lora():
            _applied = False
            if lora_adapter and getattr(self, "_lora_manager", None) is not None:
                try:
                    _applied = self._lora_manager.acquire_adapter(lora_adapter)
                except Exception as _le:  # fail loud, don't serve base
                    raise RuntimeError(
                        f"LoRA adapter '{lora_adapter}' could not be applied"
                    ) from _le
                if not _applied:
                    raise RuntimeError(
                        f"LoRA adapter '{lora_adapter}' could not be applied"
                    )
            try:
                return _run()
            finally:
                if _applied and getattr(self, "_lora_manager", None) is not None:
                    try:
                        self._lora_manager.release_adapter(lora_adapter)
                    except Exception:
                        logger.debug(
                            "LoRA release failed (n-gram spec executor)", exc_info=True
                        )

        try:
            (
                tokens,
                output_text,
                _,
                ttft_s,
                cached_tokens,
                _stopped_by_suffix,
                _stopped_by_stop_id,
            ) = await loop.run_in_executor(executor, _run_with_lora)
        except MemoryError:
            logger.warning(
                "OOM during N-gram spec generation — returning memory_limit finish reason"
            )
            try:
                import mlx.core as _mx

                await loop.run_in_executor(
                    executor, lambda: (_mx.synchronize(), _mx.clear_cache())
                )
            except Exception:
                logger.debug(
                    "GPU cache cleanup failed after n-gram spec OOM", exc_info=True
                )
            if (
                _lora_applied
                and hasattr(self, "_lora_manager")
                and self._lora_manager is not None
            ):
                try:
                    self._lora_manager.release_adapter(lora_adapter)
                except Exception:
                    logger.warning(
                        "LoRA release failed after n-gram spec OOM", exc_info=True
                    )
            return GenerationOutput(
                finished=True,
                finish_reason="memory_limit",
                prompt_tokens=prompt_tokens,
                completion_tokens=0,
                error="OOM during N-gram spec generation",
                ttft_ms=0.0,
                cached_tokens=0,
            )
        except RuntimeError as e:
            if "memory" in str(e).lower() or "out of" in str(e).lower():
                logger.warning(f"MLX OOM during N-gram spec generation: {e}")
                try:
                    import mlx.core as _mx

                    await loop.run_in_executor(
                        executor, lambda: (_mx.synchronize(), _mx.clear_cache())
                    )
                except Exception:
                    logger.debug(
                        "GPU cache cleanup failed after n-gram spec OOM (RuntimeError)",
                        exc_info=True,
                    )
                if (
                    _lora_applied
                    and hasattr(self, "_lora_manager")
                    and self._lora_manager is not None
                ):
                    try:
                        self._lora_manager.release_adapter(lora_adapter)
                    except Exception:
                        logger.warning(
                            "LoRA release failed after n-gram spec OOM (RuntimeError)",
                            exc_info=True,
                        )
                return GenerationOutput(
                    finished=True,
                    finish_reason="memory_limit",
                    prompt_tokens=prompt_tokens,
                    completion_tokens=0,
                    error=str(e),
                    ttft_ms=0.0,
                    cached_tokens=0,
                )
            if (
                _lora_applied
                and hasattr(self, "_lora_manager")
                and self._lora_manager is not None
            ):
                try:
                    self._lora_manager.release_adapter(lora_adapter)
                except Exception:
                    logger.warning(
                        "LoRA release failed after n-gram spec RuntimeError",
                        exc_info=True,
                    )
            # Return error output for non-OOM RuntimeError instead of
            # propagating to caller (which expects GenerationOutput).
            return GenerationOutput(
                finished=True,
                finish_reason="error",
                prompt_tokens=prompt_tokens,
                completion_tokens=0,
                error=f"RuntimeError during N-gram spec generation: {e}",
                ttft_ms=0.0,
                cached_tokens=0,
            )
        except Exception as e:
            logger.error(
                f"Unexpected error during N-gram spec generation: {e}", exc_info=True
            )
            if (
                _lora_applied
                and hasattr(self, "_lora_manager")
                and self._lora_manager is not None
            ):
                try:
                    self._lora_manager.release_adapter(lora_adapter)
                except Exception:
                    logger.warning(
                        "LoRA release failed after n-gram spec unexpected error",
                        exc_info=True,
                    )
            # Return error output instead of propagating exception to caller.
            return GenerationOutput(
                finished=True,
                finish_reason="error",
                prompt_tokens=prompt_tokens,
                completion_tokens=0,
                error=f"N-gram spec generation failed: {e}",
                ttft_ms=0.0,
                cached_tokens=0,
            )

        # Determine finish_reason with cancel awareness.
        # Stop tokens are popped from `tokens`, so check the flags instead.
        _cancelled = _is_cancelled(cancel_event)
        if _cancelled or _stopped_by_suffix or _stopped_by_stop_id:
            finish_reason = "stop"
        else:
            finish_reason = "length"
        output_text = _clean_special_tokens(output_text)

        # Trim stop suffix from output text when matched during generation
        if _stopped_by_suffix and stop_suffixes:
            for s in stop_suffixes:
                if output_text.endswith(s):
                    output_text = output_text[: -len(s)]
                    break

        # Build logprobs from generated tokens — n-gram spec decode does not
        # expose per-token logits from the verify step, so we cannot compute
        # real logprobs. Return None instead of fake 0.0 to avoid misleading.
        _ngram_logprobs = None
        if logprobs and tokens:
            _ngram_logprobs = None  # Real logprobs unavailable from n-gram spec path

        # Record TTFT in Prometheus for n-gram spec path
        if ttft_s > 0:
            try:
                from yunshu_gateway.middleware.prometheus_exporter import (
                    get_prometheus_metrics,
                )

                pm = get_prometheus_metrics()
                pm.observe_histogram(
                    "ttft_seconds", ttft_s, labels={"model_id": self.model_label}
                )
            except Exception:
                logger.debug(
                    "TTFT prometheus recording failed in n-gram spec path",
                    exc_info=True,
                )

        # Reasoning parser: extract thinking tokens from n-gram spec output.
        # N-gram spec decode does not track thinking tokens internally,
        # so we parse the output text for reasoning content.
        _ng_reasoning_tok = 0
        if output_text:
            try:
                from .reasoning_parser import get_reasoning_parser

                rp = get_reasoning_parser(self.model_name)
                rp_out = rp.parse(output_text)
                if rp_out.reasoning and rp_out.reasoning_tokens > 0:
                    _ng_reasoning_tok = rp_out.reasoning_tokens
                    if rp_out.content != output_text:
                        output_text = rp_out.content
            except Exception:
                logger.debug(
                    "reasoning_parser failed in n-gram spec path", exc_info=True
                )

        if (
            _lora_applied
            and hasattr(self, "_lora_manager")
            and self._lora_manager is not None
        ):
            try:
                self._lora_manager.release_adapter(lora_adapter)
            except Exception:
                logger.warning(
                    "LoRA release failed after n-gram spec normal completion",
                    exc_info=True,
                )
        return GenerationOutput(
            text=output_text,
            new_text=output_text,
            prompt_tokens=prompt_tokens,
            completion_tokens=len(tokens),
            finished=True,
            finish_reason=finish_reason,
            cached_tokens=cached_tokens,
            ttft_ms=round(ttft_s * 1000, 1),
            reasoning_tokens=_ng_reasoning_tok,
            logprobs=_ngram_logprobs,
        )

    async def _stream_generate_ngram_spec(
        self,
        prompt: str | list[dict],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 1.0,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        logit_bias: dict[int, float] | None = None,
        stop: list[str] | None = None,
        stop_token_ids: list[int] | None = None,
        seed: int | None = None,
        json_schema: dict | str | None = None,
        logprobs: bool = False,
        top_logprobs: int | None = None,
        xtc_probability: float = 0.0,
        xtc_threshold: float = 0.0,
        cancel_event: asyncio.Event | None = None,
        timeout_seconds: float = 300.0,
        logits_processors: list | None = None,
        enable_thinking: bool | None = None,
        thinking_budget: int | None = None,
        lora_adapter: str | None = None,
    ) -> AsyncIterator[GenerationOutput]:
        """Stream generate using N-gram speculative decoding (queue-based).

        UNREACHABLE in the streaming flow (honesty annotation): stream_generate()
        forces ``spec_decode = False`` (~4381) so the n-gram branch (~4537) routes to
        _stream_generate_fast instead of here (this impl
        early-terminates and is bandwidth-bound = no speedup). The documented spec-stream
        multi-token stop-prefix leak (the per-segment `suffix_hit` blanking below holds
        back only the CURRENT segment, not a prefix split across earlier tokens) is thus a
        DEAD-code defect, not a live bug — the live path _stream_generate_fast uses
        StopHoldbackBuffer. If re-enabled, route the consumer's new_text through
        StopHoldbackBuffer (the _hb pattern) before yielding.
        """
        import mlx.core as mx
        from mlx_lm.generate import generate_step
        from mlx_lm.models.cache import make_prompt_cache
        from mlx_lm.sample_utils import make_sampler

        from .mlx_executor import get_mlx_executor

        tokenizer = self._tokenizer
        model = self._model

        # SAFETY GUARD (streaming): see _generate_ngram_spec. Spec verify
        # needs a trimmable cache; Qwen3.5's hybrid attention uses non-trimmable
        # ArraysCache (recurrent state), so delegate the whole stream to the plain
        # streaming fast path rather than corrupting into ".txt.txt..." repetition.
        try:
            from mlx_lm.models.cache import (
                can_trim_prompt_cache as _can_trim,
            )
            from mlx_lm.models.cache import (
                make_prompt_cache as _mk_cache,
            )

            _spec_cache_supported = _can_trim(_mk_cache(model))
        except Exception:
            _spec_cache_supported = False
        if not _spec_cache_supported:
            logger.info(
                "N-gram spec disabled (streaming): cache not trimmable — "
                "using the plain streaming fast path."
            )
            async for _chunk in self._stream_generate_fast(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                repetition_penalty=repetition_penalty,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                logit_bias=logit_bias,
                stop=stop,
                stop_token_ids=stop_token_ids,
                seed=seed,
                enable_thinking=enable_thinking,
                thinking_budget=thinking_budget,
                xtc_probability=xtc_probability,
                xtc_threshold=xtc_threshold,
                json_schema=json_schema,
                cancel_event=cancel_event,
                logprobs=logprobs,
                top_logprobs=top_logprobs,
                logits_processors=logits_processors,
            ):
                yield _chunk
            return
        # Create a per-request NgramProposer to avoid race conditions
        # when concurrent requests call reset()/propose() on a shared instance.
        from .ngram_proposer import NgramConfig as _NgramConfig
        from .ngram_proposer import NgramProposer as _NgramProposer

        proposer = _NgramProposer(
            _NgramConfig(
                max_n=self._ngram_proposer.config.max_n,
                k=self._ngram_proposer.config.k,
                mode=self._ngram_proposer.config.mode,
            )
        )

        # Handle messages-format prompts (list of dicts) — apply chat template.
        # route through _apply_chat_template + _encode_prompt (NOT raw tokenizer
        # calls) so this spec path matches the default fast path — role remap, family
        # adapters, AND the double-BOS guard. The raw apply_chat_template(tokenize=False)
        # emits a string already opening with the literal bos_token; a following
        # tokenizer.encode (add_special_tokens=True) prepended BOS a SECOND time for
        # Gemma/Llama-3/Mistral → [BOS, BOS, …], corrupting the first-token distribution.
        if isinstance(prompt, list) and prompt and isinstance(prompt[0], dict):
            text = self._apply_chat_template(prompt, enable_thinking=enable_thinking)
            input_ids = self._encode_prompt(tokenizer, text)
        else:
            text = prompt if isinstance(prompt, str) else str(prompt)
            input_ids = tokenizer.encode(text)
        prompt_tokens = len(input_ids)

        stop_ids: set[int] = set()
        stop_suffixes = []
        if hasattr(tokenizer, "eos_token_id"):
            eid = tokenizer.eos_token_id
            if isinstance(eid, (list, tuple)):
                stop_ids.update(eid)
            elif eid is not None:
                stop_ids.add(eid)
        _eids = getattr(tokenizer, "eos_token_ids", None)
        if _eids is not None:  # may be a bare int (Qwen3.6-27B), not iterable
            stop_ids.update(
                _eids if isinstance(_eids, (list, tuple, set)) else (_eids,)
            )
        if stop_token_ids:
            stop_ids.update(stop_token_ids)
        if stop:
            for s in stop:
                try:
                    ids = tokenizer.encode(s)
                    if len(ids) == 1:
                        stop_ids.add(ids[0])
                    elif len(ids) > 1:
                        stop_suffixes.append(s)
                except Exception:
                    logger.debug(
                        f"failed to encode stop sequence: {s!r}", exc_info=True
                    )

        # route temp>0 off mlx-lm's PRNG-trapped make_sampler (see the
        # non-streaming sibling). Greedy (temp==0) stays on argmax make_sampler.
        if temperature is not None and temperature > 1e-6:
            sampler = _build_temp_sampler(
                temperature=temperature,
                top_p=top_p,
                top_k=top_k if top_k > 0 else 0,
                min_p=min_p,
                seed=seed,
                xtc_probability=xtc_probability,
                xtc_threshold=xtc_threshold,
            )
        else:
            sampler = make_sampler(
                temp=temperature,
                top_p=top_p,
                top_k=top_k if top_k > 0 else 0,
                min_p=min_p,
                xtc_probability=xtc_probability,
                xtc_threshold=xtc_threshold,
            )

        # Grammar constraint: pre-validate draft tokens against allowed set
        _stream_grammar_constraint = None
        if json_schema is not None:
            try:
                sampler = _build_constrained_sampler(sampler, json_schema, tokenizer)
                _stream_grammar_constraint = (
                    sampler.constraint if hasattr(sampler, "constraint") else None
                )
            except Exception:
                logger.warning(
                    "Grammar constraint setup failed for streaming n-gram spec",
                    exc_info=True,
                )

        # LoRA adapter: the acquire+apply / release+restore lifecycle MUST run on the
        # executor thread (inside _run), NOT here on the event loop — acquiring here
        # mutates the shared model off-thread while another request's generate_step
        # reads it (cross-thread wrong-adapter bleed). Kept False so the legacy
        # event-loop release below is an inert no-op; _run owns the lifecycle. Mirrors
        # the _generate_fast / n-gram-non-streaming keystone .
        _ng_s_lora_applied = False

        def _stream_grammar_filter_drafts(
            draft_ids: list[int],
            generated_ids: list[int],
        ) -> list[int]:
            """Filter draft tokens that violate grammar constraints (streaming path)."""
            if _stream_grammar_constraint is None:
                return draft_ids
            _stream_grammar_constraint.checkpoint()
            allowed = _stream_grammar_constraint.get_allowed_tokens(
                tokenizer, generated_ids
            )
            if not allowed:
                _stream_grammar_constraint.rollback()
                return draft_ids
            allowed_set = set(allowed)
            filtered = []
            for tid in draft_ids:
                if tid in allowed_set:
                    filtered.append(tid)
                    try:
                        tok_text = tokenizer.decode([tid])
                        _stream_grammar_constraint.advance(tok_text)
                    except Exception:
                        logger.debug(
                            "grammar constraint advance failed in streaming filter",
                            exc_info=True,
                        )
                        break
                    allowed = _stream_grammar_constraint.get_allowed_tokens(
                        tokenizer, generated_ids + filtered
                    )
                    if allowed:
                        allowed_set = set(allowed)
                    else:
                        break
                else:
                    break
            _stream_grammar_constraint.rollback()
            return filtered

        _sentinel = object()
        _q: asyncio.Queue = asyncio.Queue(maxsize=512)
        loop = asyncio.get_running_loop()
        from .streaming_optimizer import StreamingBackpressureController

        _backpressure = StreamingBackpressureController(max_queue_size=100)

        def _put(item):
            # Backpressure-aware queue with retry (same logic as main streaming path)
            if _q.qsize() > 400:  # 78% of 512
                time.sleep(0.01)
            # Retry up to 3 times if the queue is full, sleeping 1ms between
            # attempts. Thread-safe: skip get_nowait() — see main streaming
            # _put for rationale (executor thread must not mutate asyncio Queue).
            for _attempt in range(4):  # 1 initial + 3 retries
                if not _q.full():
                    loop.call_soon_threadsafe(_q.put_nowait, item)
                    return
                if _attempt < 3:
                    time.sleep(0.001)
            logger.warning(
                "N-gram spec streaming queue overflow after 3 retries — sending error sentinel. "
                "Client will see finish_reason=error."
            )
            try:
                loop.call_soon_threadsafe(
                    _q.put_nowait,
                    Exception("N-gram streaming queue overflow — output truncated"),
                )
            except Exception:
                logger.debug(
                    "Failed to put error sentinel into n-gram streaming queue",
                    exc_info=True,
                )

        # Inflight prefix sharing for streaming n-gram spec
        _ng_s_inflight_req_id = f"ng-s-{int(time.monotonic() * 1e6)}"

        def _unregister_inflight():
            try:
                from .inflight_prefix_sharing import get_inflight_tracker

                get_inflight_tracker().unregister(_ng_s_inflight_req_id)
            except Exception:
                logger.debug(
                    "inflight prefix unregister failed in streaming n-gram spec",
                    exc_info=True,
                )

        _ng_timeout_cancel = threading.Event()
        _ng_gen_t0 = time.perf_counter()

        def _run():
            # Acquire+apply the LoRA adapter HERE, on the executor thread, serialized
            # with generate_step (max_workers=1) — never on the event loop (which would
            # mutate the shared model while another request reads it).
            _applied = False
            if lora_adapter and getattr(self, "_lora_manager", None) is not None:
                try:
                    _applied = self._lora_manager.acquire_adapter(lora_adapter)
                except Exception as _le:  # fail loud, don't serve base
                    raise RuntimeError(
                        f"LoRA adapter '{lora_adapter}' could not be applied"
                    ) from _le
                if not _applied:
                    raise RuntimeError(
                        f"LoRA adapter '{lora_adapter}' could not be applied"
                    )
            try:
                _run_inner()
            except Exception as e:
                logger.error(f"N-gram streaming generation failed: {e}", exc_info=True)
                try:
                    import mlx.core as _cleanup_mx

                    _cleanup_mx.synchronize()
                    _cleanup_mx.clear_cache()
                except Exception:
                    logger.debug(
                        "GPU cache cleanup failed in n-gram streaming error handler",
                        exc_info=True,
                    )
                _put(e)
            finally:
                if _applied and getattr(self, "_lora_manager", None) is not None:
                    try:
                        self._lora_manager.release_adapter(lora_adapter)
                    except Exception:
                        logger.debug(
                            "LoRA release failed (streaming n-gram spec executor)",
                            exc_info=True,
                        )
                _unregister_inflight()
                _put(_sentinel)

        def _run_inner():
            if seed is not None:
                mx.random.seed(seed)
            ids = mx.array(input_ids)
            tokens = []
            all_token_ids = list(input_ids)

            prefix_cache = self._kv_prefix_cache
            if prefix_cache is not None:
                prefix_cache.evict_under_pressure(self._mem_pressure_threshold)
            # Also evict from paged KV manager when enabled
            if self._kv_manager is not None:
                try:
                    self._kv_manager.memory_pressure_evict(
                        self._mem_pressure_threshold / 100.0
                    )
                except Exception:
                    logger.debug("paged KV pressure eviction failed", exc_info=True)
            try:
                cached_kv, _, matched = (
                    prefix_cache.get(ids)
                    if prefix_cache is not None
                    else (None, None, 0)
                )
            except Exception:
                logger.warning(
                    "KV prefix cache get failed in spec path — falling back to full prefill",
                    exc_info=True,
                )
                cached_kv, _, matched = None, None, 0
            cache = cached_kv if cached_kv is not None else make_prompt_cache(model)
            ids_to_prefill = ids[matched:] if cached_kv is not None else ids
            if len(ids_to_prefill) == 0 and len(ids) > 0:
                ids_to_prefill = ids[-1:]

            # Inflight prefix sharing: register for concurrent KV block sharing
            try:
                from .inflight_prefix_sharing import get_inflight_tracker

                get_inflight_tracker().register(
                    _ng_s_inflight_req_id,
                    [int(t) for t in ids],
                    cache,
                    self.model_name or "",
                )
            except Exception:
                logger.debug(
                    "inflight prefix register failed in streaming n-gram spec",
                    exc_info=True,
                )

            detokenizer = tokenizer.detokenizer
            detokenizer.reset()
            n_tok = 0

            # Thinking budget enforcement — detect <think/</think via single-token IDs
            _ng_think_start_token = None
            _ng_think_end_token = None
            if thinking_budget is not None or enable_thinking:
                # bracketed-form helper (bare "</think" tokenized to 2 → guard failed).
                _ng_think_start_token, _ng_think_end_token = _resolve_think_token_ids(
                    tokenizer
                )
            _ng_in_thinking = False
            _ng_thinking_tokens_used = 0

            def _ng_track_thinking(tid):
                """Track thinking state and enforce budget. Returns True if budget exceeded."""
                nonlocal _ng_in_thinking, _ng_thinking_tokens_used
                if _ng_think_start_token is None:
                    return False
                if not _ng_in_thinking and tid == _ng_think_start_token:
                    _ng_in_thinking = True
                elif _ng_in_thinking:
                    _ng_thinking_tokens_used += 1
                    if tid == _ng_think_end_token:
                        _ng_in_thinking = False
                        return False
                    if (
                        thinking_budget is not None
                        and _ng_thinking_tokens_used >= thinking_budget
                        and _ng_think_end_token is not None
                    ):
                        _ng_in_thinking = False
                        # Force-insert </think token
                        tokens.append(_ng_think_end_token)
                        all_token_ids.append(_ng_think_end_token)
                        n_tok_cur = n_tok + 1
                        detokenizer.add_token(_ng_think_end_token)
                        _end_text = detokenizer.last_segment
                        if _end_text:
                            _put((_end_text, n_tok_cur, None, _ng_think_end_token))
                        # Stop token to end generation
                        _put(("", n_tok_cur, "stop", _ng_think_end_token))
                        return True
                return False

            with _wired_limit_ctx(model):
                # Prefill + first token
                for token, _logits in generate_step(
                    ids_to_prefill,
                    model,
                    max_tokens=1,
                    sampler=sampler,
                    prompt_cache=cache,
                ):
                    first_token = int(token)
                    tokens.append(first_token)
                    all_token_ids.append(first_token)
                    n_tok += 1
                    if first_token in stop_ids:
                        # First token is stop — don't add to detokenizer, signal stop
                        n_tok -= 1  # Exclude stop token from completion count
                        _put(("", n_tok, "stop", first_token))
                        if prefix_cache is not None:
                            prefix_cache.add(ids, cache)
                        mx.synchronize()
                        _unregister_inflight()
                        return
                    detokenizer.add_token(first_token)
                    if _ng_track_thinking(first_token):
                        if prefix_cache is not None:
                            prefix_cache.add(ids, cache)
                        mx.synchronize()
                        _unregister_inflight()
                        return
                    _put((detokenizer.last_segment, n_tok, None, first_token))
                remaining = max_tokens - 1
                while remaining > 0:
                    if _is_cancelled(cancel_event):
                        detokenizer.finalize()
                        _remaining = detokenizer.last_segment
                        if _remaining:
                            _put((_remaining, n_tok, None, 0))
                        _put(("", n_tok, "stop", 0))
                        return
                    if _ng_timeout_cancel.is_set():
                        detokenizer.finalize()
                        _remaining = detokenizer.last_segment
                        if _remaining:
                            _put((_remaining, n_tok, None, 0))
                        _put(("", n_tok, "timeout", 0))
                        return
                    _adaptive_k = (
                        self._adaptive_spec.get_draft_length()
                        if self._adaptive_spec
                        else None
                    )
                    draft_ids = proposer.propose(all_token_ids)
                    if _adaptive_k is not None:
                        draft_ids = draft_ids[:_adaptive_k]
                    # Grammar-aware draft filtering: reject drafts that violate constraints
                    draft_ids = _stream_grammar_filter_drafts(draft_ids, all_token_ids)
                    n_draft = min(len(draft_ids), remaining)

                    if n_draft == 0:
                        step_input = mx.array(
                            [tokens[-1]]
                        )  # 1D — generate_step adds the batch dim
                        for token, _logits in generate_step(
                            step_input,
                            model,
                            max_tokens=1,
                            sampler=sampler,
                            prompt_cache=cache,
                        ):
                            token_id = int(token)
                            tokens.append(token_id)
                            all_token_ids.append(token_id)
                            remaining -= 1
                            n_tok += 1
                            stop_hit = token_id in stop_ids
                            if stop_hit:
                                n_tok -= 1  # Exclude stop token from count
                                # Stop token — don't add to detokenizer
                                _put(("", n_tok, "stop", token_id))
                                if prefix_cache is not None:
                                    prefix_cache.add(ids, cache)
                                mx.synchronize()
                                _unregister_inflight()
                                return
                            detokenizer.add_token(token_id)
                            if _ng_track_thinking(token_id):
                                if prefix_cache is not None:
                                    prefix_cache.add(ids, cache)
                                mx.synchronize()
                                _unregister_inflight()
                                return
                            suffix_hit = False
                            if stop_suffixes:
                                suffix_hit = any(
                                    detokenizer.text.endswith(s) for s in stop_suffixes
                                )
                            _text = "" if suffix_hit else detokenizer.last_segment
                            if suffix_hit:
                                n_tok -= 1  # Exclude suffix-triggering token from count
                            _put(
                                (_text, n_tok, "stop" if suffix_hit else None, token_id)
                            )
                            if suffix_hit:
                                detokenizer.finalize()
                                _remaining = detokenizer.last_segment
                                if stop_suffixes and _remaining:
                                    for s in stop_suffixes:
                                        if _remaining.endswith(s):
                                            _remaining = _remaining[: -len(s)]
                                            break
                                if _remaining:
                                    _put((_remaining, n_tok, None, token_id))
                                if prefix_cache is not None:
                                    prefix_cache.add(ids, cache)
                                mx.synchronize()
                                _unregister_inflight()
                                return
                        continue

                    self._ngram_stats["proposals"] += 1
                    self._ngram_stats["total_draft"] += n_draft
                    accepted = 0
                    stopped = False

                    # C10: Batch verify all K draft tokens in one forward pass
                    draft_arr = mx.array(draft_ids[:n_draft]).reshape(1, -1)
                    batch_logits = model(draft_arr, cache=cache)
                    if hasattr(batch_logits, "logits"):
                        batch_logits = batch_logits.logits

                    # GPU-accelerated rejection sampling when enabled
                    if self._gpu_rejection_enabled:
                        from .gpu_rejection import GPURejectionSampler as _GRS

                        rej_result = self._gpu_rejection_sampler.verify_greedy(
                            batch_logits[0, :n_draft], draft_ids[:n_draft]
                        )
                        accepted = rej_result.accepted_count

                        # NG-PEN: Apply penalty/bias to bonus position logits.
                        # Only the bonus token (first new token after accepted drafts)
                        # gets penalties — draft tokens are already committed.
                        _has_ng_pen = (
                            repetition_penalty != 1.0
                            or frequency_penalty != 0.0
                            or presence_penalty != 0.0
                            or (logit_bias is not None and len(logit_bias) > 0)
                        )
                        if _has_ng_pen and accepted < n_draft:
                            _ng_bonus_logits = batch_logits[0, accepted, :]
                            _ng_token_hist = all_token_ids
                            _ng_bonus_logits = _apply_spec_bonus_penalties(
                                _ng_bonus_logits,
                                _ng_token_hist,
                                len(input_ids),
                                repetition_penalty=repetition_penalty,
                                frequency_penalty=frequency_penalty,
                                presence_penalty=presence_penalty,
                                logit_bias=logit_bias,
                            )
                            batch_logits[0, accepted, :] = _ng_bonus_logits

                        for i in range(n_draft):
                            if i < accepted:
                                accepted_id = draft_ids[i]
                            elif i == accepted:
                                accepted_id = _GRS.compute_bonus_token(
                                    batch_logits[0, :n_draft], accepted
                                )
                            else:
                                break
                            tokens.append(accepted_id)
                            all_token_ids.append(accepted_id)
                            remaining -= 1
                            n_tok += 1
                            stop_hit = accepted_id in stop_ids
                            if stop_hit:
                                n_tok -= 1  # Exclude stop token from count
                                _put(("", n_tok, "stop", accepted_id))
                                stopped = True
                            else:
                                detokenizer.add_token(accepted_id)
                                if _ng_track_thinking(accepted_id):
                                    stopped = True
                                    break
                                suffix_hit = False
                                if stop_suffixes:
                                    suffix_hit = any(
                                        detokenizer.text.endswith(s)
                                        for s in stop_suffixes
                                    )
                                _text = "" if suffix_hit else detokenizer.last_segment
                                if suffix_hit:
                                    n_tok -= (
                                        1  # Exclude suffix-triggering token from count
                                    )
                                _put(
                                    (
                                        _text,
                                        n_tok,
                                        "stop" if suffix_hit else None,
                                        accepted_id,
                                    )
                                )
                                if suffix_hit:
                                    stopped = True
                            # Advance grammar constraint for accepted/bonus token
                            if _stream_grammar_constraint is not None and not stop_hit:
                                try:
                                    tok_text = tokenizer.decode([accepted_id])
                                    _stream_grammar_constraint.advance(tok_text)
                                except Exception:
                                    logger.debug(
                                        "grammar advance failed for n-gram streaming accepted token",
                                        exc_info=True,
                                    )
                            if i >= accepted:
                                stopped = True
                            if stopped:
                                break

                        # Trim KV cache to remove entries for rejected draft tokens.
                        # The batch forward populated the cache with n_draft entries,
                        # but only `accepted` were verified. Trim the rejected ones.
                        # Then feed the bonus/correction token so its KV entry is
                        # present for the next iteration's batch forward.
                        if accepted < n_draft:
                            try:
                                from mlx_lm.models.cache import trim_prompt_cache

                                trim_prompt_cache(cache, n_draft - accepted)
                            except Exception:
                                logger.debug(
                                    "trim_prompt_cache (n-gram GPU partial) failed, falling back",
                                    exc_info=True,
                                )
                                for c in cache:
                                    if hasattr(c, "trim"):
                                        c.trim(n_draft - accepted)
                            # Feed the correction token to populate its KV entry
                            if not stopped and tokens:
                                _correction = tokens[-1]
                                _ = model(mx.array([[_correction]]), cache=cache)
                    else:
                        # CPU sequential fallback
                        # NG-PEN: Apply penalty/bias to the bonus/correction logits
                        # at the first rejection position (the first "new" token).
                        _has_ng_pen_cpu = (
                            repetition_penalty != 1.0
                            or frequency_penalty != 0.0
                            or presence_penalty != 0.0
                            or (logit_bias is not None and len(logit_bias) > 0)
                        )
                        for i in range(n_draft):
                            # Apply penalties at the first rejection position only
                            if _has_ng_pen_cpu and i == accepted:
                                _ng_bonus_logits_cpu = batch_logits[0, i, :]
                                _ng_token_hist_cpu = all_token_ids
                                _ng_bonus_logits_cpu = _apply_spec_bonus_penalties(
                                    _ng_bonus_logits_cpu,
                                    _ng_token_hist_cpu,
                                    len(input_ids),
                                    repetition_penalty=repetition_penalty,
                                    frequency_penalty=frequency_penalty,
                                    presence_penalty=presence_penalty,
                                    logit_bias=logit_bias,
                                )
                                batch_logits[0, i, :] = _ng_bonus_logits_cpu
                            model_pick = int(
                                mx.argmax(batch_logits[0, i], axis=-1).item()
                            )
                            draft_id = draft_ids[i]
                            is_accept = model_pick == draft_id
                            accepted_id = draft_id if is_accept else model_pick
                            tokens.append(accepted_id)
                            all_token_ids.append(accepted_id)
                            if is_accept:
                                accepted += 1
                            remaining -= 1
                            n_tok += 1
                            stop_hit = accepted_id in stop_ids
                            if stop_hit:
                                n_tok -= 1  # Exclude stop token from count
                                _put(("", n_tok, "stop", accepted_id))
                                stopped = True
                            else:
                                detokenizer.add_token(accepted_id)
                                if _ng_track_thinking(accepted_id):
                                    stopped = True
                                    break
                                suffix_hit = False
                                if stop_suffixes:
                                    suffix_hit = any(
                                        detokenizer.text.endswith(s)
                                        for s in stop_suffixes
                                    )
                                _text = "" if suffix_hit else detokenizer.last_segment
                                if suffix_hit:
                                    n_tok -= (
                                        1  # Exclude suffix-triggering token from count
                                    )
                                _put(
                                    (
                                        _text,
                                        n_tok,
                                        "stop" if suffix_hit else None,
                                        accepted_id,
                                    )
                                )
                                if suffix_hit:
                                    stopped = True
                            # Advance grammar constraint for accepted/bonus token
                            if _stream_grammar_constraint is not None and not stop_hit:
                                try:
                                    tok_text = tokenizer.decode([accepted_id])
                                    _stream_grammar_constraint.advance(tok_text)
                                except Exception:
                                    logger.debug(
                                        "grammar advance failed for n-gram streaming CPU token",
                                        exc_info=True,
                                    )
                            if not is_accept:
                                stopped = True
                            if stopped:
                                break

                        # CPU fallback: trim KV cache for rejected tokens
                        if accepted < n_draft:
                            try:
                                from mlx_lm.models.cache import trim_prompt_cache

                                trim_prompt_cache(cache, n_draft - accepted)
                            except Exception:
                                logger.debug(
                                    "trim_prompt_cache (n-gram CPU fallback) failed, falling back",
                                    exc_info=True,
                                )
                                for c in cache:
                                    if hasattr(c, "trim"):
                                        c.trim(n_draft - accepted)
                            # Feed the correction token to populate its KV entry
                            if not stopped and tokens:
                                _correction = tokens[-1]
                                _ = model(mx.array([[_correction]]), cache=cache)

                    self._ngram_stats["accepted"] += accepted
                    if self._adaptive_spec is not None:
                        self._adaptive_spec.record_step(n_draft, accepted)
                    if stopped:
                        detokenizer.finalize()
                        _remaining = detokenizer.last_segment
                        # Trim stop suffix from remaining text — the suffix may
                        # span multiple tokens, so detokenizer.text still
                        # contains it even after finalize().
                        if stop_suffixes and _remaining:
                            for s in stop_suffixes:
                                if _remaining.endswith(s):
                                    _remaining = _remaining[: -len(s)]
                                    break
                        if _remaining:
                            _put((_remaining, n_tok, None, 0))
                        if prefix_cache is not None:
                            prefix_cache.add(ids, cache)
                        mx.synchronize()
                        return

            if prefix_cache is not None:
                prefix_cache.add(ids, cache)
            detokenizer.finalize()
            remaining = detokenizer.last_segment
            if remaining:
                _put((remaining, n_tok, None, 0))
            _put(("", n_tok, "length", 0))
            mx.synchronize()

        executor = get_mlx_executor()
        future = loop.run_in_executor(executor, _run)

        accumulated = ""
        n_tok = 0
        _ng_ttft_recorded = False
        _ng_ttft_ms_val = 0.0
        _ng_gen_t0 = time.perf_counter()
        # Consumer-side thinking state mirrors GPU-side tracking for reporting
        _ng_consumer_think_start = None
        _ng_consumer_think_end = None
        _ng_consumer_in_thinking = False
        _ng_consumer_thinking_tokens = 0
        if thinking_budget is not None or enable_thinking:
            # bracketed-form helper (bare "</think" tokenized to 2 → guard failed).
            _ng_consumer_think_start, _ng_consumer_think_end = _resolve_think_token_ids(
                tokenizer
            )
        _ng_fp_lock = getattr(self, "_fast_path_lock", None)
        if _ng_fp_lock is not None:
            with _ng_fp_lock:
                self._active_fast_path_count += 1
        try:
            while True:
                # Check cancel_event from consumer side (mirrors MTP streaming path)
                if _is_cancelled(cancel_event):
                    yield GenerationOutput(
                        text=_clean_special_tokens(accumulated) if accumulated else "",
                        new_text="",
                        prompt_tokens=prompt_tokens,
                        completion_tokens=n_tok,
                        finished=True,
                        finish_reason="stop",
                        ttft_ms=_ng_ttft_ms_val,
                        cached_tokens=0,
                        reasoning_tokens=0,
                    )
                    break
                try:
                    item = await asyncio.wait_for(_q.get(), timeout=timeout_seconds)
                except TimeoutError:
                    logger.warning(
                        f"N-gram streaming timeout: no token for {timeout_seconds}s"
                    )
                    _ng_timeout_cancel.set()  # Signal GPU loop to stop
                    # Yield terminal output so consumer sees finished=True
                    yield GenerationOutput(
                        text=_clean_special_tokens(accumulated) if accumulated else "",
                        new_text="",
                        prompt_tokens=prompt_tokens,
                        completion_tokens=n_tok,
                        finished=True,
                        finish_reason="error",
                        error=f"N-gram streaming timeout: no token for {timeout_seconds}s",
                        ttft_ms=_ng_ttft_ms_val,
                        cached_tokens=0,
                        reasoning_tokens=0,
                    )
                    break
                if item is _sentinel:
                    break
                if isinstance(item, BaseException):
                    logger.warning(f"N-gram streaming error: {item}")
                    # Yield terminal error output so consumer sees finished=True
                    yield GenerationOutput(
                        text=_clean_special_tokens(accumulated) if accumulated else "",
                        new_text="",
                        prompt_tokens=prompt_tokens,
                        completion_tokens=n_tok,
                        finished=True,
                        finish_reason="error",
                        error=str(item),
                        ttft_ms=_ng_ttft_ms_val,
                        cached_tokens=0,
                        reasoning_tokens=0,
                    )
                    break
                new_text, tok_count, _fr_val, token_id = item
                # Backward compat: _fr_val may be bool or str or None
                if isinstance(_fr_val, bool):
                    finish_reason = "stop" if _fr_val else None
                    done = _fr_val
                else:
                    finish_reason = _fr_val
                    done = _fr_val is not None
                accumulated += new_text
                if len(accumulated) > _MAX_STREAMING_TEXT_BUFFER:
                    logger.error(
                        "Streaming text buffer exceeded 1MB limit (%d bytes) — truncating",
                        len(accumulated),
                    )
                    yield GenerationOutput(
                        text=_clean_special_tokens(accumulated),
                        new_text="",
                        prompt_tokens=prompt_tokens,
                        completion_tokens=tok_count,
                        finished=True,
                        finish_reason="length",
                        error="Streaming text buffer exceeded 1MB limit",
                        cached_tokens=0,
                        reasoning_tokens=_ng_consumer_thinking_tokens
                        if _ng_consumer_think_start is not None
                        else 0,
                        ttft_ms=_ng_ttft_ms_val,
                    )
                    break
                n_tok = tok_count
                # Consumer-side thinking state tracking for reasoning_tokens reporting
                if _ng_consumer_think_start is not None and isinstance(token_id, int):
                    if (
                        not _ng_consumer_in_thinking
                        and token_id == _ng_consumer_think_start
                    ):
                        _ng_consumer_in_thinking = True
                    elif _ng_consumer_in_thinking:
                        _ng_consumer_thinking_tokens += 1
                        if token_id == _ng_consumer_think_end:
                            _ng_consumer_in_thinking = False

                # Streaming backpressure: slow down if client can't keep up
                if _backpressure.check_backpressure(_q.qsize()):
                    _delay = _backpressure.get_delay_ms(_q.qsize())
                    if _delay > 0:
                        await asyncio.sleep(_delay / 1000)

                # Record TTFT on first token
                if not _ng_ttft_recorded and n_tok == 1:
                    _ng_ttft_recorded = True
                    _ng_ttft_s = time.perf_counter() - _ng_gen_t0
                    _ng_ttft_ms_val = round(_ng_ttft_s * 1000, 1)
                    try:
                        from yunshu_gateway.middleware.prometheus_exporter import (
                            get_prometheus_metrics,
                        )

                        pm = get_prometheus_metrics()
                        pm.observe_histogram(
                            "ttft_seconds",
                            _ng_ttft_s,
                            labels={"model_id": self.model_label},
                        )
                    except Exception:
                        logger.debug(
                            "N-gram streaming TTFT prometheus recording failed",
                            exc_info=True,
                        )

                # Build logprobs for this token — n-gram spec decode does not
                # expose per-token logits in the queue-based streaming path.
                # When logprobs=True, return an empty list with a warning so
                # consumers get a valid (but empty) structure instead of None.
                _chunk_logprobs = None
                if logprobs:
                    _chunk_logprobs: list = []
                    if n_tok <= 1:
                        logger.debug(
                            "N-gram streaming path: real logprobs unavailable "
                            "(per-token logits not exposed via queue). "
                            "Returning empty logprobs list."
                        )

                yield GenerationOutput(
                    text=_clean_special_tokens(accumulated),
                    new_text=_clean_special_tokens(new_text),
                    prompt_tokens=prompt_tokens,
                    completion_tokens=n_tok,
                    finished=done,
                    finish_reason=finish_reason,
                    reasoning_tokens=_ng_consumer_thinking_tokens
                    if _ng_consumer_think_start is not None
                    else 0,
                    cached_tokens=0,
                    logprobs=_chunk_logprobs,
                    ttft_ms=_ng_ttft_ms_val,
                )
                if done:
                    break
        finally:
            # Decrement active fast path count (prevents model eviction mid-generation)
            # Release LoRA adapter ref acquired at start of streaming n-gram spec
            if (
                _ng_s_lora_applied
                and hasattr(self, "_lora_manager")
                and self._lora_manager is not None
            ):
                try:
                    self._lora_manager.release_adapter(lora_adapter)
                except Exception:
                    logger.debug(
                        "LoRA release failed in streaming n-gram spec path",
                        exc_info=True,
                    )
            _ng_fp_lock = getattr(self, "_fast_path_lock", None)
            if _ng_fp_lock is not None:
                with _ng_fp_lock:
                    self._active_fast_path_count -= 1
            if not future.done():
                future.cancel()
                with suppress(asyncio.CancelledError, Exception):
                    await future
            # Drain queue to unblock any pending call_soon_threadsafe from
            # the executor thread, preventing GPU work from continuing after
            # the consumer has stopped iterating.
            while not _q.empty():
                try:
                    _q.get_nowait()
                except asyncio.QueueEmpty:
                    break

    async def _generate_mtp(
        self,
        prompt: str | list[dict],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 1.0,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        logit_bias: dict[int, float] | None = None,
        stop: list[str] | None = None,
        stop_token_ids: list[int] | None = None,
        seed: int | None = None,
        enable_thinking: bool | None = None,
        logprobs: bool = False,
        top_logprobs: int | None = None,
        thinking_budget: int | None = None,
        cancel_event: asyncio.Event | None = None,
        json_schema: dict | str | None = None,
        xtc_probability: float = 0.0,
        xtc_threshold: float = 0.0,
        logits_processors: list | None = None,
        timeout_seconds: float = 300.0,
        lora_adapter: str | None = None,
    ) -> GenerationOutput:
        """Generate using MTP speculative decoding (built-in prediction heads).

        Uses the model's own MTP heads to propose draft tokens, then verifies
        via the backbone forward with n_confirmed=1 for zero-cost reject.
        Best for Qwen3.5 and other models with GatedDeltaNet SSM layers.
        """
        from .mlx_executor import get_mlx_executor

        # Build grammar constraint from json_schema if provided.
        # The constraint filters MTP draft logits so draft tokens respect
        # structured output requirements.
        _mtp_constraint = None
        if json_schema is not None:
            try:
                from mlx_lm.sample_utils import make_sampler as _make_s

                _tmp_sampler = _make_s(temp=0.0)
                _constrained_sampler = _build_constrained_sampler(
                    _tmp_sampler,
                    json_schema,
                    self._tokenizer,
                )
                _mtp_constraint = getattr(
                    _constrained_sampler,
                    "constraint",
                    None,
                )
            except Exception:
                logger.warning(
                    "Grammar constraint setup failed for MTP, continuing without",
                    exc_info=True,
                )
        import mlx.core as mx

        executor = get_mlx_executor()
        loop = asyncio.get_running_loop()

        tokenizer = self._tokenizer
        mtp_decoder = self._mtp_decoder

        # double-BOS guard (see non-streaming siblings) — route through
        # _apply_chat_template + _encode_prompt instead of raw apply_chat_template+encode.
        if isinstance(prompt, list) and prompt and isinstance(prompt[0], dict):
            text = self._apply_chat_template(prompt, enable_thinking=enable_thinking)
            input_ids = self._encode_prompt(tokenizer, text)
        else:
            text = prompt if isinstance(prompt, str) else str(prompt)
            input_ids = tokenizer.encode(text)
        prompt_tokens = len(input_ids)

        # Build EOS + stop token sets
        eos_ids: set[int] = set()
        if hasattr(tokenizer, "eos_token_id"):
            eid = tokenizer.eos_token_id
            if isinstance(eid, (list, tuple)):
                eos_ids.update(eid)
            elif eid is not None:
                eos_ids.add(eid)
        if stop_token_ids:
            eos_ids.update(stop_token_ids)
        stop_suffixes = []
        if stop:
            for s in stop:
                try:
                    ids = tokenizer.encode(s)
                    if len(ids) == 1:
                        eos_ids.add(ids[0])
                    elif len(ids) > 1:
                        stop_suffixes.append(s)
                except Exception:
                    logger.debug(
                        f"failed to encode stop sequence: {s!r}", exc_info=True
                    )

        if seed is not None:
            mx.random.seed(seed)

        # Build sampler for MTP path — applied to bonus tokens and rejection
        # corrections while the draft/verify comparison stays greedy.
        from mlx_lm.sample_utils import make_sampler

        # temp>0 → per-request sampler (no mlx-lm PRNG-trap collapse, seed
        # honored); temp==0 with filters → make_sampler (argmax, unchanged); fully
        # greedy-unconstrained → None.
        if temperature is not None and temperature > 1e-6:
            _mtp_sampler = _build_temp_sampler(
                temperature=temperature,
                top_p=top_p,
                top_k=top_k if top_k > 0 else 0,
                min_p=min_p,
                seed=seed,
                xtc_probability=xtc_probability,
                xtc_threshold=xtc_threshold,
            )
        elif top_p < 1.0 or top_k > 0 or min_p > 0 or xtc_probability > 0:
            _mtp_sampler = make_sampler(
                temp=temperature,
                top_p=top_p,
                top_k=top_k if top_k > 0 else 0,
                min_p=min_p,
                xtc_probability=xtc_probability,
                xtc_threshold=xtc_threshold,
            )
        else:
            _mtp_sampler = None

        # Use incremental detokenizer for correct multi-byte UTF-8
        detokenizer = tokenizer.detokenizer
        detokenizer.reset()

        def _run():
            return mtp_decoder.generate(
                input_ids,
                max_tokens=max_tokens,
                cancel_event=cancel_event,
                sampler=_mtp_sampler,
                repetition_penalty=repetition_penalty,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                logit_bias=logit_bias,
                constraint=_mtp_constraint,
            )

        # (self-audit fix B): keep _lora_applied permanently False so the
        # _lora_release() helper below is an inert no-op; the adapter lifecycle now
        # runs INSIDE _run_with_lora on the executor thread, serialized with
        # generate_step (max_workers=1) — the same keystone as _generate_fast.
        # Acquiring/releasing on the event loop mutated the shared model while a
        # different request's generation read it (cross-thread race).
        _lora_applied = False

        def _lora_release():
            if (
                _lora_applied
                and hasattr(self, "_lora_manager")
                and self._lora_manager is not None
            ):
                try:
                    self._lora_manager.release_adapter(lora_adapter)
                except Exception:
                    logger.warning("LoRA release failed in MTP path", exc_info=True)

        def _run_with_lora():
            _applied = False
            if lora_adapter and getattr(self, "_lora_manager", None) is not None:
                try:
                    _applied = self._lora_manager.acquire_adapter(lora_adapter)
                except Exception as _le:  # fail loud, don't serve base
                    raise RuntimeError(
                        f"LoRA adapter '{lora_adapter}' could not be applied"
                    ) from _le
                if not _applied:
                    raise RuntimeError(
                        f"LoRA adapter '{lora_adapter}' could not be applied"
                    )
            try:
                return _run()
            finally:
                if _applied and getattr(self, "_lora_manager", None) is not None:
                    try:
                        self._lora_manager.release_adapter(lora_adapter)
                    except Exception:
                        logger.debug(
                            "LoRA release failed (MTP executor)", exc_info=True
                        )

        _mtp_gen_t0 = time.perf_counter()
        try:
            token_ids = await asyncio.wait_for(
                loop.run_in_executor(executor, _run_with_lora),
                timeout=timeout_seconds,
            )
        except TimeoutError:
            logger.warning(f"MTP generation timed out after {timeout_seconds}s")
            try:
                import mlx.core as _mx

                await loop.run_in_executor(
                    executor, lambda: (_mx.synchronize(), _mx.clear_cache())
                )
            except Exception:
                logger.debug(
                    "GPU cache cleanup failed after MTP timeout", exc_info=True
                )
            _lora_release()
            return GenerationOutput(
                finished=True,
                finish_reason="error",
                prompt_tokens=prompt_tokens,
                completion_tokens=0,
                error=f"MTP generation timed out after {timeout_seconds}s",
                ttft_ms=0.0,
                cached_tokens=0,
            )
        except MemoryError:
            logger.warning(
                "OOM during MTP generation — returning memory_limit finish reason"
            )
            try:
                import mlx.core as _mx

                await loop.run_in_executor(
                    executor, lambda: (_mx.synchronize(), _mx.clear_cache())
                )
            except Exception:
                logger.debug("GPU cache cleanup failed after MTP OOM", exc_info=True)
            _lora_release()
            return GenerationOutput(
                finished=True,
                finish_reason="memory_limit",
                prompt_tokens=prompt_tokens,
                completion_tokens=0,
                error="OOM during MTP generation",
                ttft_ms=0.0,
                cached_tokens=0,
            )
        except RuntimeError as e:
            if "memory" in str(e).lower() or "out of" in str(e).lower():
                logger.warning(f"MLX OOM during MTP generation: {e}")
                try:
                    import mlx.core as _mx

                    await loop.run_in_executor(
                        executor, lambda: (_mx.synchronize(), _mx.clear_cache())
                    )
                except Exception:
                    logger.debug(
                        "GPU cache cleanup failed after MTP OOM (RuntimeError)",
                        exc_info=True,
                    )
                _lora_release()
                return GenerationOutput(
                    finished=True,
                    finish_reason="memory_limit",
                    prompt_tokens=prompt_tokens,
                    completion_tokens=0,
                    error=str(e),
                    ttft_ms=0.0,
                    cached_tokens=0,
                )
            _lora_release()
            # Return error output for non-OOM RuntimeError instead of
            # propagating to caller (which expects GenerationOutput).
            return GenerationOutput(
                finished=True,
                finish_reason="error",
                prompt_tokens=prompt_tokens,
                completion_tokens=0,
                error=f"RuntimeError during MTP generation: {e}",
                ttft_ms=0.0,
                cached_tokens=0,
            )
        except Exception as e:
            logger.error(f"Unexpected error during MTP generation: {e}", exc_info=True)
            _lora_release()
            # Return error output instead of propagating exception to caller.
            return GenerationOutput(
                finished=True,
                finish_reason="error",
                prompt_tokens=prompt_tokens,
                completion_tokens=0,
                error=f"MTP generation failed: {e}",
                ttft_ms=0.0,
                cached_tokens=0,
            )
        _mtp_ttft_s = time.perf_counter() - _mtp_gen_t0

        # Thinking budget enforcement (MTP decoder does not support it natively).
        # Detect <think/</think token IDs and truncate thinking content when the
        # budget is exceeded. This is a post-processing approximation — the MTP
        # decoder already generated all tokens, but we cap the output to respect
        # the budget, matching the behavior of _generate_fast.
        _mtp_thinking_tokens_used = 0
        _mtp_think_end_token = None
        _mtp_in_thinking = False
        _mtp_think_budget_truncate_idx = None
        if (thinking_budget is not None or enable_thinking) and token_ids:
            # bracketed-form helper (bare "</think" tokenized to 2 → guard failed).
            _mtp_think_start_token, _mtp_think_end_token = _resolve_think_token_ids(
                tokenizer
            )
            if _mtp_think_end_token is not None:
                for _i, _tid in enumerate(token_ids):
                    if not _mtp_in_thinking and _tid == _mtp_think_start_token:
                        _mtp_in_thinking = True
                    elif _mtp_in_thinking:
                        if _tid == _mtp_think_end_token:
                            _mtp_in_thinking = False
                        else:
                            _mtp_thinking_tokens_used += 1
                            if (
                                thinking_budget is not None
                                and _mtp_thinking_tokens_used >= thinking_budget
                            ):
                                _mtp_think_budget_truncate_idx = _i
                                break

        if _mtp_think_budget_truncate_idx is not None:
            token_ids = token_ids[:_mtp_think_budget_truncate_idx]
            if _mtp_think_end_token is not None:
                token_ids.append(_mtp_think_end_token)
                # Count the forced think_end_token in reasoning tokens so
                # reasoning_tokens + content_tokens == completion_tokens.
                _mtp_thinking_tokens_used += 1

        # Truncate at stop tokens (exclude stop token from output).
        # Use a temporary detokenizer to probe for suffix matches WITHOUT
        # corrupting the main detokenizer's state when a suffix is hit.
        # Previously, add_token(tid) was called before the suffix check,
        # so the suffix-triggering token leaked into detokenizer.text even
        # though token_ids was correctly truncated — causing the suffix
        # to appear in the output despite the trimming below.
        hit_stop = False
        hit_suffix = False
        _mtp_completion_count = len(token_ids)  # Track actual completion count
        for i, tid in enumerate(token_ids):
            if tid in eos_ids:
                token_ids = token_ids[:i]
                _mtp_completion_count = i
                hit_stop = True
                break
            # Probe suffix match using the main detokenizer, but be
            # prepared to roll back if it triggers.
            detokenizer.add_token(tid)
            if stop_suffixes and any(
                detokenizer.text.endswith(s) for s in stop_suffixes
            ):
                # Roll back: remove the suffix-triggering token from the
                # detokenizer so its state matches the truncated token_ids.
                # NaiveStreamingDetokenizer supports .tokens attribute.
                if hasattr(detokenizer, "tokens") and detokenizer.tokens:
                    detokenizer.tokens.pop()
                # Re-initialize detokenizer state from remaining tokens
                # to ensure .text is consistent (simply popping .tokens
                # does not update the internal byte buffer).
                _kept = (
                    list(detokenizer.tokens)
                    if hasattr(detokenizer, "tokens")
                    else token_ids[:i]
                )
                detokenizer.reset()
                for _t in _kept:
                    detokenizer.add_token(_t)
                _mtp_completion_count = i
                token_ids = token_ids[:i]
                hit_suffix = True
                break
        else:
            # No stop/suffix hit — completion count is all tokens
            _mtp_completion_count = len(token_ids)
        detokenizer.finalize()
        output_text = _clean_special_tokens(detokenizer.text)

        # Determine finish_reason with cancel awareness
        _cancelled = _is_cancelled(cancel_event)
        finish_reason = "stop" if _cancelled or hit_stop or hit_suffix else "length"

        # Record MTP stats + TTFT in Prometheus
        try:
            from yunshu_gateway.middleware.prometheus_exporter import (
                get_prometheus_metrics,
            )

            pm = get_prometheus_metrics()
            s = mtp_decoder.stats
            if s.total_cycles > 0:
                _ml = {"model_id": self.model_label}
                pm.set_gauge(
                    "mtp_acceptance_rate", s.accepts / s.total_cycles, labels=_ml
                )
                pm.set_counter("mtp_total_cycles", s.total_cycles, labels=_ml)
            pm.observe_histogram(
                "ttft_seconds", _mtp_ttft_s, labels={"model_id": self.model_label}
            )
        except Exception:
            logger.debug("MTP metrics export failed", exc_info=True)

        # Build logprobs from MTP output tokens — MTP decoder uses greedy
        # decoding internally and does not expose per-token logits.
        # Return None instead of fake 0.0 to avoid misleading consumers.
        _mtp_logprobs = None
        if logprobs and token_ids:
            _mtp_logprobs = None  # Real logprobs unavailable from MTP path

        # Reasoning parser: extract thinking tokens from MTP output text when
        # the tokenizer supports <think/</think single-token encoding.
        _mtp_reasoning_tok = _mtp_thinking_tokens_used
        if _mtp_reasoning_tok == 0 and output_text:
            try:
                from .reasoning_parser import get_reasoning_parser

                rp = get_reasoning_parser(self.model_name)
                rp_out = rp.parse(output_text)
                if rp_out.reasoning and rp_out.reasoning_tokens > 0:
                    _mtp_reasoning_tok = rp_out.reasoning_tokens
                    if rp_out.content != output_text:
                        output_text = rp_out.content
            except Exception:
                logger.debug("reasoning_parser failed in MTP path", exc_info=True)

        _lora_release()
        return GenerationOutput(
            text=output_text,
            new_text=output_text,
            prompt_tokens=prompt_tokens,
            completion_tokens=_mtp_completion_count,
            finished=True,
            finish_reason=finish_reason,
            reasoning_tokens=_mtp_reasoning_tok,
            cached_tokens=0,
            logprobs=_mtp_logprobs,
            ttft_ms=round(_mtp_ttft_s * 1000, 1),
        )

    async def _stream_generate_mtp(
        self,
        prompt: str | list[dict],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 1.0,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        logit_bias: dict[int, float] | None = None,
        logprobs: bool = False,
        top_logprobs: int | None = None,
        stop: list[str] | None = None,
        stop_token_ids: list[int] | None = None,
        seed: int | None = None,
        cancel_event: asyncio.Event | None = None,
        enable_thinking: bool | None = None,
        thinking_budget: int | None = None,
        timeout_seconds: float = 300.0,
        json_schema: dict | str | None = None,
        xtc_probability: float = 0.0,
        xtc_threshold: float = 0.0,
        logits_processors: list | None = None,
        lora_adapter: str | None = None,
    ) -> AsyncIterator[GenerationOutput]:
        """Stream generate using MTP speculative decoding (queue-based).

        Runs MTPDecoder on the executor thread and yields accepted tokens
        as they are verified by the backbone forward pass.
        """
        import mlx.core as mx
        from mlx_lm.models.cache import make_prompt_cache

        from .mlx_executor import get_mlx_executor

        # Build grammar constraint from json_schema if provided.
        # The constraint is used to filter MTP draft logits so draft tokens
        # respect structured output requirements.
        _mtp_grammar_constraint = None
        if json_schema is not None:
            try:
                from mlx_lm.sample_utils import make_sampler as _make_s

                _tmp_sampler = _make_s(temp=0.0)
                _constrained_sampler = _build_constrained_sampler(
                    _tmp_sampler,
                    json_schema,
                    self._tokenizer,
                )
                _mtp_grammar_constraint = getattr(
                    _constrained_sampler,
                    "constraint",
                    None,
                )
            except Exception:
                logger.warning(
                    "Grammar constraint setup failed for MTP streaming, "
                    "continuing without constraint filtering",
                    exc_info=True,
                )

        tokenizer = self._tokenizer
        model = self._model
        mtp_decoder = self._mtp_decoder

        # double-BOS guard (see non-streaming siblings) — route through
        # _apply_chat_template + _encode_prompt instead of raw apply_chat_template+encode.
        if isinstance(prompt, list) and prompt and isinstance(prompt[0], dict):
            text = self._apply_chat_template(prompt, enable_thinking=enable_thinking)
            input_ids = self._encode_prompt(tokenizer, text)
        else:
            text = prompt if isinstance(prompt, str) else str(prompt)
            input_ids = tokenizer.encode(text)
        prompt_tokens = len(input_ids)

        eos_ids: set[int] = set()
        stop_suffixes: list[str] = []
        if hasattr(tokenizer, "eos_token_id"):
            eid = tokenizer.eos_token_id
            if isinstance(eid, (list, tuple)):
                eos_ids.update(eid)
            elif eid is not None:
                eos_ids.add(eid)
        _eids = getattr(tokenizer, "eos_token_ids", None)
        if _eids is not None:  # may be a bare int (Qwen3.6-27B), not iterable
            eos_ids.update(_eids if isinstance(_eids, (list, tuple, set)) else (_eids,))
        if stop_token_ids:
            eos_ids.update(stop_token_ids)
        if stop:
            for s in stop:
                try:
                    ids = tokenizer.encode(s)
                    if len(ids) == 1:
                        eos_ids.add(ids[0])
                    elif len(ids) > 1:
                        stop_suffixes.append(s)
                except Exception:
                    logger.debug(
                        f"failed to encode stop sequence: {s!r}", exc_info=True
                    )

        if seed is not None:
            mx.random.seed(seed)

        # Build sampler for MTP streaming path — applied to bonus tokens,
        # rejection corrections, and first token (NOT draft/verify comparison).
        from mlx_lm.sample_utils import make_sampler

        # temp>0 → per-request sampler (no mlx-lm PRNG-trap collapse, seed
        # honored); temp==0 with filters → make_sampler (argmax, unchanged); fully
        # greedy-unconstrained → None.
        if temperature is not None and temperature > 1e-6:
            _mtp_sampler = _build_temp_sampler(
                temperature=temperature,
                top_p=top_p,
                top_k=top_k if top_k > 0 else 0,
                min_p=min_p,
                seed=seed,
                xtc_probability=xtc_probability,
                xtc_threshold=xtc_threshold,
            )
        elif top_p < 1.0 or top_k > 0 or min_p > 0 or xtc_probability > 0:
            _mtp_sampler = make_sampler(
                temp=temperature,
                top_p=top_p,
                top_k=top_k if top_k > 0 else 0,
                min_p=min_p,
                xtc_probability=xtc_probability,
                xtc_threshold=xtc_threshold,
            )
        else:
            _mtp_sampler = None

        # Inflight prefix sharing: register for concurrent KV block sharing
        _inflight_req_id = f"mtp-s-{id(self)}-{int(time.monotonic() * 1e6)}"
        try:
            from .inflight_prefix_sharing import get_inflight_tracker

            get_inflight_tracker().register(
                _inflight_req_id,
                token_ids=input_ids,
                kv_cache=None,
            )
        except Exception:
            logger.debug("MTP inflight prefix register failed", exc_info=True)

        def _unregister_inflight():
            try:
                from .inflight_prefix_sharing import get_inflight_tracker

                get_inflight_tracker().unregister(_inflight_req_id)
            except Exception:
                logger.debug("MTP inflight prefix unregister failed", exc_info=True)

        _lora_applied = False
        if (
            lora_adapter
            and hasattr(self, "_lora_manager")
            and self._lora_manager is not None
        ):
            try:
                _lora_applied = self._lora_manager.acquire_adapter(lora_adapter)
            except Exception as _le:  # fail loud, don't serve base
                raise RuntimeError(
                    f"LoRA adapter '{lora_adapter}' could not be applied"
                ) from _le
            if not _lora_applied:
                raise RuntimeError(
                    f"LoRA adapter '{lora_adapter}' could not be applied"
                )

        _sentinel = object()
        _q: asyncio.Queue = asyncio.Queue(maxsize=512)
        loop = asyncio.get_running_loop()
        from .streaming_optimizer import StreamingBackpressureController

        _mtp_timeout_cancel = threading.Event()
        _mtp_gen_t0 = time.perf_counter()
        _backpressure = StreamingBackpressureController(max_queue_size=100)

        def _put(item):
            # Backpressure-aware queue with retry (same logic as main streaming path)
            if _q.qsize() > 400:  # 78% of 512
                time.sleep(0.01)
            # Retry up to 3 times if the queue is full, sleeping 1ms between
            # attempts. Thread-safe: skip get_nowait() — see main streaming
            # _put for rationale (executor thread must not mutate asyncio Queue).
            for _attempt in range(4):  # 1 initial + 3 retries
                if not _q.full():
                    loop.call_soon_threadsafe(_q.put_nowait, item)
                    return
                if _attempt < 3:
                    time.sleep(0.001)
            logger.warning(
                "MTP streaming queue overflow after 3 retries — sending error sentinel. "
                "Client will see finish_reason=error."
            )
            try:
                loop.call_soon_threadsafe(
                    _q.put_nowait,
                    Exception("MTP streaming queue overflow — output truncated"),
                )
            except Exception:
                logger.debug(
                    "Failed to put error sentinel into MTP streaming queue",
                    exc_info=True,
                )

        def _run():
            try:
                ids = mx.array(input_ids)
                cache = make_prompt_cache(model)
                detokenizer = tokenizer.detokenizer
                detokenizer.reset()
                # Multi-token stop hold-back : the MTP path emitted each
                # accepted draft/bonus/correction token's text immediately and only
                # trimmed the stop on the COMPLETING token, leaking the earlier
                # tokens of a multi-token stop . Route
                # accepted-token text through the buffer; empty stop set → passthrough.
                _hb = StopHoldbackBuffer(stop_suffixes)

                # Thinking budget enforcement — detect <think/</think via single-token IDs
                _mtp_think_start_token = None
                _mtp_think_end_token = None
                if thinking_budget is not None or enable_thinking:
                    # bracketed-form helper (bare "</think" tokenized to 2 → guard failed).
                    _mtp_think_start_token, _mtp_think_end_token = (
                        _resolve_think_token_ids(tokenizer)
                    )
                _mtp_in_thinking = False
                _mtp_thinking_tokens_used = 0

                def _mtp_track_thinking(tid):
                    """Track thinking state and enforce budget. Returns True if budget exceeded."""
                    nonlocal _mtp_in_thinking, _mtp_thinking_tokens_used
                    if _mtp_think_start_token is None:
                        return False
                    if not _mtp_in_thinking and tid == _mtp_think_start_token:
                        _mtp_in_thinking = True
                    elif _mtp_in_thinking:
                        _mtp_thinking_tokens_used += 1
                        if tid == _mtp_think_end_token:
                            _mtp_in_thinking = False
                            return False
                        if (
                            thinking_budget is not None
                            and _mtp_thinking_tokens_used >= thinking_budget
                            and _mtp_think_end_token is not None
                        ):
                            _mtp_in_thinking = False
                            # Force-insert </think token
                            generated.append(_mtp_think_end_token)
                            detokenizer.add_token(_mtp_think_end_token)
                            _end_text = detokenizer.last_segment
                            _n = len(generated)
                            if _end_text:
                                _put((_end_text, _n, None, _mtp_think_end_token))
                            _put(("", _n, "stop", _mtp_think_end_token))
                            return True
                    return False

                # Prefill
                out, hidden = model(ids.reshape(1, -1), cache=cache, return_hidden=True)
                mx.synchronize()
                # Apply sampler to first token if available
                if _mtp_sampler is not None:
                    first = int(_mtp_sampler(out[0, -1:, :]).item())
                else:
                    first = int(mx.argmax(out[0, -1, :]).item())

                generated = [first]
                primary = first
                primary_h = hidden[:, -1:, :]

                if first in eos_ids:
                    # First token is stop — don't add to detokenizer, just signal stop
                    detokenizer.finalize()
                    _put(("", 0, "stop", first))
                    return

                # Yield first token via incremental detokenizer (through the
                # hold-back buffer so a stop prefix in the first token is withheld).
                detokenizer.add_token(first)
                _mtp_track_thinking(first)
                _first_emit = _hb.feed(_clean_special_tokens(detokenizer.last_segment))
                if _first_emit:
                    _put((_first_emit, 1, None, first))

                from .n_confirmed_patch import clear_rollback, restore_rollback

                # Emit an accepted token's cleaned text through the stop hold-back
                # buffer. Returns True if a multi-token string stop completed
                # (caller must set _early_stop and break). On an EOS token, call
                # _mtp_flush_held() instead to release genuine held text.
                def _mtp_emit(tok):
                    _seg = _clean_special_tokens(detokenizer.last_segment)
                    if stop_suffixes and any(
                        detokenizer.text.endswith(s) for s in stop_suffixes
                    ):
                        # include feed()'s pre-stop return (else content fused with
                        # the stop token is lost). Parity with the VLM/text paths (this MTP
                        # streaming path is currently unreachable, but fixed for correctness).
                        _tail = _hb.feed(_seg) + _hb.take_stopped()
                        if _tail:
                            _put((_tail, len(generated) - 1, None, tok))
                        _put(("", len(generated) - 1, "stop", tok))
                        return True
                    _emit = _hb.feed(_seg)
                    if _emit:
                        _put((_emit, len(generated), None, tok))
                    return False

                def _mtp_flush_held(tok):
                    # Release any text held back as a stop-prefix — it's genuine
                    # output when generation ends on an EOS/stop *token*.
                    _f = _hb.flush()
                    if _f:
                        _put((_f, len(generated) - 1, None, tok))

                _early_stop = (
                    False  # Set True when while loop breaks due to stop/cancel
                )

                while len(generated) < max_tokens:
                    # Check cancel_event
                    if _is_cancelled(cancel_event):
                        # Emit stop chunk before breaking so consumer sees finished=True
                        detokenizer.finalize()
                        _remaining = _hb.feed(detokenizer.last_segment) + _hb.flush()
                        if _remaining:
                            _put((_remaining, len(generated), None, None))
                        _put(("", len(generated), "stop", None))
                        _early_stop = True
                        break
                    # Check timeout-driven cancel from consumer
                    if _mtp_timeout_cancel.is_set():
                        detokenizer.finalize()
                        _remaining = _hb.feed(detokenizer.last_segment) + _hb.flush()
                        if _remaining:
                            _put((_remaining, len(generated), None, None))
                        _put(("", len(generated), "timeout", None))
                        _early_stop = True
                        break

                    # MTP draft — always greedy, with optional grammar constraint masking
                    # Checkpoint grammar constraint before draft
                    if _mtp_grammar_constraint is not None and hasattr(
                        _mtp_grammar_constraint, "checkpoint"
                    ):
                        try:
                            _mtp_grammar_constraint.checkpoint()
                        except Exception:
                            logger.debug("MTP grammar checkpoint failed", exc_info=True)
                    draft = mtp_decoder._mtp_draft(
                        primary_h,
                        primary,
                        constraint=_mtp_grammar_constraint,
                        generated_ids=generated,
                    )

                    # Verify: backbone forward [primary, draft] with n_confirmed=1
                    verify_out, verify_h = model(
                        mx.array([[primary, draft]]),
                        cache=cache,
                        return_hidden=True,
                        n_confirmed=1,
                    )
                    mx.synchronize()
                    # v0 MUST be greedy for spec decode acceptance check
                    v0 = int(mx.argmax(verify_out[0, 0, :]).item())
                    # MTP-PEN: Apply penalty/bias to bonus token logits (v1).
                    # Penalties are applied to the bonus token only — draft tokens
                    # are already committed via acceptance check.
                    _has_mtp_pen = (
                        repetition_penalty != 1.0
                        or frequency_penalty != 0.0
                        or presence_penalty != 0.0
                        or (logit_bias is not None and len(logit_bias) > 0)
                    )
                    _mtp_bonus_logits = verify_out[0, 1, :]
                    if _has_mtp_pen:
                        _mtp_token_hist = list(input_ids) + generated
                        _mtp_bonus_logits = _apply_spec_bonus_penalties(
                            _mtp_bonus_logits,
                            _mtp_token_hist,
                            len(input_ids),
                            repetition_penalty=repetition_penalty,
                            frequency_penalty=frequency_penalty,
                            presence_penalty=presence_penalty,
                            logit_bias=logit_bias,
                        )
                        verify_out[0, 1, :] = _mtp_bonus_logits
                    # v1 (bonus) can use sampler for non-greedy output
                    if _mtp_sampler is not None:
                        v1 = int(_mtp_sampler(verify_out[0, 1:2, :]).item())
                    else:
                        v1 = int(mx.argmax(verify_out[0, 1, :]).item())

                    if v0 == draft:
                        # Accept
                        clear_rollback(cache)
                        # Discard grammar constraint checkpoint (all accepted)
                        if _mtp_grammar_constraint is not None:
                            try:
                                if hasattr(
                                    _mtp_grammar_constraint, "discard_checkpoint"
                                ):
                                    _mtp_grammar_constraint.discard_checkpoint()
                            except Exception:
                                logger.debug(
                                    "MTP grammar discard_checkpoint failed",
                                    exc_info=True,
                                )
                        generated.append(draft)
                        if draft in eos_ids:
                            # Stop token — flush held text, exclude EOS from count
                            _mtp_flush_held(draft)
                            _put(("", len(generated) - 1, "stop", draft))
                            _early_stop = True
                            break

                        detokenizer.add_token(draft)
                        if _mtp_track_thinking(draft):
                            _early_stop = True
                            break
                        if _mtp_emit(draft):
                            _early_stop = True
                            break

                        # Advance grammar constraint with accepted draft token
                        if _mtp_grammar_constraint is not None and draft not in eos_ids:
                            try:
                                _mtp_grammar_constraint.advance(
                                    tokenizer.decode([draft])
                                )
                            except Exception:
                                logger.debug(
                                    "MTP grammar advance (draft accepted) failed",
                                    exc_info=True,
                                )

                        # Bonus token
                        generated.append(v1)
                        if v1 in eos_ids:
                            # Stop token — flush held text, exclude EOS from count
                            _mtp_flush_held(v1)
                            _put(("", len(generated) - 1, "stop", v1))
                            _early_stop = True
                            break

                        detokenizer.add_token(v1)
                        if _mtp_track_thinking(v1):
                            _early_stop = True
                            break
                        if _mtp_emit(v1):
                            _early_stop = True
                            break
                        # Advance grammar constraint with bonus token
                        if _mtp_grammar_constraint is not None and v1 not in eos_ids:
                            try:
                                _mtp_grammar_constraint.advance(tokenizer.decode([v1]))
                            except Exception:
                                logger.debug(
                                    "MTP grammar advance (bonus) failed", exc_info=True
                                )
                        primary = v1
                        primary_h = verify_h[:, -1:, :]
                    else:
                        # Reject: restore rollback (zero-cost)
                        restore_rollback(cache)
                        # Rollback grammar constraint to pre-draft state
                        if _mtp_grammar_constraint is not None and hasattr(
                            _mtp_grammar_constraint, "rollback"
                        ):
                            try:
                                _mtp_grammar_constraint.rollback()
                            except Exception:
                                logger.debug(
                                    "MTP grammar rollback failed", exc_info=True
                                )
                        # MTP-PEN: Apply penalty/bias to rejection correction logits (v0).
                        # The correction token is the first new token after the rejection.
                        if _has_mtp_pen:
                            _mtp_corr_logits = verify_out[0, 0, :]
                            _mtp_token_hist_corr = list(input_ids) + generated
                            _mtp_corr_logits = _apply_spec_bonus_penalties(
                                _mtp_corr_logits,
                                _mtp_token_hist_corr,
                                len(input_ids),
                                repetition_penalty=repetition_penalty,
                                frequency_penalty=frequency_penalty,
                                presence_penalty=presence_penalty,
                                logit_bias=logit_bias,
                            )
                            verify_out[0, 0, :] = _mtp_corr_logits
                        # Apply sampler to rejection correction token
                        if _mtp_sampler is not None:
                            v0 = int(_mtp_sampler(verify_out[0, 0:1, :]).item())
                        generated.append(v0)
                        if v0 in eos_ids:
                            # Stop token — flush held text, exclude EOS from count
                            _mtp_flush_held(v0)
                            _put(("", len(generated) - 1, "stop", v0))
                            _early_stop = True
                            break

                        detokenizer.add_token(v0)
                        if _mtp_track_thinking(v0):
                            _early_stop = True
                            break
                        if _mtp_emit(v0):
                            _early_stop = True
                            break
                        # Advance grammar constraint with correction token
                        if _mtp_grammar_constraint is not None and v0 not in eos_ids:
                            try:
                                _mtp_grammar_constraint.advance(tokenizer.decode([v0]))
                            except Exception:
                                logger.debug(
                                    "MTP grammar advance (correction) failed",
                                    exc_info=True,
                                )
                        primary = v0
                        # Re-feed correction token through rolled-back cache to
                        # get a hidden state consistent with the new primary token.
                        # Using verify_h[:, 0:1, :] here is WRONG because verify_h
                        # was computed before rollback — the cache state has changed.
                        _out_corr, _hid_corr = model(
                            mx.array([[v0]]),
                            cache=cache,
                            return_hidden=True,
                        )
                        mx.synchronize()
                        primary_h = _hid_corr[:, -1:, :]

                # Emit "length" finish chunk only when max_tokens exhausted naturally.
                # If the loop broke early (stop/cancel), a terminal chunk was already
                # emitted inside the loop — skip the spurious second one.
                if not _early_stop:
                    detokenizer.finalize()
                    _remaining = _hb.feed(detokenizer.last_segment) + _hb.flush()
                    if _remaining:
                        _put((_remaining, len(generated), None, None))
                    _put(("", len(generated), "length", None))
            except Exception as e:
                logger.error(f"MTP streaming generation failed: {e}", exc_info=True)
                # Finalize detokenizer to flush partial UTF-8 bytes before
                # reporting the error — without this, any bytes buffered in
                # the detokenizer's internal state are silently lost.
                try:
                    detokenizer.finalize()
                    _final_segment = detokenizer.last_segment
                    if _final_segment:
                        _put((_final_segment, len(generated), None, None))
                except Exception:
                    logger.debug(
                        "detokenizer finalize in MTP error handler failed",
                        exc_info=True,
                    )
                try:
                    mx.synchronize()
                    mx.clear_cache()
                except Exception:
                    logger.debug(
                        "GPU cache cleanup failed in MTP streaming error handler",
                        exc_info=True,
                    )
                _put(e)
            finally:
                _put(_sentinel)

        executor = get_mlx_executor()
        future = loop.run_in_executor(executor, _run)

        accumulated = ""
        n_tok = 0
        _mtp_ttft_recorded = False
        _mtp_ttft_ms_val = 0.0
        _mtp_gen_t0 = time.perf_counter()
        # Consumer-side thinking state mirrors GPU-side tracking for reporting
        _mtp_consumer_think_start = None
        _mtp_consumer_think_end = None
        _mtp_consumer_in_thinking = False
        _mtp_consumer_thinking_tokens = 0
        if thinking_budget is not None or enable_thinking:
            # bracketed-form helper (bare "</think" tokenized to 2 → guard failed).
            _mtp_consumer_think_start, _mtp_consumer_think_end = (
                _resolve_think_token_ids(tokenizer)
            )
        _mtp_fp_lock = getattr(self, "_fast_path_lock", None)
        if _mtp_fp_lock is not None:
            with _mtp_fp_lock:
                self._active_fast_path_count += 1
        try:
            while True:
                # Check cancel_event from consumer side
                if _is_cancelled(cancel_event):
                    # Yield terminal stop chunk so consumer sees finished=True
                    yield GenerationOutput(
                        text=_clean_special_tokens(accumulated) if accumulated else "",
                        new_text="",
                        prompt_tokens=prompt_tokens,
                        completion_tokens=n_tok,
                        finished=True,
                        finish_reason="stop",
                        ttft_ms=_mtp_ttft_ms_val,
                        cached_tokens=0,
                        reasoning_tokens=0,
                    )
                    break
                try:
                    item = await asyncio.wait_for(_q.get(), timeout=timeout_seconds)
                except TimeoutError:
                    logger.warning(
                        f"MTP streaming timeout: no token for {timeout_seconds}s"
                    )
                    _mtp_timeout_cancel.set()  # Signal GPU loop to stop
                    # Yield terminal output so consumer sees finished=True
                    yield GenerationOutput(
                        text=_clean_special_tokens(accumulated) if accumulated else "",
                        new_text="",
                        prompt_tokens=prompt_tokens,
                        completion_tokens=n_tok,
                        finished=True,
                        finish_reason="error",
                        error=f"MTP streaming timeout: no token for {timeout_seconds}s",
                        ttft_ms=_mtp_ttft_ms_val,
                        cached_tokens=0,
                        reasoning_tokens=0,
                    )
                    break
                if item is _sentinel:
                    break
                if isinstance(item, BaseException):
                    logger.warning(f"MTP streaming error: {item}")
                    # Yield terminal error output so consumer sees finished=True
                    yield GenerationOutput(
                        text=_clean_special_tokens(accumulated) if accumulated else "",
                        new_text="",
                        prompt_tokens=prompt_tokens,
                        completion_tokens=n_tok,
                        finished=True,
                        finish_reason="error",
                        error=str(item),
                        ttft_ms=_mtp_ttft_ms_val,
                        cached_tokens=0,
                        reasoning_tokens=0,
                    )
                    break
                new_text, tok_count, _fr_val, token_id = item
                # Backward compat: _fr_val may be bool or str or None
                if isinstance(_fr_val, bool):
                    finish_reason = "stop" if _fr_val else None
                    done = _fr_val
                else:
                    finish_reason = _fr_val
                    done = _fr_val is not None
                accumulated += new_text
                if len(accumulated) > _MAX_STREAMING_TEXT_BUFFER:
                    logger.error(
                        "Streaming text buffer exceeded 1MB limit (%d bytes) — truncating",
                        len(accumulated),
                    )
                    yield GenerationOutput(
                        text=_clean_special_tokens(accumulated),
                        new_text="",
                        prompt_tokens=prompt_tokens,
                        completion_tokens=tok_count,
                        finished=True,
                        finish_reason="length",
                        error="Streaming text buffer exceeded 1MB limit",
                        cached_tokens=0,
                        reasoning_tokens=_mtp_consumer_thinking_tokens
                        if _mtp_consumer_think_start is not None
                        else 0,
                        ttft_ms=_mtp_ttft_ms_val,
                    )
                    break
                n_tok = tok_count
                # Consumer-side thinking state tracking for reasoning_tokens reporting
                if _mtp_consumer_think_start is not None and isinstance(token_id, int):
                    if (
                        not _mtp_consumer_in_thinking
                        and token_id == _mtp_consumer_think_start
                    ):
                        _mtp_consumer_in_thinking = True
                    elif _mtp_consumer_in_thinking:
                        _mtp_consumer_thinking_tokens += 1
                        if token_id == _mtp_consumer_think_end:
                            _mtp_consumer_in_thinking = False

                # Streaming backpressure: slow down if client can't keep up
                if _backpressure.check_backpressure(_q.qsize()):
                    _delay = _backpressure.get_delay_ms(_q.qsize())
                    if _delay > 0:
                        await asyncio.sleep(_delay / 1000)

                # Record TTFT on first token
                if not _mtp_ttft_recorded and n_tok == 1:
                    _mtp_ttft_recorded = True
                    _mtp_ttft_s = time.perf_counter() - _mtp_gen_t0
                    _mtp_ttft_ms_val = round(_mtp_ttft_s * 1000, 1)
                    try:
                        from yunshu_gateway.middleware.prometheus_exporter import (
                            get_prometheus_metrics,
                        )

                        pm = get_prometheus_metrics()
                        pm.observe_histogram(
                            "ttft_seconds",
                            _mtp_ttft_s,
                            labels={"model_id": self.model_label},
                        )
                    except Exception:
                        logger.debug(
                            "MTP streaming TTFT prometheus recording failed",
                            exc_info=True,
                        )

                # Build logprobs for this token — MTP uses greedy decoding
                # internally and does not expose per-token logits in the
                # queue-based streaming path.
                # When logprobs=True, return an empty list with a warning so
                # consumers get a valid (but empty) structure instead of None.
                _chunk_logprobs = None
                if logprobs:
                    _chunk_logprobs: list = []
                    if n_tok <= 1:
                        logger.debug(
                            "MTP streaming path: real logprobs unavailable "
                            "(per-token logits not exposed via queue). "
                            "Returning empty logprobs list."
                        )

                yield GenerationOutput(
                    text=_clean_special_tokens(accumulated),
                    new_text=_clean_special_tokens(new_text),
                    prompt_tokens=prompt_tokens,
                    completion_tokens=n_tok,
                    finished=done,
                    finish_reason=finish_reason,
                    reasoning_tokens=_mtp_consumer_thinking_tokens
                    if _mtp_consumer_think_start is not None
                    else 0,
                    cached_tokens=0,
                    logprobs=_chunk_logprobs,
                    ttft_ms=_mtp_ttft_ms_val,
                )
                if done:
                    break
        finally:
            _unregister_inflight()
            if (
                _lora_applied
                and hasattr(self, "_lora_manager")
                and self._lora_manager is not None
            ):
                try:
                    self._lora_manager.release_adapter(lora_adapter)
                except Exception:
                    logger.warning(
                        "LoRA release failed in MTP streaming finally", exc_info=True
                    )
            # Decrement active fast path count (prevents model eviction mid-generation)
            _mtp_fp_lock = getattr(self, "_fast_path_lock", None)
            if _mtp_fp_lock is not None:
                with _mtp_fp_lock:
                    self._active_fast_path_count -= 1
            if not future.done():
                future.cancel()
                with suppress(asyncio.CancelledError, Exception):
                    await future
            # Drain queue to unblock any pending call_soon_threadsafe from
            # the executor thread, preventing GPU work from continuing after
            # the consumer has stopped iterating.
            while not _q.empty():
                try:
                    _q.get_nowait()
                except asyncio.QueueEmpty:
                    break

    @staticmethod
    def _normalize_messages_for_chat_template(messages: list[dict]) -> list[dict]:
        """Normalize messages before chat-template rendering.

        - Closes dangling <think> spans before raw <tool_call> XML in assistant
          content (Qwen 3.6 produces history where <think> is left open when a
          tool call follows — conditioning the next turn as still-reasoning).
        - Converts tool-call argument JSON strings to dicts for templates that
          iterate argument keys.
        """
        import json as _json

        normalized = []
        for m in messages:
            if m.get("role") != "assistant":
                normalized.append(m)
                continue
            m = dict(m)
            content = m.get("content")
            if (
                isinstance(content, str)
                and "<tool_call>" in content
                and "<think>" in content
            ):
                last_think = content.rfind("<think>")
                last_close = content.rfind("</think>")
                tool_pos = content.find("<tool_call>")
                if (
                    not (last_close >= last_think and last_close != -1)
                    and tool_pos > last_think
                ):
                    m["content"] = content[:tool_pos] + "</think>" + content[tool_pos:]
            tool_calls = m.get("tool_calls")
            if isinstance(tool_calls, list):
                patched = []
                for tc in tool_calls:
                    if not isinstance(tc, dict):
                        patched.append(tc)
                        continue
                    func = tc.get("function")
                    if isinstance(func, dict):
                        args = func.get("arguments")
                        if isinstance(args, str):
                            try:
                                parsed = _json.loads(args)
                            except Exception:
                                parsed = {"value": args}
                            tc = dict(tc)
                            tc["function"] = dict(func)
                            tc["function"]["arguments"] = (
                                parsed
                                if isinstance(parsed, dict)
                                else {"value": parsed}
                            )
                    patched.append(tc)
                m["tool_calls"] = patched
            normalized.append(m)
        return normalized

    @staticmethod
    def _encode_prompt(tokenizer, text: str) -> list[int]:
        """Encode a prompt string, avoiding a DOUBLE-BOS.

        When ``text`` came from apply_chat_template(tokenize=False) for a BOS-prepending
        model (Gemma/Llama/Mistral), it already starts with the literal bos_token; a plain
        encode() defaults to add_special_tokens=True and prepends BOS *again* → [BOS, BOS,
        …], which corrupts the first-token distribution. Qwen has no BOS so it was immune
        (and hid this). Mirrors mlx-lm's own generate.py guard. A raw /v1/completions
        string (no template, no leading BOS) still correctly gets its single BOS.
        """
        bos = getattr(tokenizer, "bos_token", None)
        # add_special unless the text already opens with a real bos_token string.
        add_special = not (isinstance(bos, str) and bos and text.startswith(bos))
        try:
            return tokenizer.encode(text, add_special_tokens=add_special)
        except TypeError:
            # Tokenizer.encode doesn't accept the kwarg — fall back (no double-BOS guard).
            return tokenizer.encode(text)

    def _apply_chat_template(
        self,
        messages: list[dict],
        enable_thinking: bool | None = None,
    ) -> str:
        """Apply chat template to convert messages to text."""
        thinking = (
            enable_thinking if enable_thinking is not None else self.enable_thinking
        )
        tokenizer = self._tokenizer

        # normalize OpenAI's `developer` role → `system` and the
        # legacy `function` role → `tool` BEFORE the family adapter + template run. Request
        # validation accepts both, but no chat template knows them → apply_chat_template
        # raises "Unknown role" → the WHOLE prompt collapses to the structureless plaintext
        # fallback (special tokens + chat structure lost → degraded generation). `developer`
        # is OpenAI's current recommended replacement for `system`, so clients send it
        # routinely. Doing this pre-adapter lets gemma4/mistral merge it as a system msg.
        if any(m.get("role") in ("developer", "function") for m in messages):
            _remapped = []
            for _m in messages:
                _r = _m.get("role")
                if _r in ("developer", "function"):
                    _m = dict(_m)
                    _m["role"] = "system" if _r == "developer" else "tool"
                _remapped.append(_m)
            messages = _remapped

        # Apply model-specific message adapter
        try:
            from yunshu_engine.message_adapter import adapt_messages

            messages = adapt_messages(messages, self.model_name)
        except Exception:
            logger.debug("message adapter failed", exc_info=True)

        # Safety normalization: close dangling <think> before <tool_call>,
        # convert tool-call argument strings to dicts (vllm-mlx pattern)
        messages = self._normalize_messages_for_chat_template(messages)

        if tokenizer and hasattr(tokenizer, "apply_chat_template"):
            try:
                clean = []
                for m in messages:
                    # coerce content=None → "" BEFORE the template. m.get("content",
                    # "") returns the default only when the key is MISSING; an explicit
                    # content=None (the canonical OpenAI agent-loop assistant turn
                    # {"role":"assistant","content":null,"tool_calls":[...]}) passes None through.
                    # A Jinja `{{ content }}` then renders Python None as the literal text "None"
                    # (verified) — corrupting every GLM/Llama/Qwen tool-calling turn with history;
                    # a `{{ "x" + content }}` template instead raises TypeError → plaintext
                    # fallback. The Gemma adapter and VLMEngine._format_prompt already coerce
                    # None→"" via _extract_text; this is the un-swept BatchedEngine text sibling.
                    _content = m.get("content")
                    msg = {
                        "role": m.get("role", "user"),
                        "content": "" if _content is None else _content,
                    }
                    # Preserve tool-related fields for correct template rendering
                    if m.get("tool_calls"):
                        msg["tool_calls"] = m["tool_calls"]
                    if m.get("tool_call_id"):
                        msg["tool_call_id"] = m["tool_call_id"]
                    if m.get("name"):
                        msg["name"] = m["name"]
                    # preserve reasoning_content — the family
                    # adapters thread it through, but this clean step dropped it, so
                    # DeepSeek-v3.2's thinking template (which asserts
                    # `reasoning_content or tool_calls` for an assistant turn after the
                    # last user msg) raised AssertionError → whole prompt collapsed to the
                    # plaintext fallback.
                    if m.get("reasoning_content"):
                        msg["reasoning_content"] = m["reasoning_content"]
                    clean.append(msg)
                # ASSISTANT PREFILL. A trailing assistant message means
                # "continue THIS turn" (Anthropic prefill, also OpenAI's) — the model
                # must continue from the prefilled text and NOT have a fresh assistant
                # turn opened after it. add_generation_prompt=True closes the prefill and
                # opens an empty turn (the prefix is ignored, output restarts). When the
                # last message is assistant, use continue_final_message=True instead so
                # the template keeps the turn open. Gated on trailing-assistant only, so
                # the normal case (last msg user/tool) is byte-identical to before.
                # prefill ONLY when the trailing assistant has non-empty STRING
                # content to continue. The gate (any trailing assistant) also matched
                # the canonical OpenAI agent-loop shape {"role":"assistant","content":null,
                # "tool_calls":[...]}, where continue_final_message makes the Jinja template
                # raise ValueError ("no content to continue") → not caught (only TypeError
                # was) → the WHOLE prompt collapsed to the plaintext fallback. A trailing
                # assistant with null/empty content (e.g. a tool_calls-only turn) is treated
                # as a completed turn → normal add_generation_prompt.
                _last = clean[-1] if clean else None
                _is_prefill = (
                    _last is not None
                    and _last.get("role") == "assistant"
                    and isinstance(_last.get("content"), str)
                    and _last["content"] != ""
                )
                kwargs = {"tokenize": False}
                if _is_prefill:
                    kwargs["continue_final_message"] = True
                else:
                    kwargs["add_generation_prompt"] = True
                if thinking is not None:
                    kwargs["enable_thinking"] = thinking
                try:
                    text = tokenizer.apply_chat_template(clean, **kwargs)
                except (TypeError, ValueError) as e:
                    _es = str(e)
                    if "continue_final_message" in _es:
                        # TypeError = tokenizer too old for the kwarg; ValueError
                        # = template rejects continue_final_message (e.g. "no content to
                        # continue"). Either way, retry without it — don't open a NEW turn
                        # after the prefix, and never collapse to the plaintext fallback.
                        logger.warning(
                            f"Model {self.model_name} rejected continue_final_message ({_es[:80]}); retrying without"
                        )
                        kwargs.pop("continue_final_message", None)
                        kwargs["add_generation_prompt"] = False
                        try:
                            text = tokenizer.apply_chat_template(clean, **kwargs)
                        except (TypeError, ValueError) as e2:
                            if "enable_thinking" in str(e2):
                                kwargs.pop("enable_thinking", None)
                                text = tokenizer.apply_chat_template(clean, **kwargs)
                            else:
                                raise
                    elif "enable_thinking" in _es:
                        logger.warning(
                            f"Model {self.model_name} doesn't support enable_thinking, retrying without"
                        )
                        kwargs.pop("enable_thinking", None)
                        text = tokenizer.apply_chat_template(clean, **kwargs)
                    else:
                        raise
                if text:
                    return text
            except Exception:
                logger.debug("chat template failed, using fallback", exc_info=True)

        # Generic fallback
        parts = []
        for m in messages:
            # same None→"" coercion as the template clean step, so the plaintext
            # fallback doesn't print a literal "None" for a content=null assistant turn.
            _fc = m.get("content")
            parts.append(
                f"{m.get('role', 'user').capitalize()}: {'' if _fc is None else _fc}"
            )
        parts.append("Assistant:")
        return "\n".join(parts)

    def has_active_requests(self) -> bool:
        """Check if engine has in-flight requests (including fast-path)."""
        if getattr(self, "_active_fast_path_count", 0) > 0:
            return True
        if self._engine_core:
            return bool(self._engine_core.has_active_requests)
        return False

    def resolve_model_id(self, model_id: str) -> bool:
        """Check if a model ID matches this engine."""
        if not self.model_name:
            return False
        display = (
            self.model_name.rsplit("/", 1)[-1]
            if "/" in self.model_name
            else self.model_name
        )
        known = {display, self.model_name}
        known_lower = {k.lower() for k in known if k}
        if model_id in known or model_id.lower() in known_lower:
            return True
        if "/" in model_id:
            stripped = model_id.rsplit("/", 1)[-1]
            if stripped in known or stripped.lower() in known_lower:
                return True
        return False

    def get_stats(self) -> dict:
        if self._engine_core:
            stats = self._engine_core.get_stats()
            stats["model"] = self.model_name
            stats["loaded"] = self._loaded
        else:
            stats = {"model": self.model_name, "loaded": self._loaded}
        if self._thinking_store is not None:
            stats["thinking_segment_store"] = self._thinking_store.get_stats()
        if self._adaptive_spec is not None:
            stats["adaptive_spec"] = self._adaptive_spec.get_stats()
        if getattr(self, "_spec_decoder", None) is not None:
            stats["spec_decode"] = {
                **self._spec_decoder._stats,
                "enabled": getattr(self, "_spec_enabled", False),
            }
        if self._ngram_proposer is not None:
            stats["ngram"] = {**self._ngram_stats, **self._ngram_proposer.get_stats()}
        if self._mtp_decoder is not None:
            s = self._mtp_decoder.stats
            stats["mtp"] = {
                "accepts": s.accepts,
                "rejects": s.rejects,
                "cooldowns": s.cooldowns,
                "tokens_generated": s.tokens_generated,
                "total_cycles": s.total_cycles,
            }
        if self._lookahead_reasoning is not None:
            stats["lookahead_reasoning"] = self._lookahead_reasoning.get_stats()
        # Metal kernel stats removed (kernels deleted — slower than mx.fast).
        # ANE embedding co-processor status (when enabled via YUNSHU_ANE_EMBEDDINGS=1)
        try:
            from .ane_embedding import get_ane_embedding_stats

            stats["ane_embeddings"] = get_ane_embedding_stats()
        except Exception:
            logger.debug("ane embedding stats failed", exc_info=True)
            stats["ane_embeddings"] = {"enabled": False, "active": False}
        # DeltaNet inversion status (when enabled via YUNSHU_DELTANET_INVERSION=1)
        stats["deltanet_inversion"] = {
            "enabled": getattr(self, "_deltanet_inversion_enabled", False),
            "hooks_registered": getattr(self, "_deltanet_inverter", None) is not None,
            **getattr(
                self,
                "_deltanet_inversion_stats",
                {
                    "evictions_captured": 0,
                    "inversions_attempted": 0,
                    "inversions_succeeded": 0,
                    "states_stored": 0,
                },
            ),
        }
        # Model preprocessor registry stats
        if (
            hasattr(self, "_preprocessor_registry")
            and self._preprocessor_registry is not None
        ):
            stats["model_preprocessor"] = self._preprocessor_registry.get_stats()
        stats["reasoning_tokens"] = getattr(self, "_total_reasoning_tokens", 0)
        stats["response_cache"] = {
            "hits": getattr(self, "_response_cache_hits", 0),
            "misses": getattr(self, "_response_cache_misses", 0),
        }
        # Prompt cache stats (exact-match KV state reuse)
        if hasattr(self, "_prompt_cache") and self._prompt_cache is not None:
            stats["prompt_cache"] = self._prompt_cache.get_stats()
        # Warm prompt preloading stats (prefill popular prefixes at startup)
        stats["warm_prompt_prefill"] = getattr(
            self,
            "_warm_prompt_stats",
            {
                "prompts_loaded": 0,
                "prompts_prefilled": 0,
                "prompts_skipped_cached": 0,
                "prompts_failed": 0,
                "total_tokens_prefilled": 0,
                "prefill_time_s": 0.0,
                "source": "none",
            },
        )
        # Inflight prefix sharing stats
        try:
            from .inflight_prefix_sharing import get_inflight_tracker

            stats["inflight_prefix_sharing"] = get_inflight_tracker().get_stats()
        except Exception:
            logger.debug("inflight prefix stats unavailable", exc_info=True)
            stats["inflight_prefix_sharing"] = {"enabled": False}
        return stats

    def backend_capabilities(self, model: Any = None) -> Any:
        """Derive this LM backbone's serving capabilities via the shared,
        backbone-agnostic `model_backend` layer (Part 2/3) — the same classifier
        VLMEngine uses, so both backends make the SAME reuse decision from the
        SAME logic. LM models are never mRoPE. Not memoized (cheap; called rarely
        for introspection). See docs/VLM_TEXT_KV_PREFIX.md."""
        from .model_backend import BackendKind, derive_capabilities

        if model is None:
            model = getattr(self, "model", None) or getattr(self, "_model", None)
        layers = []
        try:
            from mlx_lm.models.cache import make_prompt_cache

            layers = make_prompt_cache(model) if model is not None else []
        except Exception:
            logger.debug("LM cache probe failed; assuming non-reusable", exc_info=True)
        return derive_capabilities(BackendKind.LM, layers, is_mrope=False)

    def _cache_supports_trim(self, model: Any) -> bool:
        """Whether this model's KV cache can be safely trimmed/snapshotted.

        Prompt and prefix KV reuse both rely on trimming the cached KV. Hybrid
        models (e.g. Qwen3.5) mix full-attention KVCache layers (trimmable) with
        linear/recurrent ArraysCache layers whose state cannot be sliced back —
        reusing them corrupts the recurrent layers. Probe an empty cache once
        and memoize the verdict per model object.
        """
        cached = getattr(self, "_cache_trimmable_flag", None)
        if (
            cached is not None
            and getattr(self, "_cache_trimmable_model", None) is model
        ):
            return cached
        ok = True
        try:
            from mlx_lm.models.cache import make_prompt_cache

            probe = make_prompt_cache(model)
            # Sliding-window (RotatingKVCache) layers report is_trimmable()=True ONLY
            # while empty (offset < max_size); once a prompt exceeds the window the
            # ring buffer rotates and trim becomes UNSOUND (can_trim flips to False).
            # The empty probe here would memoize a stale True, so the prefix/prompt
            # cache would later trim a rotated cache → misordered KV reuse. Treat any
            # RotatingKVCache as non-trimmable up front (same bypass as hybrid).
            _rotating = False
            try:
                from mlx_lm.models.cache import RotatingKVCache

                _rotating = any(isinstance(c, RotatingKVCache) for c in probe)
            except Exception:
                _rotating = False
            if _rotating:
                ok = False
            else:
                try:
                    from mlx_lm.models.cache import can_trim_prompt_cache

                    ok = bool(can_trim_prompt_cache(probe))
                except Exception:
                    # Fall back to per-layer is_trimmable() inspection.
                    def _trimmable(c: Any) -> bool:
                        f = getattr(c, "is_trimmable", None)
                        if callable(f):
                            try:
                                return bool(f())
                            except Exception:
                                return False
                        # Standard attention caches expose keys/values and slice fine.
                        return hasattr(c, "keys") and hasattr(c, "values")

                    ok = all(_trimmable(c) for c in probe)
        except Exception:
            logger.debug(
                "cache trimmability probe failed; assuming trimmable", exc_info=True
            )
            ok = True
        self._cache_trimmable_flag = ok
        self._cache_trimmable_model = model
        if not ok:
            logger.info(
                "KV prefix/prompt caching disabled for %s: cache is not trimmable "
                "(hybrid/recurrent model)",
                self.model_name,
            )
        return ok

    def get_kv_cache_stats(self) -> dict:
        """Return KV cache statistics (prompt cache + prefix cache + paged KV)."""
        if self._kv_prefix_cache is not None:
            result = {"prefix_cache": self._kv_prefix_cache.get_stats()}
        else:
            result = {"prefix_cache": {"enabled": False}}
        # : surface the PromptCacheManager stats too. Exact-repeat requests
        # hit the prompt cache (full-KV exact match) which SHADOWS the partial-
        # prefix cache (see _run: `if not _pc_hit`), so prefix_cache.hit_rate
        # legitimately stays 0 for repeats while the prompt cache serves them.
        # Reporting only prefix_cache made caching look broken (hit_rate always 0)
        # when it was actually working via the prompt cache.
        if getattr(self, "_prompt_cache", None) is not None:
            try:
                result["prompt_cache"] = self._prompt_cache.get_stats()
            except Exception:
                logger.debug("prompt cache stats unavailable", exc_info=True)
        if self._engine_core:
            paged = self._engine_core.get_kv_cache_stats()
            result["paged_kv"] = paged
        return result

    def get_radix_tree_stats(self) -> dict:
        """Return RadixTree statistics (node count, eviction metrics, block usage)."""
        if not self._engine_core:
            return {"enabled": False}
        # Attr is `scheduler`, not `_scheduler` — the old typo made this always
        # return {"enabled": False}, masking real RadixTree state .
        scheduler = getattr(self._engine_core, "scheduler", None) or getattr(
            self._engine_core, "_scheduler", None
        )
        if scheduler is None:
            return {"enabled": False}
        kv_mgr = getattr(scheduler, "_kv_manager", None) or getattr(
            scheduler, "kv_manager", None
        )
        if kv_mgr is None:
            return {"enabled": False}
        # In tiered mode (YUNSHU_SSD_CACHE_DIR), kv_mgr is a TieredKVCacheManager
        # whose RadixTree lives on `.hot` (: reach through the wrapper, else
        # the stat is silently empty in tiered config).
        tree = getattr(kv_mgr, "_radix_tree", None)
        if tree is None:
            hot = getattr(kv_mgr, "hot", None)
            tree = getattr(hot, "_radix_tree", None) if hot is not None else None
        if tree is None:
            return {"enabled": False}
        return {"enabled": True, **tree.get_stats()}

    @staticmethod
    def _extract_model_arch(model: Any) -> dict:
        """Extract model architecture parameters for KV cache sizing.

        Reads from the model's config attribute (standard HuggingFace pattern).
        Returns kwargs dict for EngineCoreConfig.
        """
        if model is None:
            return {}

        config = getattr(model, "config", None) or getattr(model, "args", None)

        def _first(obj, names, default=0):
            for n in names:
                v = getattr(obj, n, None)
                if isinstance(v, int) and v > 0:
                    return v
            return default

        # Try several naming conventions (HF + mlx-lm ModelArgs variants).
        num_layers = (
            _first(config, ("num_hidden_layers", "n_layers", "num_layers"))
            if config
            else 0
        )
        # Robust fallback: count the actual decoder layers on the model.
        if not num_layers:
            layers = getattr(model, "layers", None)
            if layers is None:
                layers = getattr(getattr(model, "model", None), "layers", None)
            try:
                num_layers = len(layers) if layers is not None else 0
            except TypeError:
                num_layers = 0

        num_kv_heads = (
            _first(config, ("num_key_value_heads", "n_kv_heads", "num_kv_heads"))
            if config
            else 0
        )
        num_attn_heads = (
            _first(config, ("num_attention_heads", "n_heads", "num_heads"))
            if config
            else 0
        )
        if not num_kv_heads:
            num_kv_heads = num_attn_heads  # MHA models: kv heads == attn heads
        head_dim = _first(config, ("head_dim", "kv_head_dim")) if config else 0
        if not head_dim:
            hidden = (
                _first(config, ("hidden_size", "dim", "model_dim")) if config else 0
            )
            if hidden and num_attn_heads:
                head_dim = hidden // num_attn_heads

        if num_layers and num_kv_heads and head_dim:
            return {
                "num_layers": num_layers,
                "num_kv_heads": num_kv_heads,
                "head_dim": head_dim,
            }
        return {}

    def _ensure_memory_guard(self):
        """Lazily build a per-engine MemoryGuard so the preflight check actually
        fires on the DEFAULT FAST PATH.

        Previously `_check_memory_guard` read `self._engine_core._memory_guard`,
        but `_engine_core` is None in default serving, so the preflight was a
        guaranteed no-op for the path that actually serves users — a safety valve
        that never fired. We build a guard from the loaded model's arch on first
        use. It is a SAFETY VALVE, not an admission throttle: it uses a permissive
        KV budget (0.6 of the working set, vs the engine-loop's 0.25) so it only
        rejects a request whose KV alone would blow memory, never throttling
        normal traffic. Fail-open: any construction error → None (no rejection).
        The gateway's token-count `validate_prefill_memory` remains the primary
        per-request guard; this is defense-in-depth on actual memory.
        """
        g = getattr(self, "_fastpath_memory_guard", "unset")
        if g != "unset":
            return g
        # Prefer the engine-loop's already-configured guard if present.
        guard = getattr(self._engine_core, "_memory_guard", None)
        if guard is None:
            try:
                arch = self._extract_model_arch(self._model)
                if not arch:
                    self._fastpath_memory_guard = None
                    return None
                from .memory_guard import MemoryGuard
                from .memory_monitor import MemoryMonitor

                kv_budget = 0
                try:
                    from .utils.hardware import get_hardware_info

                    kv_budget = int(get_hardware_info().max_working_set_bytes * 0.6)
                except Exception:
                    logger.debug("fast-path guard: hw info failed", exc_info=True)
                monitor = MemoryMonitor(max_kv_cache_memory=kv_budget)
                _cfg = getattr(self._model, "config", None) or getattr(
                    self._model, "args", None
                )
                monitor.set_model_info(
                    num_layers=arch["num_layers"],
                    num_kv_heads=arch["num_kv_heads"],
                    head_dim=arch["head_dim"],
                    num_attention_heads=getattr(_cfg, "num_attention_heads", None)
                    if _cfg
                    else None,
                )
                monitor.set_baseline_memory()
                guard = MemoryGuard(memory_monitor=monitor)
            except Exception:
                logger.debug(
                    "fast-path memory guard construction failed", exc_info=True
                )
                guard = None
        self._fastpath_memory_guard = guard
        return guard

    def _check_memory_guard(
        self,
        prompt: str | list,
        max_tokens: int,
    ) -> GenerationOutput | None:
        """Run memory guard preflight check. Returns None if OK.

        Returns a GenerationOutput with finish_reason="memory_limit"
        if the memory guard rejects the request.
        """
        guard = self._ensure_memory_guard()
        if guard is None:
            return None

        # Estimate prompt tokens. This is a defense-in-depth MEMORY preflight (the gateway
        # already ran validate_prefill_memory with an exact count) — it only needs a rough,
        # conservative estimate, so DO NOT pay a full tokenizer.encode on the event loop
        # here (it inflates TTFT; the real encode happens later in _encode_prompt). For
        # chat messages especially, the old `encode(str(prompt))` tokenized the Python
        # repr — both wasteful and inaccurate. Use a cheap char-based estimate (~3 chars/
        # token, deliberately low divisor → over-estimate, the safe direction for a guard).
        if isinstance(prompt, list):
            _chars = 0
            for _m in prompt:
                if isinstance(_m, dict):
                    _c = _m.get("content")
                    if isinstance(_c, str):
                        _chars += len(_c)
                    elif isinstance(_c, list):
                        for _part in _c:
                            if isinstance(_part, dict) and isinstance(
                                _part.get("text"), str
                            ):
                                _chars += len(_part["text"])
                else:
                    _chars += len(str(_m))
            num_prompt_tokens = _chars // 3 + 16  # +16 for chat-template framing
        elif isinstance(prompt, str):
            num_prompt_tokens = len(prompt) // 3 + 16
        else:
            num_prompt_tokens = len(str(prompt).split()) * 2

        ok, reason = guard.preflight_check(
            num_prompt_tokens=num_prompt_tokens,
            max_tokens=max_tokens,
        )
        if not ok:
            logger.info(f"Memory guard rejected request: {reason}")
            return GenerationOutput(
                finished=True,
                finish_reason="memory_limit",
                prompt_tokens=num_prompt_tokens,
                completion_tokens=0,
                error=f"Memory guard rejected: {reason}",
                ttft_ms=0.0,
                cached_tokens=0,
            )
        return None


from .text_utils import cache_tokenizer_vocab as _cache_tokenizer_vocab


def _clean_special_tokens(text: str) -> str:
    """Remove special tokens from output ."""
    if not text:
        return ""
    import re

    text = re.sub(r"<\|im_end\|>", "", text)
    text = re.sub(r"<\|endoftext\|>", "", text)
    text = re.sub(r"<\|end\|>", "", text)
    return text
