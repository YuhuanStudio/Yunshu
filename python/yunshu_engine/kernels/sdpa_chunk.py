# Upstream (inspired): jundot/omlx (Apache-2.0) omlx/patches/sdpa256_attention.py @ 03bc3dd7
# Patches upstream mlx-vlm / mlx-lm symbols (see vendor.json kind=patches; `just vendor-check` flags source changes)
"""Bound the score matrix of head-dim-256 chunked prefill on pre-NAX GPUs (M1-M4).

MLX 0.32.3 (``ScaledDotProductAttention::use_fallback``) runs head-dim-256 attention with more
than 8 query rows through the unfused fallback unless the GPU has NAX (M5 and macOS 26.2+) or the
call is a square causal bf16/fp16 prefill of at least 2048 rows. A chunked prefill (the prompt's
queries against a longer KV) is neither, so the fallback materialises
``[n_heads, q_len, kv_len]`` scores in the query dtype, several times over (scores, mask select,
softmax): quadratic in context.

oMLX bounds it with ``force_fused=True`` kernels. That is a different arithmetic from the unfused
fallback and would change every M1-M4 prefill, so Yunshu does not take it. Instead the same
fallback runs on query-row chunks sized to a score budget: every chunk keeps the full key axis and
gets its rows of the same ``q_idx >= k_idx`` mask the fallback builds, so each row sees the same
elementwise ops, the same softmax over the same keys, and a matmul whose reduction axes are
unchanged (batched, so no split-K); only the number of rows per dispatch differs. Calls whose
scores fit the budget are not touched.

Gate: Apple GPU generation < 17 (M1-M4; see ``lane_qmm._generation``), head dim 256, more than 8
query rows, plain (not quantized) KV, scores above ``SCORE_BUDGET``. M5 without NAX (older macOS)
is not detected and keeps stock behaviour.
"""

from __future__ import annotations

import logging
import sys
from functools import cache
from typing import Any

import mlx.core as mx

logger = logging.getLogger(__name__)

HEAD_DIM = 256
MIN_Q_LEN = 9  # MLX: queries <= 8 take the vector kernel
SCORE_BUDGET = 1 << 30  # bytes of one score tensor per dispatch; the fallback holds ~3
MIN_ROWS = 16
_STATE: dict[str, Any] = {"installed": False, "chunked_calls": 0, "force_device": None}


@cache
def _device_generation() -> int | None:
    try:
        from .tensorfold.lane_qmm import _generation

        if not mx.metal.is_available():
            return None
        return _generation(str(mx.device_info().get("architecture", "")))
    except Exception:
        return None


def pre_nax_device() -> bool:
    """M1-M4 (generation < 17). ``_STATE["force_device"]`` overrides for tests."""
    forced = _STATE["force_device"]
    if forced is not None:
        return bool(forced)
    gen = _device_generation()
    return gen is not None and gen < 17


def _stock_is_fused(queries, keys, mask) -> bool:
    """The one unfused-exempt shape on pre-NAX: square causal non-fp32 prefill >= 2048 rows."""
    return (
        queries.dtype != mx.float32
        and isinstance(mask, str)
        and mask == "causal"
        and queries.shape[-2] >= 2048
        and queries.shape[-2] == keys.shape[-2]
    )


def score_bytes(queries, keys) -> int:
    """Bytes of one fallback score tensor [B, n_q_heads, q_len, kv_len] in the query dtype."""
    batch = int(queries.shape[0]) if queries.ndim == 4 else 1
    return (
        batch
        * int(queries.shape[-3])
        * int(queries.shape[-2])
        * int(keys.shape[-2])
        * queries.dtype.size
    )


def rows_per_chunk(queries, keys, budget: int | None = None) -> int:
    budget = SCORE_BUDGET if budget is None else budget
    per_row = max(1, score_bytes(queries, keys) // max(1, int(queries.shape[-2])))
    return max(MIN_ROWS, budget // per_row)


def should_chunk(queries, keys, cache, mask, budget: int | None = None) -> bool:
    """Never raises: any odd input keeps the stock call."""
    try:
        if (
            queries.ndim != 4
            or queries.shape[-1] != HEAD_DIM
            or keys.shape[-1] != HEAD_DIM
        ):
            return False
        if queries.shape[-2] < MIN_Q_LEN:
            return False
        if cache is not None and (
            hasattr(cache, "bits") or "TurboQuant" in type(cache).__name__
        ):
            return False
        if mask is not None and not isinstance(mask, (str, mx.array)):
            return False
        if isinstance(mask, str) and mask != "causal":
            return False
        if isinstance(mask, mx.array) and not 1 <= mask.ndim <= 4:
            return False
        if not pre_nax_device() or _stock_is_fused(queries, keys, mask):
            return False
        if int(queries.shape[-3]) % max(1, int(keys.shape[-3])):
            return False
        return score_bytes(queries, keys) > (SCORE_BUDGET if budget is None else budget)
    except Exception:
        return False


def chunked_sdpa(
    stock,
    queries,
    keys,
    values,
    cache,
    scale,
    mask,
    sinks=None,
    budget: int | None = None,
):
    """``stock`` over query-row chunks; each chunk keeps all keys (see module docstring)."""
    q_len, kv_len = int(queries.shape[-2]), int(keys.shape[-2])
    rows = rows_per_chunk(queries, keys, budget)
    causal = isinstance(mask, str)
    per_row_mask = (
        isinstance(mask, mx.array) and mask.ndim >= 2 and mask.shape[-2] == q_len
    )
    k_idx = mx.arange(kv_len)[None, :] if causal else None
    outs = []
    for start in range(0, q_len, rows):
        stop = min(q_len, start + rows)
        if causal:
            # MLX's own fallback mask: q_idx (offset to the end of the keys) >= k_idx
            q_idx = mx.arange(start + kv_len - q_len, stop + kv_len - q_len)[:, None]
            chunk_mask = q_idx >= k_idx
        elif per_row_mask:
            chunk_mask = mask[..., start:stop, :]
        else:
            chunk_mask = mask
        out = stock(
            queries[..., start:stop, :], keys, values, cache, scale, chunk_mask, sinks
        )
        # Evaluate each chunk so its scores are freed before the next is built.
        mx.eval(out)
        outs.append(out)
    _STATE["chunked_calls"] += 1
    if _STATE["chunked_calls"] == 1:
        logger.info(
            "sdpa chunking engaged (pre-NAX, head_dim 256): q_len=%d kv_len=%d rows/chunk=%d",
            q_len,
            kv_len,
            rows,
        )
    return outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=-2)


def wrap(stock):
    def scaled_dot_product_attention(
        queries, keys, values, cache, scale, mask=None, sinks=None
    ):
        if should_chunk(queries, keys, cache, mask):
            return chunked_sdpa(stock, queries, keys, values, cache, scale, mask, sinks)
        return stock(queries, keys, values, cache, scale, mask, sinks)

    scaled_dot_product_attention._yunshu_sdpa_chunk = True  # type: ignore[attr-defined]
    return scaled_dot_product_attention


def install() -> bool:
    """Wrap mlx-vlm's and mlx-lm's base SDPA and rebind the model modules that copied it.
    No-op (False) off pre-NAX hardware, so M5 and CPU hosts run the stock functions untouched."""
    from .. import settings

    if _STATE["installed"] or not pre_nax_device():
        return False
    if not settings.get_bool("YUNSHU_SDPA_CHUNK"):
        return False  # experimental until the M1-M4 bitwise A/B (see the setting's decide=)
    for base_name, prefix in (
        ("mlx_vlm.models.base", "mlx_vlm.models."),
        ("mlx_lm.models.base", "mlx_lm.models."),
    ):
        try:
            base = __import__(base_name, fromlist=["scaled_dot_product_attention"])
        except ImportError:
            continue
        original = getattr(base, "scaled_dot_product_attention", None)
        if original is None or getattr(original, "_yunshu_sdpa_chunk", False):
            continue
        wrapped = wrap(original)
        base.scaled_dot_product_attention = wrapped
        for name, mod in list(sys.modules.items()):
            if (
                mod is not None
                and name.startswith(prefix)
                and getattr(mod, "scaled_dot_product_attention", None) is original
            ):
                mod.scaled_dot_product_attention = wrapped
    _STATE["installed"] = True
    return True
