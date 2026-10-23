# Upstream (derived): ml-explore/mlx (MIT), steel_attention_nax.h @ v0.32.3
# Patches upstream mlx-vlm symbols (see vendor.json kind=patches; `just vendor-check` flags source changes)
"""Prefill attention of the Qwen3.5-family full-attention layers (D=256, GQA) with a smaller query tile.

MLX 0.32.3 runs head-dim-256 prefill attention through ``attention_nax_dsplit`` with BQ=64 (8
simdgroups). This module compiles the very same installed kernel body for BQ=32 (4 simdgroups: more
resident threadgroups per core) and replaces its score-exchange layout (per-lane-linear, 16-way bank
conflicts) with a lane-minor one. Per-row arithmetic is untouched: same BK=32 key blocks in the same
order, same D-split reduction ``slot0 + slot1``, same online softmax and normalisation. Output is
bit-identical to ``mx.fast.scaled_dot_product_attention`` (checked by tests/unit and the GPU identity
probe scripts/research/prefillspeed_attn_identity.py); only the schedule changes.

Engaged only for: MLX 0.32.3, M5-generation GPU, bf16, batch 1, causal mask, no sinks, a plain (not
quantized) KV cache, 24/4 heads of D=256 (any GQA factor), and at least ``MIN_ROWS`` query rows on the legacy path. Qualified split-invariant 27B
prompt scopes use BQ32 at every positive query length, including unaligned tails; their arithmetic
is explicitly named in APC keys. Short-piece outputs can differ from old stock MLX. Decode and
speculative verify keep their original dispatch. Everything else calls the stock function.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import logging
import re
from functools import cache
from pathlib import Path
from typing import Any, cast

import mlx.core as mx

from . import gdn_prefill

logger = logging.getLogger(__name__)

BQ, BK, BD, WM, WN = 32, 32, 256, 2, 2
MIN_ROWS = 1024
_SCALE = BD**-0.5
_STATE: dict[str, Any] = {
    "installed": False,
    "enabled": False,
    "dispatches": 0,
    "stock": 0,
    "why": {},
}


@cache
def header(align_q: bool, align_k: bool) -> str:
    """The installed MLX D-split attention, as a callable function with fixed function constants."""
    root = Path(str(importlib.metadata.distribution("mlx").locate_file("mlx/include")))
    seen = {"mlx/backend/metal/kernels/utils.h"}  # mx.fast.metal_kernel prepends utils

    def expand(name: str) -> str:
        if name in seen:
            return ""
        seen.add(name)
        text = (root / name).read_text()
        return re.sub(
            r'^\s*#include "([^"]+)"', lambda m: expand(m[1]), text, flags=re.M
        ).replace("#pragma once", "")

    path = "mlx/backend/metal/kernels/steel/attn/kernels/steel_attention_nax.h"
    text = (root / path).read_text()
    prefix = text[: text.index("// clang-format off")]
    at = text.index("void attention_nax_dsplit(")
    begin = text.rfind("template <", 0, at)
    body = text[begin:]
    opening = body.index("{ // clang-format on")
    signature, impl = body[:opening], body[opening:]
    signature = re.sub(r"\[\[.*?\]\]", "", signature, flags=re.S)
    signature = signature.replace(
        " void attention_nax_dsplit(", " METAL_FUNC void yunshu_attn_dsplit("
    )
    signature = signature.replace("const constant Attn", "const thread Attn")
    signature = signature.replace(
        "uint3 lid )", "uint3 lid, threadgroup AccumType (&s_xchg)[WM][WN][BK * 16])"
    )
    declaration = "threadgroup AccumType s_xchg[WM][WN][TK * kEPF * 32];"
    if declaration not in impl or "(&s_xchg)" not in signature:
        raise ValueError("native D-split scratch signature changed")
    impl = impl.replace(
        declaration, 'static_assert(TK * kEPF * 32 == BK * 16, "scratch shape");'
    )
    old = "base + ik * kEPF + i]"
    if impl.count(old) != 3:
        raise ValueError("native exchange indexing changed")
    impl = impl.replace(old, "(ik * kEPF + i) * 32 + short(simd_lane_id)]")
    values = {
        "align_Q": align_q,
        "align_K": align_k,
        "has_mask": False,
        "do_causal": True,
        "has_sinks": False,
    }
    prefix = re.sub(
        r"constant bool (\w+) \[\[function_constant\(\d+\)\]\];",
        lambda m: f"constant bool {m[1]} = {'true' if values[m[1]] else 'false'};",
        prefix,
    )
    prefix = re.sub(
        r'^\s*#include "([^"]+)"', lambda m: expand(m[1]), prefix, flags=re.M
    )
    return prefix + signature + impl


@cache
def kernel(align_q: bool, align_k: bool, heads: int, kv_heads: int) -> Any:
    h = header(align_q, align_k)
    source = f"""
    threadgroup float s_xchg[{WM}][{WN}][{BK} * 16];
    AttnParams params;
    params.B = 1; params.H = {heads}; params.D = {BD};
    params.qL = Q_shape[2]; params.kL = K_shape[2]; params.gqa_factor = {heads // kv_heads};
    params.scale = {_SCALE}f;
    params.NQ = (params.qL + {BQ} - 1) / {BQ};
    params.NK = (params.kL + {BK} - 1) / {BK};
    params.NQ_aligned = params.qL / {BQ}; params.NK_aligned = params.kL / {BK};
    params.qL_rem = params.qL % {BQ}; params.kL_rem = params.kL % {BK};
    params.qL_off = params.kL - params.qL;
    for (int i = 0; i < 3; i++) {{
      params.Q_strides[i] = Q_strides[i];
      params.K_strides[i] = K_strides[i];
      params.V_strides[i] = V_strides[i];
    }}
    params.O_strides[0] = {heads} * params.qL * {BD};
    params.O_strides[1] = params.qL * {BD}; params.O_strides[2] = {BD};
    yunshu_attn_dsplit<bfloat, {BQ}, {BK}, {BD}, {WM}, {WN}, float, float>(Q, K, V, O, &params,
      nullptr, nullptr, nullptr, thread_index_in_simdgroup,
      simdgroup_index_in_threadgroup, threadgroup_position_in_grid,
      thread_position_in_threadgroup, s_xchg);
    """
    name = (
        "yunshu_attn_prefill_" + hashlib.sha256((h + source).encode()).hexdigest()[:16]
    )
    return mx.fast.metal_kernel(
        name=name,
        input_names=["Q", "K", "V"],
        output_names=["O"],
        header=h,
        source=source,
        ensure_row_contiguous=False,
    )


def eligible(
    q_shape: tuple[int, ...], k_shape: tuple[int, ...], dtype: Any, scale: float
) -> bool:
    """Pure shape/dtype test; the caller has already checked mask/sinks/cache kind."""
    if len(q_shape) != 4 or len(k_shape) != 4:
        return False
    b, h, rows, d = q_shape
    bk, hk, keys, dk = k_shape
    return bool(
        b == 1
        and bk == 1
        and d == BD
        and dk == BD
        and hk > 0
        and h % hk == 0
        and rows
        >= (
            1
            if gdn_prefill.split_enabled()
            and (gdn_prefill.in_prefill() or gdn_prefill.in_tail())
            else MIN_ROWS
        )
        and keys >= rows
        and dtype == mx.bfloat16
        and abs(scale - _SCALE) < 1e-12
    )


def skip_reason(queries, keys, values, cache, scale, mask, sinks) -> str | None:
    """Why a call keeps the stock function (None: the replacement takes it)."""
    if sinks is not None:
        return "sinks"
    singleton_prefill = (
        gdn_prefill.split_enabled()
        and gdn_prefill.in_prefill()
        and queries.shape[-2] == 1
        and mask is None
    )
    if not singleton_prefill and not (isinstance(mask, str) and mask == "causal"):
        return f"mask:{type(mask).__name__}"
    if type(cache).__name__ not in ("KVCache", "BatchKVCache"):
        return f"cache:{type(cache).__name__}"
    if hasattr(cache, "bits"):
        return "quantized-cache"
    if keys.dtype != mx.bfloat16 or values.shape != keys.shape:
        return "kv-dtype-shape"
    if not eligible(tuple(queries.shape), tuple(keys.shape), queries.dtype, scale):
        return "shape"
    return None


def stats() -> dict[str, Any]:
    return {
        "dispatches": _STATE["dispatches"],
        "stock": _STATE["stock"],
        "long_skips": dict(_STATE["why"]),
    }


def attend(q: mx.array, k: mx.array, v: mx.array) -> mx.array:
    _, h, rows, _ = q.shape
    _, hk, keys, _ = k.shape
    fn = kernel(rows % BQ == 0, keys % BK == 0, h, hk)
    _STATE["dispatches"] += 1
    if _STATE["dispatches"] == 1:
        logger.info(
            "Prefill attention BQ%d engaged: rows=%d keys=%d heads=%d/%d",
            BQ,
            rows,
            keys,
            h,
            hk,
        )
    out = fn(
        inputs=[q, k, v],
        grid=((rows + BQ - 1) // BQ * 32, h * WM * WN, 1),
        threadgroup=(32, WM * WN, 1),
        output_shapes=[q.shape],
        output_dtypes=[mx.bfloat16],
    )
    return cast(mx.array, out[0])


def supported_device() -> bool:
    if importlib.metadata.version("mlx") != "0.32.3":
        return False
    info = mx.device_info()
    from .tensorfold.lane_qmm import _generation

    return bool(
        info.get("device_name") == "Apple M5 Max"
        and _generation(str(info.get("architecture", ""))) in (17, 18)
    )


def arithmetic_id() -> str:
    """Identifies the attention arithmetic. Output equals stock MLX, so this is informational only."""
    return (
        "attn-bq32-swz-" + hashlib.sha256(header(True, True).encode()).hexdigest()[:12]
    )


def enabled() -> bool:
    return bool(_STATE["enabled"])


def dispatches() -> int:
    return int(_STATE["dispatches"])


def enable() -> bool:
    """Turn the replacement on for the measured MLX/device pair and patch the mlx-vlm call (idempotent)."""
    _STATE["enabled"] = False
    _STATE["dispatches"] = 0
    if not supported_device():
        return False
    try:
        header(True, True)
        header(False, False)
    except (OSError, ValueError, RuntimeError):
        return False
    _install()
    _STATE["enabled"] = True
    return True


def disable() -> None:
    _STATE["enabled"] = False


def _install() -> None:
    if _STATE["installed"]:
        return
    from mlx_vlm.models.qwen3_5 import language as lang

    stock = lang.scaled_dot_product_attention

    def scaled_dot_product_attention(
        queries, keys, values, cache, scale, mask, sinks=None
    ):
        if _STATE["enabled"]:
            why = skip_reason(queries, keys, values, cache, scale, mask, sinks)
            if why is None:
                return attend(queries, keys, values)
            if queries.shape[-2] >= MIN_ROWS:
                _STATE["why"][why] = _STATE["why"].get(why, 0) + 1
        _STATE["stock"] += 1
        return stock(queries, keys, values, cache, scale, mask, sinks)

    lang.scaled_dot_product_attention = scaled_dot_product_attention
    _STATE["installed"] = True
