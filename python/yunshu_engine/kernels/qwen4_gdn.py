"""Qwen4 (Flash-Next) Gated DeltaNet layer on oMLX's fused verify plan and deferred rollback.

One call replaces the upstream layer's chain of per-op kernels (stacked in-projection, conv + SiLU, L2
q/k normalization, gates, delta-rule recurrence, gated norm, out-projection) by the oMLX plan: row-exact
unrolled projection tiles around one fused step kernel.  For a speculative window the per-step recurrent
states are not written (``record_deferred_states``): a partial accept recomputes only the kept prefix,
instead of every round writing S-1 FP32 state histories.  A single decode row takes the same plan, so a
window row and the serial row are computed by the same arithmetic.

Anything the plan does not admit (other geometry or bits, padded or batched cache, masks, fp32 inputs, more
than 16 rows) falls back to the caller's original layer.
2-bit packs (TensorFold 2bit) are such a case: the plan has no 2-bit tiles, so those layers keep the
composed upstream layer (identity-checked against it).
"""

from __future__ import annotations

import logging
from typing import Any

import mlx.core as mx

logger = logging.getLogger(__name__)

MAX_ROWS = 8
_STATE: dict[str, Any] = {"engaged": 0, "fallback": 0, "packed": 0}


def counters() -> dict[str, int]:
    return dict(_STATE)


def _split_in_rows(prework: Any, projections: Any) -> Any:
    """In-projection callable for layers whose four projections differ in bits or group size.

    Each projection runs its own row-exact unrolled tile (the arithmetic upstream's four separate one-row
    quantized matmuls use when they cannot be fused), and the outputs are concatenated into the layout the
    fused step kernel reads."""
    callables = []
    for linear in projections:
        call = prework.rows_qmv(
            linear.weight,
            linear.scales,
            linear.biases,
            linear.bits,
            linear.group_size,
            linear.mode,
            mx.bfloat16,
            prework._qwen4_verify_in_tile,
            unrolled=True,
        )
        if call is None:
            return None
        callables.append(call)

    def in_rows(x: Any) -> Any:
        return mx.concatenate([c(x) for c in callables], axis=-1)

    return in_rows


def _build_plan(module: Any) -> Any:
    from .omlx_q4 import qwen35_gdn_prework as prework

    plan = prework._build_qwen4_verify_plan(module)
    if plan is not None:
        return plan
    if not prework._qwen4_gdn_geometry_ok(module):
        return None
    projections = (
        module.in_proj_qkv,
        module.in_proj_z,
        module.in_proj_b,
        module.in_proj_a,
    )
    sizes = (2 * 16 * 128 + 48 * 128, 48 * 128, 48, 48)
    out = module.out_proj
    if not (
        all(
            prework._canonical_projection(lin, n, None)
            for lin, n in zip(projections, sizes, strict=True)
        )
        and prework._canonical_projection(out, 2560, None, in_dim=48 * 128)
    ):
        return None
    in_rows = _split_in_rows(prework, projections)
    out_rows = prework.rows_qmv(
        out.weight,
        out.scales,
        out.biases,
        out.bits,
        out.group_size,
        out.mode,
        mx.bfloat16,
        prework._qwen4_verify_out_tile,
        unrolled=True,
    )
    if in_rows is None or out_rows is None:
        return None
    return (
        in_rows,
        out_rows,
        module.conv1d.weight,
        mx.array(module.head_k_dim**-0.5, dtype=mx.bfloat16),
        module.A_log,
        module.dt_bias,
        module.norm.weight,
        mx.array(module.norm.eps, dtype=mx.float32),
        module.num_k_heads,
        module.num_v_heads,
        module.head_k_dim,
        module.head_v_dim,
    )


def plan_for(layer: Any) -> Any:
    """Cached verify plan of one GDN layer (uniform-format layers use oMLX's stacked plan)."""
    from .omlx_q4 import qwen35_gdn_prework as prework

    if not prework._is_qwen4_gdn(layer):
        return None
    return prework.cached_per_module(layer, "_yunshu_qwen4_gdn_plan", _build_plan)


def gated_delta(
    verifier: Any, layer: Any, inputs: Any, mask: Any, cache: Any, original: Any
) -> Any:
    from .omlx_q4 import qwen35_gdn_prework as prework

    length = int(inputs.shape[1]) if getattr(inputs, "ndim", 0) == 3 else 0
    if (
        mask is None
        and cache is not None
        and getattr(cache, "lengths", None) is None
        and 1 <= length <= MAX_ROWS
        and prework._qwen4_verify_state_eligible(inputs, cache)
    ):
        plan = plan_for(layer)
        if plan is not None:
            result = prework._qwen4_verify(plan, inputs, cache)
            if hasattr(cache, "advance"):
                cache.advance(length)
                verifier._helpers()._qwen3_5_advance_lengths_info(cache, length)
            _STATE["engaged"] += 1
            if _STATE["engaged"] == 1:
                logger.info("Qwen4 fused GDN verify engaged (%d rows)", length)
            return result
    _STATE["fallback"] += 1
    if mask is None:  # a masked verify is the composed layer by design
        from .. import fallbacks

        fallbacks.record(
            "qwen4_gdn_verify",
            lambda: (
                f"rows={length} cache={type(cache).__name__} lengths={getattr(cache, 'lengths', None) is not None}: composed layer"
            ),
        )
    if _STATE["fallback"] == 1:
        logger.info(
            "Qwen4 fused GDN verify: first fallback to the composed layer (%d rows, mask %s)",
            length,
            mask is not None,
        )
    return original(verifier, layer, inputs, mask, cache)


def _row_ok(inputs: Any, mask: Any, cache: Any) -> bool:
    """The window of one row may take the fused verify plan (``gated_delta``'s gate)."""
    from .omlx_q4 import qwen35_gdn_prework as prework

    length = int(inputs.shape[1]) if getattr(inputs, "ndim", 0) == 3 else 0
    return (
        mask is None
        and cache is not None
        and getattr(cache, "lengths", None) is None
        and 1 <= length <= MAX_ROWS
        and prework._qwen4_verify_state_eligible(inputs, cache)
    )


def gated_delta_rows(
    verifier: Any,
    layer: Any,
    mixed: Any,
    spans: list,
    masks: list,
    caches: list,
    original: Any,
) -> Any:
    """``gated_delta`` of several rows' windows packed in ``mixed`` [1, N, D] (row ``b`` at
    ``spans[b]``, on ``caches[b]`` with ``masks[b]``); returns the branches packed the same way.

    The in- and out-projections run once over every row (row-exact tiles: each row gets its one-row
    bits whatever the row count, so a row's result equals its own ``gated_delta`` call), so the layer's
    weights are read once per step instead of once per row; the fused step runs per row on its state.
    Rows the plan does not admit, or a pack wider than the projection tile, take ``gated_delta`` per row.
    """
    from .omlx_q4 import qwen35_gdn_prework as prework
    from .omlx_q4.row_exact_qmv import MAX_ROWS as QMV_ROWS

    windows = [mixed[:, s:e] for s, e in spans]
    plan = plan_for(layer) if len(spans) > 1 else None
    if (
        plan is None
        or int(mixed.shape[1]) > QMV_ROWS
        or not all(
            _row_ok(w, m, c) for w, m, c in zip(windows, masks, caches, strict=True)
        )
    ):
        parts = [
            gated_delta(verifier, layer, w, m, c, original)
            for w, m, c in zip(windows, masks, caches, strict=True)
        ]
        return parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=1)
    in_rows, out_rows = plan[0], plan[1]
    proj = in_rows(mixed)
    gated = []
    for (s, e), cache in zip(spans, caches, strict=True):
        gated.append(prework._qwen4_verify_step(plan, proj[:, s:e], cache))
        if hasattr(cache, "advance"):
            cache.advance(e - s)
            verifier._helpers()._qwen3_5_advance_lengths_info(cache, e - s)
    _STATE["engaged"] += len(spans)
    _STATE["packed"] += 1
    return out_rows(mx.concatenate(gated, axis=1))


def install_cache_patch() -> bool:
    """Teach ArraysCache transactions the ``deferred`` recurrent-state record (idempotent)."""
    from .omlx_q4 import qwen35_gdn_verify_fused

    return bool(qwen35_gdn_verify_fused.apply_arrays_cache_replay_patch())
