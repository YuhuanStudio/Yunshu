"""Qwen4 (Flash-Next) Gated DeltaNet layer on oMLX's fused verify plan and deferred rollback.

One call replaces the upstream layer's chain of per-op kernels (stacked in-projection, conv + SiLU, L2
q/k normalization, gates, delta-rule recurrence, gated norm, out-projection) by the oMLX plan: row-exact
unrolled projection tiles around one fused step kernel.  For a speculative window the per-step recurrent
states are not written (``record_deferred_states``): a partial accept recomputes only the kept prefix,
instead of every round writing S-1 FP32 state histories.  A single decode row takes the same plan, so a
window row and the serial row are computed by the same arithmetic.

Anything the plan does not admit (other geometry or bits, padded or batched cache, masks, fp32 inputs, more
than 16 rows) falls back to the caller's original layer.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx

MAX_ROWS = 8
_STATE: dict[str, Any] = {"engaged": 0, "fallback": 0}


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
            return result
    _STATE["fallback"] += 1
    return original(verifier, layer, inputs, mask, cache)


def install_cache_patch() -> bool:
    """Teach ArraysCache transactions the ``deferred`` recurrent-state record (idempotent)."""
    from .omlx_q4 import qwen35_gdn_verify_fused

    return bool(qwen35_gdn_verify_fused.apply_arrays_cache_replay_patch())
