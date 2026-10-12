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

MAX_ROWS = 16
_STATE: dict[str, Any] = {"engaged": 0, "fallback": 0}


def counters() -> dict[str, int]:
    return dict(_STATE)


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
        plan = prework._qwen4_verify_plan(layer)
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
