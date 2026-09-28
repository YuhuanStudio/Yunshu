"""Batch-invariant decode for Qwen3.5-family targets (Splash-style lossless spec).

Every target matmul for decode (1 row) and speculative verify (2-8 rows) goes
through one kernel whose per-row result does not depend on how many rows share
the call: oMLX's ``sg8`` simdgroup-matrix quantized matmul, fed at least 4 rows
(zero-padded; measured row-invariant for 4..8 rows and for a padded single row,
docs/research/runs/2026-09-28-matrix/sg8-invariance.jsonl). Layers ``sg8`` does
not cover (tiny GDN ``in_proj_a/b``) use mlx-vlm's exact verify path, whose rows
already equal single-row decode. GatedDeltaNet and attention verify paths are the
bit-exact ones from ``kernels.omlx``.

Result: greedy output with speculative decoding on is token-identical to the
same engine with it off (the guarantee Splash calls lossless). It is *not*
bit-identical to stock MLX single-row GEMV decode; that drift is expected and
measured separately.

Only modules marked by :func:`mark_target` are rerouted, so a draft model keeps
its stock kernels. Prefill chunks (> 8 rows) keep stock kernels in both modes.
"""

from __future__ import annotations

import logging
from typing import Any

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)

MAX_ROWS = 8
MIN_ROWS = 4
_STATE: dict = {"installed": False, "active": True}


def set_active(active: bool) -> None:
    """Route the next forward passes through the invariant kernels or not.

    Invariance only matters for greedy requests (spec on must equal spec off);
    sampled requests never draft, so they take the faster stock kernels. Set
    per request on the single MLX thread, before its graphs are built.
    """
    _STATE["active"] = bool(active)


def is_installed() -> bool:
    return bool(_STATE["installed"])


def mark_target(language_model: Any) -> int:
    """Flag every QuantizedLinear of the target language model; return count."""
    n = 0
    for _, module in language_model.named_modules():
        if isinstance(module, nn.QuantizedLinear):
            module._yunshu_invariant = True
            n += 1
    return n


def _sg8():
    from .omlx import qwen35_verify_qmm

    return qwen35_verify_qmm


def invariant_linear(linear: Any, x: mx.array, exact_fallback) -> mx.array:
    """Row-invariant projection for a marked target layer, else ``exact_fallback``."""
    if (
        _STATE["active"]
        and getattr(linear, "_yunshu_invariant", False)
        and isinstance(linear, nn.QuantizedLinear)
        and x.ndim == 3
        and getattr(linear, "mode", "affine") == "affine"
    ):
        batch, length, k = x.shape
        rows = batch * length
        n = linear.scales.shape[0]
        q = _sg8()
        if rows <= MAX_ROWS and q.sg8_eligible(
            max(rows, MIN_ROWS), k, n, linear.bits, linear.group_size, x.dtype
        ):
            x2 = x.reshape(rows, k)
            if rows < MIN_ROWS:
                x2 = mx.concatenate([x2, mx.zeros((MIN_ROWS - rows, k), dtype=x.dtype)])
            y = q.vk_qmm_sg8(
                x2,
                linear.weight,
                linear.scales,
                linear.biases,
                group_size=linear.group_size,
                bits=linear.bits,
            )[:rows]
            if "bias" in linear:
                y = y + linear["bias"]
            return y.reshape(batch, length, n)
    return exact_fallback(linear, x)


def _install_packed(model: Any) -> int:
    """Repack eligible 4-bit target projections for the M5 tensor unit and pad
    single-row calls to 2 rows: the packed tensor-unit kernel is row-invariant
    for 2..8 rows (docs/.../nax-packed-invariance.jsonl) but 1 row takes a
    different matvec kernel."""
    from .omlx import pack_projections, qwen35_packed_linear

    packed = pack_projections(model)
    cls = qwen35_packed_linear.PackedLinear
    if packed and not getattr(cls, "_yunshu_invariant_pad", False):
        orig = cls.__call__

        def call(self, x):
            if _STATE["active"] and x.ndim == 3 and x.shape[0] * x.shape[1] == 1:
                pad = mx.concatenate([x, mx.zeros_like(x)], axis=1)
                return orig(self, pad)[:, :1]
            return orig(self, x)

        cls.__call__ = call
        cls._yunshu_invariant_pad = True
    return packed


def install(language_model: Any, model: Any = None, packed: bool = False) -> dict:
    """Route the target's decode + verify matmuls through invariant kernels.

    ``packed`` (needs the full ``model``) moves eligible 4-bit projections to
    oMLX's NAX packed kernel; remaining quantized projections use sg8.
    """
    n_packed = _install_packed(model) if packed and model is not None else 0
    import mlx_vlm.speculative.ops.linear as ops
    from mlx_vlm.models import quantized_verifier as qv
    from mlx_vlm.models.qwen3_5 import language as q35
    from mlx_vlm.models.qwen3_5 import speculative_verifier as sv

    marked = mark_target(language_model)
    if not _STATE["installed"]:
        stock_call = nn.QuantizedLinear.__call__

        def exact_rows(linear, x):
            # Layers sg8 cannot take: every row must equal stock single-row
            # decode. Never re-enter the patched ops functions from here.
            if not isinstance(linear, nn.QuantizedLinear):
                return linear(x)
            if x.ndim != 3 or x.shape[0] * x.shape[1] == 1:
                return stock_call(linear, x)
            out = qv.optimized_affine_linear(linear, x)
            if out is not None:
                return out
            rows = [
                stock_call(linear, x[b : b + 1, t : t + 1])
                for b in range(x.shape[0])
                for t in range(x.shape[1])
            ]
            return mx.concatenate(rows, axis=0).reshape(*x.shape[:2], -1)

        # Inactive (sampled or multi-row batches): the verify kernels that
        # were installed before us, not the slow exact per-row fallback.
        orig_linear = ops._target_verify_linear
        orig_linears = ops._target_verify_linears
        orig_quantized = ops._target_verify_quantized_linear

        def inv_linear(linear, x):
            if not _STATE["active"]:
                return orig_linear(linear, x)
            return invariant_linear(linear, x, exact_rows)

        def inv_linears(linears, x):
            if not _STATE["active"]:
                return orig_linears(linears, x)
            return tuple(inv_linear(linear, x) for linear in linears)

        def inv_quantized(linear, x):
            if not _STATE["active"]:
                return orig_quantized(linear, x)
            return inv_linear(linear, x)

        ops._target_verify_linear = inv_linear
        ops._target_verify_linears = inv_linears
        ops._target_verify_quantized_linear = inv_quantized
        for mod in (sv, q35):
            for name, fn in (
                ("_target_verify_linear", inv_linear),
                ("_target_verify_linears", inv_linears),
                ("_target_verify_quantized_linear", inv_quantized),
            ):
                if hasattr(mod, name):
                    setattr(mod, name, fn)

        def call(self, x):
            if (
                _STATE["active"]
                and getattr(self, "_yunshu_invariant", False)
                and x.ndim == 3
            ):
                if x.shape[0] * x.shape[1] <= MAX_ROWS:
                    return invariant_linear(self, x, exact_rows)
            return stock_call(self, x)

        nn.QuantizedLinear.__call__ = call

        verifier_cls = sv.Qwen3_5BatchInvariantForward
        orig_v = {
            name: getattr(verifier_cls, name)
            for name in ("_linear", "_linears", "quantized_linear", "quantized_argmax")
        }

        def v_linear(self, linear, x):
            if not _STATE["active"]:
                return orig_v["_linear"](self, linear, x)
            return inv_linear(linear, x)

        def v_linears(self, linears, x):
            if not _STATE["active"]:
                return orig_v["_linears"](self, linears, x)
            return inv_linears(linears, x)

        def v_quantized_linear(self, linear, x):
            if not _STATE["active"]:
                return orig_v["quantized_linear"](self, linear, x)
            return inv_linear(linear, x)

        def v_quantized_argmax(self, linear, x, *args, **kwargs):
            if not _STATE["active"]:
                return orig_v["quantized_argmax"](self, linear, x, *args, **kwargs)
            if kwargs.get("token_mask") is not None or (args and args[0] is not None):
                return None
            return mx.argmax(inv_linear(linear, x), axis=-1)

        verifier_cls._linear = v_linear
        verifier_cls._linears = v_linears
        verifier_cls.quantized_linear = v_quantized_linear
        verifier_cls.quantized_argmax = v_quantized_argmax
        _STATE["installed"] = True

    # Decode must take the same forward as verify: no fused greedy shortcut.
    language_model.fused_greedy_decode = None
    logger.info(
        "Batch-invariant decode: %d sg8 + %d packed target projections",
        marked,
        n_packed,
    )
    return {"marked": marked, "packed": n_packed}
