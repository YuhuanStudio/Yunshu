# Upstream (derived): ashhart/TensorFold (MIT) src/tensorfold/kernels/qwen/dense/v1/lane_qmm.py @ 34bae79a
"""Row-invariant quantized projections for the speculative lane (any row count).

TensorFold's lane matmul (``kernels.tensorfold.lane_qmm``, MIT) multiplies bf16
rows by MLX affine-quantized weights (2..8 bits) with hardware-selected cooperative tensor fragment layouts and a
per-group fma order fixed by the weight shape: a row's result has the same bits
whether it is alone or one of up to 128 rows in the call. ``LaneLinear`` holds
a projection in the kernel's layout (weight tiled 32 columns wide, group-major
(scale, bias) pairs — the same bytes as the checkpoint's weight, scales and
biases, so memory does not grow) and cuts calls wider than 128 rows into
128-row pieces, so *every* call is row-invariant: decode rows, speculative
verify rows and prefill chunks packed into one forward get the bits each row
would get alone.

``convert`` swaps a language model's eligible ``nn.QuantizedLinear`` layers —
every caller of those modules then uses it.
"""

from __future__ import annotations

import logging
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from . import gdn_prefill, nax_prefill
from .tensorfold import lane_qmm

logger = logging.getLogger(__name__)

PIECE = 512  # prefill dispatch only; other lane callers retain the 128-row guard
NARROW = 256  # outputs below this run the lane matmul in prefill too
# Calls above this many rows (a prefill chunk) run MLX's own quantized matmul on the weight
# untiled for the call: the lane kernel reads the whole weight once per 128 rows (16 reads per
# 2048-token chunk), which costs ~25% of a cold prefill. 0 (the default) keeps the lane kernel
# everywhere; the runner turns it on (``YUNSHU_PREFILL_MATMUL``).
STOCK_ROWS = 0
STOCK_MIN_ROWS = 512  # what the runner turns it on at (untiling the weight costs ~0.16 s a call; lane costs ~0.3 ms/token more)

_SUM_POLICY = False
_SUM_REUSE = False


def configure_sum_policy(qualified: bool) -> None:
    global _SUM_POLICY, _SUM_REUSE
    _SUM_POLICY = bool(qualified)
    _SUM_REUSE = False


def set_sum_context(ids) -> None:
    global _SUM_REUSE
    _SUM_REUSE = bool(_SUM_POLICY and ids is not None and len(ids) >= 32768)


def sum_reuse_enabled() -> bool:
    return _SUM_REUSE


def set_sum_reuse(enabled: bool) -> None:
    global _SUM_REUSE
    _SUM_REUSE = bool(enabled)


def set_stock_rows(rows: int) -> None:
    global STOCK_ROWS
    STOCK_ROWS = max(0, int(rows))


def prefill_kernel_id() -> str:
    """Names the prefill arithmetic; part of every APC key and SSD namespace, so states written
    by one prefill kernel are never read back as another's."""
    if gdn_prefill.split_enabled():
        return "split64-v1+" + nax_prefill.arithmetic_id()
    base = f"stock-qmm-gt{STOCK_ROWS}" if STOCK_ROWS else "lane-qmm"
    return (
        base + "+" + nax_prefill.arithmetic_id()
        if STOCK_ROWS and nax_prefill.enabled()
        else base
    )


def eligible(module: Any) -> bool:
    """An affine bias-free-or-biased QuantizedLinear the lane kernel reads."""
    if type(module) is not nn.QuantizedLinear:
        return False
    if getattr(module, "mode", "affine") != "affine":
        return False
    if module.get("biases") is None or module.scales.dtype != mx.bfloat16:
        return False
    bits, group = int(module.bits), int(module.group_size)
    if not lane_qmm.readable(bits, group) or not lane_qmm.ready():
        return False
    n = int(module.weight.shape[0])
    k = int(module.weight.shape[1]) * 32 // bits
    # columns not a multiple of NT stay untiled: the kernel slices the whole weight tensor
    tiled = n % lane_qmm.NT == 0
    return k % 64 == 0 and n % 4 == 0 and lane_qmm.slice_offset_ok(n, k, bits, tiled=tiled)


class LaneLinear(nn.Module):
    """A quantized projection whose every call runs the row-invariant lane
    matmul (in 128-row pieces above 128 rows)."""

    def __init__(self, weight, scales, biases, bits: int, group_size: int, bias=None):
        super().__init__()
        self.bits = int(bits)
        self.group_size = int(group_size)
        self.output_dims = int(weight.shape[0])
        self.input_dims = int(weight.shape[1]) * 32 // self.bits
        # 32-column tiles where the width allows (contiguous weight reads);
        # narrow projections (GDN in_proj_a/b) keep MLX's layout
        self.tiled = self.output_dims % lane_qmm.NT == 0
        self.weight = (
            lane_qmm.tile_weight(weight, lane_qmm.NT, self.group_size, bits=self.bits)
            if self.tiled
            else weight
        )
        self.sbt = lane_qmm.pack_scales(scales, biases)
        if bias is not None:
            self.bias = bias
        self.freeze()

    @classmethod
    def from_quantized(cls, linear: nn.QuantizedLinear) -> LaneLinear:
        return cls(
            linear.weight,
            linear.scales,
            linear.biases,
            linear.bits,
            linear.group_size,
            linear.get("bias"),
        )

    def quantized_rows(self, ids: mx.array):
        """MLX-layout (weight, scales, biases) of the rows ``ids`` (draft vocabulary)."""
        n, kg = self.output_dims, self.input_dims // self.group_size
        if self.tiled:
            nt = lane_qmm.NT
            w = self.weight.reshape(n // nt, kg, nt, -1)
            rows = w[ids // nt, :, ids % nt, :]
            rows = rows.reshape(int(ids.shape[0]), -1)
        else:
            rows = mx.take(self.weight, ids, axis=0)
        sb = mx.take(self.sbt, ids, axis=1)  # (KG, n_ids, 2)
        return rows, sb[..., 0].T, sb[..., 1].T

    def _extra_repr(self) -> str:
        return (
            f"input_dims={self.input_dims}, output_dims={self.output_dims}, "
            f"bits={self.bits}, group_size={self.group_size}, lane"
        )

    def _rows(
        self, x2: mx.array, *, prefill_narrow: bool = False, cache_input=None
    ) -> mx.array:
        # 17..48 rows: 16-row threadgroup blocks (same bits per row as the
        # default 32-row block, 10-35% faster on projections up to ~20K wide;
        # the 248K-wide LM head is slower that way)
        m = int(x2.shape[0])
        block = (
            32
            if prefill_narrow
            else (16 if 16 < m <= 48 and self.output_dims < 100_000 else None)
        )
        return lane_qmm.lane_matmul(
            cache_input if cache_input is not None else x2,
            self.weight,
            self.sbt,
            tiled=self.tiled,
            group=self.group_size,
            row_block=block,
            row_limit=PIECE,
            # a per-module split (the drafter's) is tuned for draft rows, not for a
            # context-window projection (2047 rows at a warm hit: +18 ms TTFT)
            sk=getattr(self, "sk", None) if m <= 32 else None,
            prefill_narrow=prefill_narrow,
        )

    def stock(self) -> tuple[mx.array, mx.array, mx.array]:
        """The projection in MLX's own layout (weight, scales, biases): the
        tiling undone, the pairs split. Fresh buffers, meant to live for one
        layer of a prefill forward (``prefill``)."""
        weight = (
            lane_qmm.untile_weight(
                self.weight, lane_qmm.NT, self.group_size, bits=self.bits
            )
            if self.tiled
            else self.weight
        )
        scales = mx.contiguous(self.sbt[..., 0].T)
        biases = mx.contiguous(self.sbt[..., 1].T)
        return weight, scales, biases

    def prefill(self, xs: list[mx.array]) -> list[mx.array]:
        """Each of ``xs`` through MLX's quantized matmul, one call per array.

        Prefill spans are large enough that the lane kernel (128 rows a call,
        a weight read per piece, a per-group epilogue) runs at ~75% of stock
        matmul speed. A stock call's bits depend on its own row count, so the
        caller passes each prompt span as its own array: a prompt's prefill
        never depends on what shares the step. The weight is untiled once for
        all spans (a transient copy, not a second resident one). Qualified
        short tails instead pad only the input rows for the native tiled-weight
        loader and crop its output, avoiding a transient stock weight copy.

        Narrow projections (GDN ``in_proj_a`` / ``in_proj_b``, a few dozen
        outputs) are not span-invariant in the stock matmul (its kernel choice
        follows the row count), and cost nothing: they take the lane path, so
        a prompt's bits do not depend on how it was cut into spans."""
        if self.output_dims < NARROW:
            return [self.lane_only(x) for x in xs]
        stock = None
        out = []
        for x in xs:
            lead = x.shape[:-1]
            x2 = x.reshape(-1, self.input_dims)
            dtype = x2.dtype
            if dtype != mx.bfloat16:
                x2 = x2.astype(mx.bfloat16)
            m = int(x2.shape[0])
            if (
                m % 512 == 0
                and m <= 4096
                and nax_prefill.eligible(
                    max(m, 640),  # the row bound is the engine's; atoms start at 512
                    self.input_dims,
                    self.output_dims,
                    self.bits,
                    self.group_size,
                    self.tiled,
                )
            ):
                # a run of full 512-row atoms: the NAX tile kernel computes each
                # row from its own row and the weight (one 128-row tile config
                # up to 4096 rows), so an atom's bits equal its run's
                y = nax_prefill.matmul(x2, self.weight, self.sbt, bits=self.bits)
                if "bias" in self:
                    y = y + self["bias"]
                out.append(y.reshape(*lead, self.output_dims).astype(dtype))
                continue
            if nax_prefill.tail_eligible(
                m,
                self.input_dims,
                self.output_dims,
                self.bits,
                self.group_size,
                self.tiled,
            ):
                y = nax_prefill.tail_matmul(x2, self.weight, self.sbt, bits=self.bits)
                if "bias" in self:
                    y = y + self["bias"]
                out.append(y.reshape(*lead, self.output_dims).astype(dtype))
                continue
            if stock is None:
                stock = self.stock()
            weight, scales, biases = stock
            y = mx.quantized_matmul(
                x2,
                weight,
                scales,
                biases,
                transpose=True,
                group_size=self.group_size,
                bits=self.bits,
            )
            if "bias" in self:
                y = y + self["bias"]
            out.append(y.reshape(*lead, self.output_dims).astype(dtype))
        return out

    def lane_only(self, x: mx.array) -> mx.array:
        """The lane kernel for any row count, in PIECE-row pieces, never the
        stock / NAX prefill dispatch of ``__call__`` (whose bits follow the
        call's row count, and which the engine turns on for the upstream path)."""
        lead = x.shape[:-1]
        x2 = x.reshape(-1, self.input_dims)
        dtype = x2.dtype
        if dtype != mx.bfloat16:
            x2 = x2.astype(mx.bfloat16)
        y = mx.concatenate(
            [self._rows(x2[i : i + PIECE]) for i in range(0, int(x2.shape[0]), PIECE)],
            axis=0,
        )
        if "bias" in self:
            y = y + self["bias"]
        return y.reshape(*lead, self.output_dims).astype(dtype)

    def __call__(self, x: mx.array) -> mx.array:
        lead = x.shape[:-1]
        x2 = x.reshape(-1, self.input_dims)
        dtype = x2.dtype
        if dtype != mx.bfloat16:
            x2 = x2.astype(mx.bfloat16)
        m = int(x2.shape[0])
        if gdn_prefill.split_enabled() and gdn_prefill.in_prefill():
            if self.output_dims < NARROW or self.output_dims >= 100_000:
                # Both have fixed per-row reductions. Preserve the measured
                # single-launch schedule for full-atom narrow gates.
                if nax_prefill.narrow_eligible(
                    m, self.input_dims, self.output_dims, self.bits, self.group_size
                ):
                    y = self._rows(x2, prefill_narrow=True)
                    if "bias" in self:
                        y = y + self["bias"]
                    return y.reshape(*lead, self.output_dims).astype(dtype)
                if m <= PIECE:
                    y = self._rows(
                        x2,
                        cache_input=x
                        if _SUM_REUSE and dtype == mx.bfloat16 and m <= 128
                        else None,
                    )
                    if "bias" in self:
                        y = y + self["bias"]
                    return y.reshape(*lead, self.output_dims).astype(dtype)
                return self.lane_only(x)
            if not nax_prefill.split_eligible(
                m,
                self.input_dims,
                self.output_dims,
                self.bits,
                self.group_size,
                self.tiled,
            ):
                raise ValueError("unqualified split-invariant prefill projection")
            # Full atoms keep their measured M128/M64 schedule; short or
            # unaligned pieces use the equivalent M64 row loader.
            if nax_prefill.eligible(
                m,
                self.input_dims,
                self.output_dims,
                self.bits,
                self.group_size,
                self.tiled,
            ):
                y = nax_prefill.matmul(x2, self.weight, self.sbt, bits=self.bits)
            else:
                y = nax_prefill.split_matmul(x2, self.weight, self.sbt, bits=self.bits)
        elif (
            STOCK_ROWS
            and m > STOCK_ROWS
            and nax_prefill.narrow_eligible(
                m, self.input_dims, self.output_dims, self.bits, self.group_size
            )
        ):
            nax_prefill.record_dispatch(m, self.input_dims, self.output_dims, self.bits)
            y = self._rows(x2, prefill_narrow=True)
        elif (
            STOCK_ROWS
            and m > STOCK_ROWS
            and nax_prefill.eligible(
                m,
                self.input_dims,
                self.output_dims,
                self.bits,
                self.group_size,
                self.tiled,
            )
        ):
            y = nax_prefill.matmul(x2, self.weight, self.sbt, bits=self.bits)
        elif (
            STOCK_ROWS
            and m > STOCK_ROWS
            and nax_prefill.tail_eligible(
                m,
                self.input_dims,
                self.output_dims,
                self.bits,
                self.group_size,
                self.tiled,
            )
        ):
            y = nax_prefill.tail_matmul(x2, self.weight, self.sbt, bits=self.bits)
        elif STOCK_ROWS and m > STOCK_ROWS and self.output_dims >= NARROW:
            weight, scales, biases = self.stock()
            y = mx.quantized_matmul(
                x2,
                weight,
                scales,
                biases,
                transpose=True,
                group_size=self.group_size,
                bits=self.bits,
            )
        elif m <= PIECE:
            y = self._rows(
                x2,
                cache_input=x
                if _SUM_REUSE and dtype == mx.bfloat16 and m <= 128
                else None,
            )
        else:
            y = mx.concatenate(
                [self._rows(x2[i : i + PIECE]) for i in range(0, m, PIECE)], axis=0
            )
        if "bias" in self:
            y = y + self["bias"]
        return y.reshape(*lead, self.output_dims).astype(dtype)


def convert(language_model: Any) -> dict:
    """Replace the eligible quantized projections of ``language_model`` (the
    decoder and ``lm_head``) with ``LaneLinear``; returns counts. Layers the
    kernel cannot read stay as they are (``skipped``)."""
    converted, skipped, pending = 0, [], []
    for name, module in list(language_model.named_modules()):
        children = [
            (key, child)
            for key, child in module.children().items()
            if isinstance(child, nn.QuantizedLinear)
        ]
        updates = {}
        for key, child in children:
            if eligible(child):
                lane = LaneLinear.from_quantized(child)
                updates[key] = lane
                pending += [lane.weight, lane.sbt]
                converted += 1
            else:
                skipped.append(f"{name}.{key}" if name else key)
        if updates:
            module.update_modules(updates)
        if sum(a.nbytes for a in pending) > 2 * 1024**3:
            mx.eval(pending)
            pending = []
    if pending:
        mx.eval(pending)
    mx.clear_cache()
    # Wide verify windows are row-invariant only when every projection is a lane one
    # (tree_verify.lane_projections); a skipped one keeps them at sg8's 8 rows.
    language_model._yunshu_lane_skipped = len(skipped)
    if skipped:
        logger.info("lane projections: %d skipped (%s)", len(skipped), skipped[:4])
    return {"converted": converted, "skipped": skipped}


__all__ = ["LaneLinear", "convert", "eligible"]
