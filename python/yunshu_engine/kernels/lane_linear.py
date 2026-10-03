# Upstream (derived): ashhart/TensorFold (MIT) src/tensorfold/kernels/qwen/dense/v1/lane_qmm.py @ 34bae79a
"""Row-invariant quantized projections for the round driver (any row count).

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

``convert`` swaps a language model's eligible ``nn.QuantizedLinear`` layers
(and, for tied embeddings, builds a lane head from the quantized embedding) —
the round driver's forward and every other caller of those modules then use it.
"""

from __future__ import annotations

import logging
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from . import nax_prefill
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


def set_stock_rows(rows: int) -> None:
    global STOCK_ROWS
    STOCK_ROWS = max(0, int(rows))


def prefill_kernel_id() -> str:
    """Names the prefill arithmetic; part of every APC key and SSD namespace, so states written
    by one prefill kernel are never read back as another's."""
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
    return k % 64 == 0 and n % 4 == 0


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

    def _rows(self, x2: mx.array, *, prefill_narrow: bool = False) -> mx.array:
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
            x2,
            self.weight,
            self.sbt,
            tiled=self.tiled,
            group=self.group_size,
            row_block=block,
            row_limit=PIECE,
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
        all spans (a transient copy, not a second resident one).

        Narrow projections (GDN ``in_proj_a`` / ``in_proj_b``, a few dozen
        outputs) are not span-invariant in the stock matmul (its kernel choice
        follows the row count), and cost nothing: they take the lane path, so
        a prompt's bits do not depend on how it was cut into spans."""
        if self.output_dims < NARROW:
            return [self(x) for x in xs]
        weight, scales, biases = self.stock()
        out = []
        for x in xs:
            lead = x.shape[:-1]
            x2 = x.reshape(-1, self.input_dims)
            dtype = x2.dtype
            if dtype != mx.bfloat16:
                x2 = x2.astype(mx.bfloat16)
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

    def __call__(self, x: mx.array) -> mx.array:
        lead = x.shape[:-1]
        x2 = x.reshape(-1, self.input_dims)
        dtype = x2.dtype
        if dtype != mx.bfloat16:
            x2 = x2.astype(mx.bfloat16)
        m = int(x2.shape[0])
        if (
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
            y = self._rows(x2)
        else:
            y = mx.concatenate(
                [self._rows(x2[i : i + PIECE]) for i in range(0, m, PIECE)], axis=0
            )
        if "bias" in self:
            y = y + self["bias"]
        return y.reshape(*lead, self.output_dims).astype(dtype)


def lane_head(embedding: Any) -> LaneLinear | None:
    """A lane projection with a quantized embedding's table (tied LM head)."""
    if not isinstance(embedding, nn.QuantizedEmbedding):
        return None
    if getattr(embedding, "mode", "affine") != "affine":
        return None
    probe = nn.QuantizedLinear.__new__(nn.QuantizedLinear)
    nn.Module.__init__(probe)
    probe.weight, probe.scales = embedding.weight, embedding.scales
    probe.biases = embedding.biases
    probe.bits, probe.group_size, probe.mode = (
        embedding.bits,
        embedding.group_size,
        "affine",
    )
    if not eligible(probe):
        return None
    return LaneLinear(
        embedding.weight,
        embedding.scales,
        embedding.biases,
        embedding.bits,
        embedding.group_size,
    )


def convert(language_model: Any) -> dict:
    """Replace the eligible quantized projections of ``language_model`` (the
    decoder and ``lm_head``) with ``LaneLinear``; returns counts. Layers the
    kernel cannot read stay as they are (``skipped``) — the round driver runs
    those one row at a time."""
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
    if skipped:
        logger.info("lane projections: %d skipped (%s)", len(skipped), skipped[:4])
    return {"converted": converted, "skipped": skipped}


__all__ = ["LaneLinear", "convert", "eligible", "lane_head"]
