# Derived from Splash 1.4.0 runtime/ops/AneFfn.{hpp,cpp} (Apache-2.0).
# Licensed under the Apache License, Version 2.0; see THIRD_PARTY_NOTICES.md.
# Modified for Yunshu: Python/MLX orchestration over the native ANE bridge.
"""Splash-style prefill FFN split: the GPU runs the leading intermediate channels, the ANE the rest.

Per layer: rotate+quantize the normalized rows on the GPU into IOSurface-shared int8 segments,
signal the ANE through a shared event inside the GPU command stream, run the GPU's share, stage
the next layer's int8 weights (rotated, requantized from the quantized planes) into the other
double-buffer set, wait on the event and join the ANE's partial down projection in fp32. Any ANE
failure retires the split for the life of the process; the caller reruns the chunk on the GPU.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from yunshu_engine.ane import ffn_calibration as cal
from yunshu_engine.ane import ffn_kernels as K
from yunshu_engine.ane import ffn_program as P
from yunshu_engine.ane.native import PoolSession
from yunshu_engine.ane.native.mil import Blob

log = logging.getLogger(__name__)

MAXIMUM_ERROR = 0.10


class AneUnavailable(RuntimeError):
    """The split cannot run here (private API, compile, memory); the GPU runs alone."""


class AneRetired(RuntimeError):
    """The ANE's work failed or was too slow: discard the chunk and rerun it on the GPU."""


@dataclass
class QProj:
    w: object
    s: object
    b: object
    bits: int
    gs: int
    out: int
    inp: int


@dataclass
class Layer:
    gate: QProj
    up: QProj
    down: QProj


def layer_from_module(mlp) -> Layer:
    def proj(m) -> QProj:
        out, packed = m.weight.shape
        return QProj(m.weight, m.scales, m.biases, m.bits, m.group_size, out, packed * 32 // m.bits)

    return Layer(proj(mlp.gate_proj), proj(mlp.up_proj), proj(mlp.down_proj))


def _qmm(x, p: QProj, w=None, s=None, b=None):
    import mlx.core as mx

    return mx.quantized_matmul(
        x,
        p.w if w is None else w,
        p.s if s is None else s,
        p.b if b is None else b,
        transpose=True,
        group_size=p.gs,
        bits=p.bits,
    )


def unsupported(layers: Sequence[Layer]) -> str | None:
    if not layers:
        return "ANE FFN split has no layers"
    hidden, inter = layers[0].gate.inp, layers[0].gate.out
    if hidden % P.SEGMENT:
        return "ANE FFN split needs a hidden size of whole 2560-channel segments"
    if hidden % (P.INPUT_BLOCK * 8) or hidden // (P.INPUT_BLOCK * 8) > P.ROTATE_BLOCKS:
        return "ANE FFN split needs a hidden size the rotate kernel takes"
    if inter % P.CHANNEL_UNIT or inter < 2 * P.CHANNEL_UNIT:
        return "ANE FFN split needs two or more whole rotation blocks of intermediate channels"
    for lay in layers:
        for p in (lay.gate, lay.up, lay.down):
            if p.bits not in (2, 3, 4, 5, 6, 8):
                return f"ANE FFN split does not take {p.bits}-bit weights"
        if (lay.gate.out, lay.gate.inp, lay.up.out, lay.up.inp, lay.down.out, lay.down.inp) != (
            inter,
            hidden,
            inter,
            hidden,
            hidden,
            inter,
        ):
            return "ANE FFN split layers differ in shape"
    return None


def units(layers: Sequence[Layer]) -> int:
    reason = unsupported(layers)
    if reason:
        raise ValueError(reason)
    return layers[0].gate.out // P.CHANNEL_UNIT


def planned_bytes(layers: Sequence[Layer], ane_units: int) -> int:
    """Surface memory of the split (the GPU's down-projection copy is counted separately)."""
    shape = P.Shape.of(layers[0].gate.inp, layers[0].gate.out, len(layers), ane_units)
    mem = P.allocate(shape)
    return sum(P.surface_bytes(s.rows, s.width, s.element) for s in mem.surfaces)


class AneFfn:
    def __init__(
        self,
        layers: Sequence[Layer],
        ane_units: int,
        cache: Path,
        *,
        function_rows: Sequence[int] | None = None,
    ):
        import mlx.core as mx

        self._mx = mx
        reason = unsupported(layers)
        if reason:
            raise AneUnavailable(reason)
        self.layers = list(layers)
        hidden, inter = layers[0].gate.inp, layers[0].gate.out
        self.shape = P.Shape.of(hidden, inter, len(layers), ane_units)
        self.rows = sorted(function_rows) if function_rows else P.function_rows()
        self.maximum_rows = max(self.rows)
        self.minimum_rows = self.rows[0]
        self.breaker = cal.Breaker()
        self.lost = None
        self.served_commands = 0
        self.served_evaluations = 0
        self.served_ms = 0.0
        self._tickets: list = []
        self._statuses: list = []
        self._next_layer = 0
        self._committed_evals = 0
        self._committed_rows = 0
        self._unfinished = False
        self._warned = False
        self.memory = P.allocate(self.shape, self.maximum_rows)
        self.inputs = P.program_inputs(self.shape, self.memory)
        signs = P.rotation_signs()
        self.signs = mx.array(signs)
        blob = Blob()
        offset = blob.add(P.rotation_weight(self.shape.ane, signs))
        if offset != 64:
            raise AssertionError("rotation constant must sit at the first blob record")
        mil = P.program_text(self.shape, self.inputs, self.memory.partial, self.rows)
        specs = [(s.element, s.rows, s.width) for s in self.memory.surfaces]
        pool, why = PoolSession.try_create(
            mil, blob.finish(), specs, cache, mlx_root=Path(mx.__file__).parent
        )
        if pool is None:
            raise AneUnavailable(why or "native ANE unavailable")
        self.pool = pool
        # Bindings: function x weight set.
        self.bindings: dict[tuple[int, int], int] = {}
        for rows in self.rows:
            fname = P.function_name(rows)
            names = pool.inputs(fname)
            by_set = [{i.name: i.surfaces[s] for i in self.inputs} for s in range(2)]
            for s in range(2):
                idx = [by_set[s][n].index for n in names]
                self.bindings[(rows, s)] = pool.bind(fname, idx, self.memory.partial.index)
        self._alias: dict[int, object] = {}
        # GPU slices (contiguous) and per-row int8 scales, computed once.
        self._gpu: list[dict] = []
        self._row_scales: list[dict] = []
        for lay in self.layers:
            g = self.shape.gpu
            gate = (lay.gate.w[:g], lay.gate.s[:g], lay.gate.b[:g])
            up = (lay.up.w[:g], lay.up.s[:g], lay.up.b[:g])
            d = lay.down
            words = g * d.bits // 32
            down = (
                mx.contiguous(d.w[:, :words]),
                mx.contiguous(d.s[:, : g // d.gs]),
                mx.contiguous(d.b[:, : g // d.gs]),
            )
            mx.eval(*down)
            self._gpu.append({"gate": gate, "up": up, "down": down})
        scales = []
        for lay in self.layers:
            entry = {
                "gate": self._row_scale(lay.gate, self.shape.gpu, 0, hidden, self.shape.ane, 1),
                "up": self._row_scale(lay.up, self.shape.gpu, 0, hidden, self.shape.ane, 1),
                "down": self._row_scale(lay.down, 0, self.shape.gpu, self.shape.ane, hidden, 4),
            }
            scales.append(entry)
            mx.eval(*entry.values())
        self._row_scales = scales
        # Weights of layer 0 are staged by the first command; set 0 first.

    # ---- surface aliases -------------------------------------------------------------
    def alias(self, surf: P.Surf):
        a = self._alias.get(surf.index)
        if a is None:
            a = self.pool.alias(surf.index)
            self._alias[surf.index] = a
        return a

    # ---- kernels ---------------------------------------------------------------------
    def _row_scale(self, p: QProj, row0: int, in0: int, width: int, rows: int, k: int):
        mx = self._mx
        (out,) = K.kernel("row_scale")(
            inputs=[p.w, p.s, p.b, self.signs],
            template=[
                ("K", k),
                ("KIN", p.inp),
                ("BITS", p.bits),
                ("GS", p.gs),
                ("ROW0", row0),
                ("IN0", in0),
                ("WIDTH", width),
            ],
            grid=(rows // P.WEIGHT_ROWS * 256, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(rows,)],
            output_dtypes=[mx.float16],
        )
        return out

    def _stage(self, layer: int, s: int) -> list:
        """Kernels staging layer `layer`'s int8 weights into set `s`; returns markers to evaluate."""
        mx = self._mx
        lay, shape, mem = self.layers[layer], self.shape, self.memory
        w, sc = mem.sets[s], self._row_scales[layer]
        markers = []

        def run(proj, kind, surf, scale_surf, row0, in0, width, rows, k):
            (m,) = K.kernel("weights")(
                inputs=[
                    proj.w,
                    proj.s,
                    proj.b,
                    sc[kind],
                    self.signs,
                    self.alias(surf),
                    self.alias(scale_surf),
                ],
                template=[
                    ("K", k),
                    ("KIN", proj.inp),
                    ("BITS", proj.bits),
                    ("GS", proj.gs),
                    ("ROW0", row0),
                    ("IN0", in0),
                    ("STRIDE", surf.stride_elems),
                    ("SSTRIDE", scale_surf.stride_elems),
                ],
                grid=(rows // P.WEIGHT_ROWS * 256, width // (k * P.ROTATION_UNIT), 1),
                threadgroup=(256, 1, 1),
                output_shapes=[(1,)],
                output_dtypes=[mx.uint32],
            )
            markers.append(m)

        for k in range(len(w["gate"])):
            run(lay.gate, "gate", w["gate"][k], w["gate_scale"], shape.gpu, k * P.SEGMENT, P.SEGMENT, shape.ane, 1)
            run(lay.up, "up", w["up"][k], w["up_scale"], shape.gpu, k * P.SEGMENT, P.SEGMENT, shape.ane, 1)
        begin = shape.gpu
        for i, width in enumerate(shape.down_segments):
            run(lay.down, "down", w["down"][i], w["down_scale"], 0, begin, width, shape.hidden, 4)
            begin += width
        return markers

    def _pack(self, normalized, rows: int) -> list:
        mx = self._mx
        shape, mem = self.shape, self.memory
        (rotated,) = K.kernel("rotate")(
            inputs=[normalized, self.signs, self.alias(mem.token_scale)],
            template=[("H", shape.hidden)],
            grid=(P.ROTATE_THREADS * rows, 1, 1),
            threadgroup=(P.ROTATE_THREADS, 1, 1),
            output_shapes=[(rows, shape.hidden)],
            output_dtypes=[mx.float16],
        )
        tiles = (rows + P.TILE - 1) // P.TILE
        markers = []
        for k, surf in enumerate(mem.inputs):
            (m,) = K.kernel("pack")(
                inputs=[rotated, self.alias(surf)],
                template=[
                    ("H", shape.hidden),
                    ("R", rows),
                    ("CHANNEL0", k * P.SEGMENT),
                    ("STRIDE", surf.stride_elems),
                ],
                grid=(tiles * P.TILE, (P.SEGMENT // P.TILE) * P.TILE_ROWS, 1),
                threadgroup=(P.TILE, P.TILE_ROWS, 1),
                output_shapes=[(1,)],
                output_dtypes=[mx.uint32],
            )
            markers.append(m)
        return markers

    def _join(self, gpu_out, rows: int):
        mx = self._mx
        shape, mem = self.shape, self.memory
        tiles = (rows + P.TILE - 1) // P.TILE
        out, status = K.kernel("join")(
            inputs=[gpu_out, self.alias(mem.partial), self.alias(mem.token_scale)],
            template=[("H", shape.hidden), ("R", rows), ("PSTRIDE", mem.partial.stride_elems)],
            grid=(tiles * P.TILE, (shape.hidden // P.TILE) * P.TILE_ROWS, 1),
            threadgroup=(P.TILE, P.TILE_ROWS, 1),
            output_shapes=[(rows, shape.hidden), (tiles * (shape.hidden // P.TILE),)],
            output_dtypes=[mx.bfloat16, mx.uint32],
        )
        return out, status

    # ---- the split -------------------------------------------------------------------
    @property
    def share(self) -> float:
        return self.shape.ane / self.shape.intermediate

    def retired(self) -> bool:
        return self.pool.retired

    @property
    def reason(self) -> str:
        return self.pool.reason

    def splits(self, rows: int) -> bool:
        return self.minimum_rows <= rows <= self.maximum_rows and not self.retired()

    def set_minimum_rows(self, rows: int) -> None:
        if rows < self.rows[0] or rows > self.maximum_rows:
            raise ValueError(f"ANE FFN split has no function of {rows} rows")
        self.minimum_rows = rows

    def _function_rows(self, rows: int) -> int:
        for n in self.rows:
            if n >= rows:
                return n
        raise ValueError("ANE FFN has no function of the chunk's rows")

    def begin(self) -> bool:
        """Starts a command (a chunk's layers 0..n-1); False once the split stopped."""
        if self._unfinished:
            self.pool.retire("a command it split did not finish")
            self._warn()
        self._unfinished = False
        self._tickets, self._statuses = [], []
        self._next_layer = 0
        self._committed_evals = 0
        return not self.retired()

    def add(self, layer: int, normalized, residual, rows: int | None = None, *, ane: bool = True):
        """Layer `layer`'s FFN of the chunk: residual + FFN(normalized), lazily; None -> GPU alone."""
        mx = self._mx
        rows = normalized.shape[0] if rows is None else rows
        if layer != self._next_layer or layer >= len(self.layers):
            raise RuntimeError(f"ANE FFN split adds layer {layer}, not its next layer {self._next_layer}")
        if not self.splits(rows):
            raise RuntimeError("ANE FFN split does not take this chunk")
        if self.retired():
            raise AneRetired(self.reason)
        fn_rows = self._function_rows(rows)
        s = layer & 1
        pre = []
        if layer == 0:
            pre = self._stage(0, 0)
        pack = self._pack(normalized, rows)
        mx.async_eval(*pack, *pre)
        ticket = None
        if ane:
            ticket, why = self.pool.begin(self.bindings[(fn_rows, s)])
            if ticket is None:
                self.pool.retire(why or "ANE submission failed")
                self._warn()
                raise AneRetired(self.reason)
            self._tickets.append(ticket)
        g = self._gpu[layer]
        lay = self.layers[layer]
        gate = _qmm(normalized, lay.gate, *g["gate"])
        up = _qmm(normalized, lay.up, *g["up"])
        h = (gate * mx.sigmoid(gate)) * up
        y = _qmm(h, lay.down, *g["down"])
        part = y if residual is None else residual + y
        nxt = self._stage(layer + 1, s ^ 1) if layer + 1 < len(self.layers) else []
        mx.async_eval(part, *nxt)
        if ticket is not None and not ticket.wait_mlx():
            self.pool.retire(ticket.reason or "ANE wait failed")
            self._warn()
            raise AneRetired(self.reason)
        out, status = self._join(part, rows)
        if ane:
            self._statuses.append(status)
            self._committed_evals += 1
            self._committed_rows = fn_rows
            self._unfinished = True
        self._next_layer += 1
        return out

    def finish(self) -> bool:
        """After the chunk's outputs are evaluated: True when every ANE job succeeded and every
        join read finite values; False retires the split (the caller reruns the chunk on the GPU)."""
        mx = self._mx
        ok = True
        for t in self._tickets:
            ok = t.finish() and ok
        ran = self.pool.finish() if ok else None
        if ran is None:
            ok = False
        if ok and self._statuses:
            bad = int(mx.max(mx.concatenate(self._statuses)).item())
            if bad:
                self.pool.retire("the Neural Engine's output or its scales were not finite")
                ok = False
        for t in self._tickets:
            t.close()
        self._tickets, self._statuses = [], []
        self._unfinished = False
        if not ok:
            self._warn()
            return False
        if self._committed_evals:
            self.served_commands += 1
            self.served_evaluations += self._committed_evals
            self.served_ms += ran
            losing = self.breaker.add(self._committed_rows, self._committed_evals, ran)
            if losing:
                self.pool.retire(losing)
                self._warn()
                if self.lost:
                    cb, self.lost = self.lost, None
                    cb()
        return True

    def _warn(self) -> None:
        if not self._warned:
            self._warned = True
            log.warning(
                "Neural Engine FFN split stopped (%s); the GPU runs the prefill FFN alone "
                "until the engine restarts.",
                self.reason,
            )

    # ---- checks ----------------------------------------------------------------------
    @staticmethod
    def after(x, previous):
        """`x` ordered after `previous` (a layer's output): a model's normalized rows already
        depend on the previous layer; synthetic inputs must, or layer l+1's pack may overwrite the
        shared surfaces before layer l's join has read them."""
        return x + previous[:1, :1] * 0

    def gpu_alone(self, layer: int, normalized, residual):
        mx = self._mx
        lay = self.layers[layer]
        gate, up = _qmm(normalized, lay.gate), _qmm(normalized, lay.up)
        y = _qmm((gate * mx.sigmoid(gate)) * up, lay.down)
        return y if residual is None else residual + y

    def verify(self, seed: int = 20261001) -> float:
        """Splash AneFfn::verify: each function at its own rows over the first two layers, the
        ANE's contribution against the GPU alone's. Returns the worst relative RMS error."""
        mx = self._mx
        count = min(2, len(self.layers))
        rng = np.random.default_rng(seed)
        worst = 0.0
        for rows in self.rows:
            x = mx.array(rng.uniform(-2.0, 2.0, (rows, self.shape.hidden)).astype(np.float32)).astype(
                mx.bfloat16
            )
            h_alone = h_split = h_part = mx.zeros((rows, self.shape.hidden), dtype=mx.bfloat16)
            self.begin()
            for layer in range(count):
                h_alone = self.gpu_alone(layer, x, h_alone)
                lay, g = self.layers[layer], self._gpu[layer]
                gate, up = _qmm(x, lay.gate, *g["gate"]), _qmm(x, lay.up, *g["up"])
                h_part = h_part + _qmm((gate * mx.sigmoid(gate)) * up, lay.down, *g["down"])
                h_split = self.add(layer, self.after(x, h_split) if layer else x, h_split, rows)
            mx.eval(h_alone, h_split, h_part)
            if not self.finish():
                raise AneRetired(self.reason)
            a, sp, pt = (np.array(t.astype(mx.float32)) for t in (h_alone, h_split, h_part))
            if not np.isfinite(sp).all():
                raise AneRetired("output is not finite")
            err = float(np.sqrt(((sp - a) ** 2).sum() / ((a - pt) ** 2).sum()))
            if not err <= MAXIMUM_ERROR:
                self.pool.retire(f"its output of {rows} rows differs by {100 * err:.2f}% of the ANE's part")
                raise AneRetired(self.reason)
            worst = max(worst, err)
        return worst

    def close(self) -> None:
        self._alias.clear()
        self.pool.close()
