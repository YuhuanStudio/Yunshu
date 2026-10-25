# Derived from Splash 1.4.0 runtime/ops/AneFfn.{hpp,cpp} and runtime/metal/abi/AneFfn.h (Apache-2.0).
# Licensed under the Apache License, Version 2.0; see THIRD_PARTY_NOTICES.md.
# Modified for Yunshu: Python port of the ANE program text, surfaces and constants.
"""MIL program of the Splash-style W8A8 FFN split: shapes, surfaces, rotation, functions.

CPU only (no MLX, no ANE). The ANE multiplies int8 weights and activations after a
block-diagonal Hadamard rotation v -> diag(sign) H v / sqrt(n) (gate/up inputs in blocks
of 128, the ANE's intermediate rows in blocks of 512); per-token and per-row int8 scales
carry the magnitudes and the GPU joins the partial down projection in fp32.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

SEGMENT = 2560
INPUT_BLOCK = 128
INTERMEDIATE_BLOCK = 512
ROTATION_UNIT = 128
ROTATE_THREADS = 256
ROTATE_BLOCKS = 8
WEIGHT_ROWS = 8
TILE = 32
TILE_ROWS = 8
INT8_PEAK = 127.0
INT8_UNIT = 128.0
PEAK_FLOOR = 2.0**-14
INTERMEDIATE_FLOOR = 2.0**-9
MINIMUM_ROWS = 512
PROGRAM_STEP = 128
MAXIMUM_ROWS = 2048  # Splash's prefill token budget
CHANNEL_UNIT = 512  # max(QUANT_TILE_ROWS, intermediate block); 512 divides 17408 = 34 units
SIGN_SEED = 20260930


def rotation_signs() -> np.ndarray:
    """Splash's std::mt19937(20260930) draws: odd -> -1."""
    rs = np.random.RandomState(SIGN_SEED)
    draws = rs.randint(0, 2**32, size=INTERMEDIATE_BLOCK, dtype=np.uint64)
    return np.where(draws & 1, -1.0, 1.0).astype(np.float32)


def hadamard(n: int) -> np.ndarray:
    """Sylvester Hadamard H[o, i] = (-1)^popcount(o & i)."""
    idx = np.arange(n)
    both = idx[:, None] & idx[None, :]
    pop = np.zeros_like(both)
    for bit in range(int(math.log2(n))):
        pop += (both >> bit) & 1
    return np.where(pop & 1, -1.0, 1.0).astype(np.float32)


def rotation_matrix(n: int, signs: np.ndarray) -> np.ndarray:
    """R = diag(sign[:n]) H / sqrt(n)."""
    return (signs[:n, None] * hadamard(n) / math.sqrt(n)).astype(np.float32)


def rotation_weight(channels: int, signs: np.ndarray) -> np.ndarray:
    """[channels, 512, 1, 1] weight of the grouped 1x1 convolution (per-channel row of R)."""
    block = INTERMEDIATE_BLOCK
    r = rotation_matrix(block, signs)
    reps = channels // block
    # channel c uses row (c % 512) of R: values[c, i] = sign[o] * H[o, i] * norm
    full = np.tile(r, (reps, 1))
    return full.astype(np.float16).reshape(channels, block, 1, 1)


def function_name(rows: int) -> str:
    return f"ffn{rows}"


def function_rows(maximum: int = MAXIMUM_ROWS) -> list[int]:
    return list(range(MINIMUM_ROWS, maximum + 1, PROGRAM_STEP))


def hexfloat(value: float) -> str:
    """C printf("%a") spelling that MIL's fp16() literal takes."""
    text = float(value).hex()
    mantissa, exponent = text.split("p")
    if "." in mantissa:
        mantissa = mantissa.rstrip("0").rstrip(".")
    return f"{mantissa}p{exponent}"


def fp16(value: float) -> str:
    return f"fp16({hexfloat(value)})"


def row_bytes(width: int, element: str) -> int:
    item = 1 if element == "int8" else 2
    return (width * item + 63) // 64 * 64


def surface_bytes(rows: int, width: int, element: str) -> int:
    page = 16384
    return (row_bytes(width, element) * rows + page - 1) // page * page


def segments(channels: int) -> list[int]:
    return [min(SEGMENT, channels - b) for b in range(0, channels, SEGMENT)]


@dataclass(frozen=True)
class Shape:
    hidden: int
    intermediate: int
    layers: int
    gpu: int
    ane: int
    down_segments: tuple[int, ...]

    @staticmethod
    def of(hidden: int, intermediate: int, layers: int, ane_units: int) -> Shape:
        if hidden % SEGMENT:
            raise ValueError("ANE FFN split needs a hidden size of whole 2560-channel segments")
        if intermediate % CHANNEL_UNIT or intermediate < 2 * CHANNEL_UNIT:
            raise ValueError("ANE FFN split needs two or more whole rotation blocks")
        if not ane_units or ane_units >= intermediate // CHANNEL_UNIT:
            raise ValueError("ANE FFN split leaves the GPU or the ANE no channels")
        ane = ane_units * CHANNEL_UNIT
        return Shape(hidden, intermediate, layers, intermediate - ane, ane, tuple(segments(ane)))


@dataclass
class Surf:
    """One pool surface: element 'fp16'|'int8', rows x width."""

    name: str
    element: str
    rows: int
    width: int
    index: int = -1

    @property
    def stride_elems(self) -> int:
        return row_bytes(self.width, self.element) // (1 if self.element == "int8" else 2)

    def strides(self, rows: int) -> str:
        plane = rows * self.stride_elems
        return f"[{plane}, {plane}, {self.stride_elems}, 1]"

    def buffer_type(self, rows: int, width: int) -> str:
        return (
            f"tensor_buffer<{self.element}, shape=[1, 1, {rows}, {width}], "
            f"strides={self.strides(rows)}, interleave_factors=[1, 1, 1, 1]>"
        )


@dataclass
class Memory:
    """Surfaces of the split (Splash AneFfn::allocate), in pool order."""

    shape: Shape
    surfaces: list[Surf] = field(default_factory=list)
    inputs: list[Surf] = field(default_factory=list)
    token_scale: Surf | None = None
    partial: Surf | None = None
    sets: list[dict] = field(default_factory=list)

    def add(self, name: str, element: str, rows: int, width: int) -> Surf:
        s = Surf(name, element, rows, width, len(self.surfaces))
        self.surfaces.append(s)
        return s


def allocate(shape: Shape, maximum_rows: int = MAXIMUM_ROWS) -> Memory:
    m = Memory(shape)
    segs = shape.hidden // SEGMENT
    for k in range(segs):
        m.inputs.append(m.add(f"x{k}", "int8", SEGMENT, maximum_rows))
    m.token_scale = m.add("tx", "fp16", 1, maximum_rows)
    m.partial = m.add("partial", "fp16", shape.hidden + 1, maximum_rows)
    for s in range(2):
        w = {"gate": [], "up": [], "down": []}
        for k in range(segs):
            w["gate"].append(m.add(f"wg{k}.{s}", "int8", shape.ane, SEGMENT))
            w["up"].append(m.add(f"wu{k}.{s}", "int8", shape.ane, SEGMENT))
        for i, width in enumerate(shape.down_segments):
            w["down"].append(m.add(f"wd{i}.{s}", "int8", shape.hidden, width))
        w["gate_scale"] = m.add(f"sg.{s}", "fp16", shape.ane, 1)
        w["up_scale"] = m.add(f"su.{s}", "fp16", shape.ane, 1)
        w["down_scale"] = m.add(f"sd.{s}", "fp16", shape.hidden, 1)
        m.sets.append(w)
    return m


@dataclass
class Input:
    name: str
    surfaces: tuple[Surf, Surf]
    rows: int
    width: int  # 0: the chunk's rows


def program_inputs(shape: Shape, m: Memory) -> list[Input]:
    both = lambda key: (m.sets[0][key], m.sets[1][key])  # noqa: E731
    out: list[Input] = []
    for k, s in enumerate(m.inputs):
        out.append(Input(f"x{k}", (s, s), SEGMENT, 0))
    out.append(Input("tx", (m.token_scale, m.token_scale), 1, 0))
    for k in range(len(m.inputs)):
        out.append(Input(f"wg{k}", (m.sets[0]["gate"][k], m.sets[1]["gate"][k]), shape.ane, SEGMENT))
        out.append(Input(f"wu{k}", (m.sets[0]["up"][k], m.sets[1]["up"][k]), shape.ane, SEGMENT))
    out.append(Input("sg", both("gate_scale"), shape.ane, 1))
    out.append(Input("su", both("up_scale"), shape.ane, 1))
    for i, width in enumerate(shape.down_segments):
        out.append(Input(f"wd{i}", (m.sets[0]["down"][i], m.sets[1]["down"][i]), shape.hidden, width))
    out.append(Input("sd", both("down_scale"), shape.hidden, 1))
    return out


def _tensor(dtype: str, rows: int, width: int) -> str:
    return f"tensor<{dtype}, [1, 1, {rows}, {width}]>"


def _dims(rows: int, width: int) -> str:
    return f"tensor<int32, [4]>([1, 1, {rows}, {width}])"


def function_text(shape: Shape, inputs: list[Input], output: Surf, rows: int) -> str:
    """The ANE's share of the FFN over `rows` rows (Splash AneFfn::function)."""
    input_segments = shape.hidden // SEGMENT
    channels = shape.ane
    unit = fp16(1.0 / INT8_UNIT)
    params: list[str] = []
    body: list[str] = []

    def line(text: str) -> None:
        body.append(f"        {text};")

    def f16(name: str, height: int, width: int, expr: str) -> None:
        line(f"{_tensor('fp16', height, width)} {name} = {expr}")

    for inp in inputs:
        width = inp.width if inp.width else rows
        params.append(f"{inp.surfaces[0].buffer_type(inp.rows, width)} {inp.name}")
        line(
            f"{_tensor(inp.surfaces[0].element, inp.rows, width)} {inp.name}_t = "
            f"tensor_buffer_to_tensor<ios17>(input = {inp.name})"
        )

    def matmul(name: str, weights: str, values: str, height: int, width: int) -> None:
        f16(f"{weights}_d", height, width, f"dequantize(input = {weights}_t, scale = {unit})")
        line(
            f"{_tensor('fp16', height, rows)} {name} = matmul(transpose_x = bool(false), "
            f"transpose_y = bool(false), x = {weights}_d, y = {values})"
        )

    def total(prefix: str, terms: int, height: int) -> str:
        acc = f"{prefix}0"
        for term in range(1, terms):
            nxt = f"{prefix}_sum{term}"
            f16(nxt, height, rows, f"add(x = {acc}, y = {prefix}{term})")
            acc = nxt
        return acc

    for k in range(input_segments):
        x = f"x{k}"
        f16(f"{x}_d", SEGMENT, rows, f"dequantize(input = {x}_t, scale = {unit})")
    for p in ("g", "u"):
        for k in range(input_segments):
            matmul(f"{p}m{k}", f"w{p}{k}", f"x{k}_d", channels, SEGMENT)
        f16(f"{p}s", channels, rows, f"mul(x = {total(p + 'm', input_segments, channels)}, y = s{p}_t)")
    c, r, block = channels, rows, INTERMEDIATE_BLOCK
    f16("gt", c, r, "mul(x = gs, y = tx_t)")
    f16("gh", c, r, f"mul(x = gt, y = {fp16(0.5)})")
    f16("th", c, r, "tanh(x = gh)")
    f16("tp", c, r, f"add(x = th, y = {fp16(1.0)})")
    f16("silu", c, r, "mul(x = gh, y = tp)")
    f16("h", c, r, "mul(x = silu, y = us)")
    line(f"tensor<fp16, [1, {c}, 1, {r}]> h4 = reshape(x = h, shape = tensor<int32, [4]>([1, {c}, 1, {r}]))")
    line(
        f"tensor<fp16, [{c}, {block}, 1, 1]> rotation = const()[name = string(\"rotation\"), "
        f"val = tensor<fp16, [{c}, {block}, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights.bin\"), "
        f"offset = uint64(64)))]"
    )
    line(
        f"tensor<fp16, [1, {c}, 1, {r}]> hr4 = conv(dilations = tensor<int32, [2]>([1, 1]), "
        f"groups = int32({c // block}), pad = tensor<int32, [4]>([0, 0, 0, 0]), "
        f"pad_type = string(\"valid\"), strides = tensor<int32, [2]>([1, 1]), weight = rotation, x = h4)"
    )
    f16("hr", c, r, f"reshape(x = hr4, shape = {_dims(c, r)})")
    f16("habs", c, r, "abs(x = hr)")
    f16("peak", 1, r, "reduce_max(x = habs, axes = tensor<int32, [1]>([2]), keep_dims = bool(true))")
    f16("floor", 1, r, f"maximum(x = peak, y = {fp16(INTERMEDIATE_FLOOR)})")
    f16("inverse", 1, r, f"real_div(x = {fp16(INT8_PEAK)}, y = floor)")
    f16("hs", c, r, "mul(x = hr, y = inverse)")
    line(
        f"{_tensor('int8', c, r)} hq = quantize(input = hs, scale = fp16(1), "
        'output_dtype = string("int8"))'
    )
    f16("hd", c, r, f"dequantize(input = hq, scale = {unit})")
    f16("hscale", 1, r, f"mul(x = floor, y = {fp16(INT8_UNIT / INT8_PEAK)})")
    begin = 0
    for i, width in enumerate(shape.down_segments):
        slice_name = f"hd{i}"
        f16(
            slice_name,
            width,
            r,
            f"slice_by_size(x = hd, begin = tensor<int32, [4]>([0, 0, {begin}, 0]), size = {_dims(width, r)})",
        )
        matmul(f"dm{i}", f"wd{i}", slice_name, shape.hidden, width)
        begin += width
    f16("ds", shape.hidden, r, f"mul(x = {total('dm', len(shape.down_segments), shape.hidden)}, y = sd_t)")
    f16("yt", shape.hidden + 1, r, "concat(axis = int32(2), interleave = bool(false), values = (ds, hscale))")
    line(
        f"{output.buffer_type(shape.hidden + 1, r)} y = tensor_to_tensor_buffer<ios17>(input = yt, "
        f"interleave_factors = tensor<uint8, [4]>([1, 1, 1, 1]), "
        f"strides = tensor<int64, [4]>({output.strides(shape.hidden + 1)}))"
    )
    return f"    func {function_name(rows)}<ios18>({', '.join(params)}) {{\n" + "\n".join(body) + "\n    } -> (y);\n"


def program_text(shape: Shape, inputs: list[Input], output: Surf, rows: list[int]) -> str:
    return "program(1.3)\n{\n" + "".join(function_text(shape, inputs, output, n) for n in rows) + "}\n"
