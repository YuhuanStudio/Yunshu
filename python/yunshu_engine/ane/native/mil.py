# # Upstream (Splash 1.4.0, Apache-2.0): runtime/ops/AneFfn.cpp
# Derived from Splash 1.4.0 runtime/ops/AneFfn.cpp (Apache-2.0).
# Modified for Yunshu: constant fp16/int8 weights, fp16 activations, h_exp scaling.
"""MIL text and blob generation; CPU-only. Channel-major IOSurface ABI."""

from __future__ import annotations

import struct

import numpy as np


def surface_type(height: int, width: int) -> tuple[str, str]:
    stride = (width * 2 + 63) // 64 * 32
    strides = f"[{height * stride}, {height * stride}, {stride}, 1]"
    return (
        f"tensor_buffer<fp16, shape=[1, 1, {height}, {width}], strides={strides}, "
        "interleave_factors=[1, 1, 1, 1]>",
        strides,
    )


class Blob:
    def __init__(self):
        self.data = bytearray(64)
        self.count = 0

    def add(self, value: np.ndarray) -> int:
        value = np.ascontiguousarray(value)
        self.data.extend(bytes((-len(self.data)) % 64))
        offset = len(self.data)
        # Storage format v2: 64-byte record followed by aligned tensor data.
        dtype = {np.dtype("float16"): 1, np.dtype("int8"): 4}[value.dtype]
        self.data.extend(
            struct.pack("<IIQQ", 0xDEADBEEF, dtype, value.nbytes, offset + 64)
        )
        self.data.extend(bytes(40))
        self.data.extend(value.tobytes())
        self.count += 1
        return offset

    def finish(self) -> bytes:
        struct.pack_into("<II", self.data, 0, self.count, 2)
        return bytes(self.data)


def mlp_program(gate, up, down, rows: int, mode: str = "fp16", h_exp: int = 6):
    """y = down(silu(gate(x))*up(x)); fp16 or symmetric per-output-channel W8.

    W8 is an additional weight quantization, not a bit-identical GPU substitute.
    Input/output surfaces have shapes [hidden, rows]; copies transpose outside run.
    """
    if mode not in {"fp16", "int8"} or rows <= 0 or not -15 <= h_exp <= 15:
        raise ValueError("invalid MIL mode/rows/h_exp")
    gate, up, down = (np.asarray(v, dtype=np.float16) for v in (gate, up, down))
    channels, hidden = gate.shape
    if up.shape != gate.shape or down.shape != (hidden, channels):
        raise ValueError("MLP weight shape mismatch")
    up = (up * np.float16(2.0**h_exp)).astype(np.float16)
    down = (down * np.float16(2.0**-h_exp)).astype(np.float16)
    if not all(np.isfinite(v).all() for v in (gate, up, down)):
        raise ValueError("nonfinite/scaled-overflow weights")
    blob = Blob()
    lines = []

    def tensor(dtype, h, w):
        return f"tensor<{dtype}, [1, 1, {h}, {w}]>"

    def emit(name, h, w, expr, dtype="fp16"):
        lines.append(f"        {tensor(dtype, h, w)} {name} = {expr};")
        return name

    def constant(name, value):
        h, w = value.shape
        dtype = "fp16" if value.dtype == np.float16 else "int8"
        offset = blob.add(value)
        return emit(
            name,
            h,
            w,
            f'const()[name = string("{name}"), val = {tensor(dtype, h, w)}'
            f'(BLOBFILE(path = string("@model_path/weights.bin"), offset = uint64({offset})))]',
            dtype,
        )

    def weight(name, value):
        if mode == "fp16":
            return constant(name, value)
        scale = (
            np.maximum(
                np.max(np.abs(value.astype(np.float32)), axis=1, keepdims=True) / 127,
                2.0**-24,
            )
        ).astype(np.float16)
        codes = np.clip(
            np.rint(value.astype(np.float32) / scale.astype(np.float32)), -127, 127
        ).astype(np.int8)
        constant(name + "q", codes)
        constant(name + "s", scale)
        # Dequantize with scalar 1 then multiply per-channel scales. ANE compiler
        # can select its quantized weight path; no activation quantization here.
        h, w = value.shape
        emit(name + "d", h, w, f"dequantize(input = {name}q, scale = fp16(1))")
        return emit(name, h, w, f"mul(x = {name}d, y = {name}s)")

    st, strides = surface_type(hidden, rows)
    emit("xt", hidden, rows, "tensor_buffer_to_tensor<ios17>(input = x)")
    for name, value in (("wg", gate), ("wu", up), ("wd", down)):
        weight(name, value)
    for name, w in (("g", "wg"), ("u", "wu")):
        emit(
            name,
            channels,
            rows,
            f"matmul(transpose_x = bool(false), transpose_y = bool(false), x = {w}, y = xt)",
        )
    emit("sig", channels, rows, "sigmoid(x = g)")
    emit("silu", channels, rows, "mul(x = g, y = sig)")
    emit("h", channels, rows, "mul(x = silu, y = u)")
    emit(
        "yt",
        hidden,
        rows,
        "matmul(transpose_x = bool(false), transpose_y = bool(false), x = wd, y = h)",
    )
    lines.append(
        f"        {st} y = tensor_to_tensor_buffer<ios17>(input = yt, "
        "interleave_factors = tensor<uint8, [4]>([1, 1, 1, 1]), "
        f"strides = tensor<int64, [4]>({strides}));"
    )
    mil = (
        "program(1.3)\n{\n"
        + f"    func main<ios18>({st} x) {{\n"
        + "\n".join(lines)
        + "\n    } -> (y);\n}\n"
    )
    return mil, blob.finish()
