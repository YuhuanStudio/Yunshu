"""Core ML converter for slices of Qwen-style dense FFN (SwiGLU MLP) layers from MLX affine weights.

Shared by the ANE-split prefill (anesplit) and the ANE-prefill-beside-decode work (aneconc).
Only numpy (and coremltools, imported lazily) are needed, so it runs in an isolated venv with no MLX.

MLX affine layout: ``weight`` is uint32 with ``32 // bits`` values per word (bits 2/4/8) or, for 3/5/6 bits,
8 values packed in 3/5/6 bytes, little endian; ``scales``/``biases`` have one entry per ``group_size`` inputs
along the input axis and ``w = q * scale + bias``.
"""

from __future__ import annotations

import numpy as np


def unpack_affine(weight: np.ndarray, bits: int) -> np.ndarray:
    """Unpack MLX-packed uint32 ``weight`` [out, in*bits/32] to uint8 integer codes [out, in]."""
    out_dim = weight.shape[0]
    raw = (
        np.ascontiguousarray(weight).view(np.uint8).reshape(out_dim, -1)
    )  # little endian bytes
    if bits in (2, 4, 8):
        per = 8 // bits
        shifts = (np.arange(per, dtype=np.uint8) * bits)[None, None, :]
        codes = (raw[:, :, None] >> shifts) & ((1 << bits) - 1)
        return codes.reshape(out_dim, -1).astype(np.uint8)
    if bits in (3, 5, 6):
        # 8 values in `bits` bytes
        grp = raw.reshape(out_dim, -1, bits).astype(np.uint64)
        word = np.zeros(grp.shape[:2], dtype=np.uint64)
        for i in range(bits):
            word |= grp[:, :, i] << np.uint64(8 * i)
        shifts = (np.arange(8, dtype=np.uint64) * bits)[None, None, :]
        codes = (word[:, :, None] >> shifts) & np.uint64((1 << bits) - 1)
        return codes.reshape(out_dim, -1).astype(np.uint8)
    raise ValueError(f"unsupported bits {bits}")


def dequantize_affine(
    weight: np.ndarray,
    scales: np.ndarray,
    biases: np.ndarray,
    bits: int,
    group_size: int = 64,
) -> np.ndarray:
    """The affine weights as float32 [out, in]: ``q * scale + bias`` per group of inputs."""
    codes = unpack_affine(weight, bits).astype(np.float32)
    out_dim, in_dim = codes.shape
    g = codes.reshape(out_dim, in_dim // group_size, group_size)
    s = scales.astype(np.float32)[:, :, None]
    b = biases.astype(np.float32)[:, :, None]
    return (g * s + b).reshape(out_dim, in_dim)


def split_channels(total: int, share: float, multiple: int = 64) -> int:
    """ANE-side channel count for a ``share`` of ``total`` intermediate channels, a multiple of ``multiple``."""
    n = int(round(total * share / multiple)) * multiple
    return max(multiple, min(total - multiple, n))


def mlp_slice(
    gate: np.ndarray, up: np.ndarray, down: np.ndarray, lo: int, hi: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dense fp weights of intermediate channels [lo, hi): gate/up rows, down columns."""
    return gate[lo:hi], up[lo:hi], down[:, lo:hi]


def affine_slice(weight, scales, biases, bits, group_size, rows=None, cols=None):
    """Codes [out, in] (uint8) and fp16 per-block LUT [out, in/gs, 2**nbits, 1] of an affine projection,
    restricted to output rows ``rows=(lo, hi)`` or input columns ``cols=(lo, hi)`` (multiples of group_size).

    nbits is the palette width Core ML takes: 4 for 4-bit, 6 for 5/6-bit weights, 8 for 8-bit. The LUT entries
    are ``code * scale + bias``, i.e. exactly the dequantized values (rounded to fp16)."""
    codes = unpack_affine(weight, bits)
    sc, bi = scales.astype(np.float32), biases.astype(np.float32)
    if rows is not None:
        lo, hi = rows
        codes, sc, bi = codes[lo:hi], sc[lo:hi], bi[lo:hi]
    if cols is not None:
        lo, hi = cols
        a, b = lo // group_size, hi // group_size
        codes, sc, bi = codes[:, lo:hi], sc[:, a:b], bi[:, a:b]
    nbits = {2: 2, 3: 3, 4: 4, 5: 6, 6: 6, 8: 8}[bits]
    k = np.arange(1 << nbits, dtype=np.float32)
    lut = (sc[:, :, None] * k[None, None, :] + bi[:, :, None])[..., None].astype(
        np.float16
    )
    return np.ascontiguousarray(codes), np.ascontiguousarray(lut), nbits


def build_mlp_model(
    gate, up, down, rows: int, compress: str = "fp16", out_path: str | None = None
):
    """A Core ML program y = down(silu(gate(x)) * up(x)) over x [1, rows, hidden] (fp16) for one slice.

    gate/up/down are dense float matrices ([out, in]) for compress "fp16" and "int8" (per-channel symmetric
    weights), or ``affine_slice`` results (codes, lut, nbits) for compress "lut": an exact per-block palette,
    so the ANE weights equal the MLX values (no extra quantization).
    ``h_exp`` (dense modes): up is scaled by 2**h_exp and down by 2**-h_exp, both exact in fp16, so the fp16
    intermediate silu(gate) * up does not underflow for small activations (the output is unchanged).
    Returns the ``ct.models.MLModel``; saves to ``out_path`` when given.
    """
    import coremltools as ct
    from coremltools.converters.mil import Builder as mb  # noqa: N813
    from coremltools.converters.mil.mil import types
    from coremltools.converters.mil.mil.types.type_mapping import (
        np_uint4_dtype,
        np_uint6_dtype,
    )

    if compress == "lut":
        hidden = gate[0].shape[1]

        def const(t):
            codes, lut, nbits = t
            dt = {4: np_uint4_dtype, 6: np_uint6_dtype, 8: np.uint8}[nbits]
            return mb.constexpr_lut_to_dense(indices=codes.astype(dt), lut=lut)

        def ws():
            return [const(gate), const(up), const(down)]

    else:
        hidden = gate.shape[1]
        g16, u16, d16 = (
            np.ascontiguousarray(a, dtype=np.float16) for a in (gate, up, down)
        )
        if h_exp:
            u16 = np.ascontiguousarray(u16 * np.float16(2.0**h_exp))
            d16 = np.ascontiguousarray(d16 * np.float16(2.0**-h_exp))

        def ws():
            return [g16, u16, d16]

    @mb.program(
        input_specs=[mb.TensorSpec(shape=(1, rows, hidden), dtype=types.fp16)],
        opset_version=ct.target.iOS18,
    )
    def prog(x):
        wg, wu, wd = ws()
        a = mb.linear(x=x, weight=wg, name="gate")
        b = mb.linear(x=x, weight=wu, name="up")
        h = mb.mul(x=mb.silu(x=a), y=b)
        return mb.linear(x=h, weight=wd, name="out")

    model = ct.convert(
        prog,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        skip_model_load=True,
    )
    if compress == "int8":
        cfg = ct.optimize.coreml.OpLinearQuantizerConfig(
            mode="linear_symmetric", dtype="int8", granularity="per_channel"
        )
        conf = ct.optimize.coreml.OptimizationConfig(global_config=cfg)
        model = ct.optimize.coreml.linear_quantize_weights(model, conf)
    if out_path:
        model.save(out_path)
    return model


def mlp_slices(layer: dict, n_gpu: int, compress: str):
    """(gate, up, down) arguments of ``build_mlp_model`` for the trailing intermediate channels [n_gpu, inter)."""
    g, u, d = layer["gate_proj"], layer["up_proj"], layer["down_proj"]
    inter = g["weight"].shape[0]
    if compress == "lut":

        def sl(p, **kw):
            args = (p["weight"], p["scales"], p["biases"], p["bits"], p["group_size"])
            return affine_slice(*args, **kw)

        return (
            sl(g, rows=(n_gpu, inter)),
            sl(u, rows=(n_gpu, inter)),
            sl(d, cols=(n_gpu, inter)),
        )
    W = {
        k: dequantize_affine(
            v["weight"], v["scales"], v["biases"], v["bits"], v["group_size"]
        )
        for k, v in layer.items()
    }
    return mlp_slice(W["gate_proj"], W["up_proj"], W["down_proj"], n_gpu, inter)


def _read_tensors(path: str, names: set[str]) -> dict[str, np.ndarray]:
    """Minimal safetensors reader (bf16 -> float32) so no torch/MLX is needed."""
    import json
    import struct

    out: dict[str, np.ndarray] = {}
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
        base = 8 + n
        for name in names:
            meta = header.get(name)
            if meta is None:
                continue
            a, b = meta["data_offsets"]
            f.seek(base + a)
            buf = f.read(b - a)
            dt = meta["dtype"]
            if dt == "U32":
                arr = np.frombuffer(buf, dtype=np.uint32)
            elif dt == "BF16":
                arr = (
                    np.frombuffer(buf, dtype=np.uint16).astype(np.uint32) << 16
                ).view(np.float32)
            elif dt == "F16":
                arr = np.frombuffer(buf, dtype=np.float16).astype(np.float32)
            elif dt == "F32":
                arr = np.frombuffer(buf, dtype=np.float32)
            else:
                raise ValueError(dt)
            out[name] = arr.reshape(meta["shape"])
    return out


def load_mlp_layer(model_dir: str, layer: int) -> dict[str, dict]:
    """Quantized gate/up/down of one dense layer: {proj: {weight, scales, biases, bits, group_size}}."""
    import json
    import os

    with open(os.path.join(model_dir, "config.json")) as f:
        cfg = json.load(f)
    q = cfg.get("quantization") or cfg["text_config"].get("quantization")
    with open(os.path.join(model_dir, "model.safetensors.index.json")) as f:
        wmap = json.load(f)["weight_map"]
    res: dict[str, dict] = {}
    for proj in ("gate_proj", "up_proj", "down_proj"):
        prefix = f"language_model.model.layers.{layer}.mlp.{proj}"
        spec = q.get(prefix, {})
        names = {f"{prefix}.{s}" for s in ("weight", "scales", "biases")}
        files = {wmap[n] for n in names}
        t: dict[str, np.ndarray] = {}
        for fn in files:
            t.update(_read_tensors(os.path.join(model_dir, fn), names))
        res[proj] = {
            "weight": t[f"{prefix}.weight"],
            "scales": t[f"{prefix}.scales"],
            "biases": t[f"{prefix}.biases"],
            "bits": spec.get("bits", q["bits"]),
            "group_size": spec.get("group_size", q["group_size"]),
        }
    return res
