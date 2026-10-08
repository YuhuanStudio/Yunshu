"""Published MLX MoE pack formats: tensor-derived quantization and unsupported-pack checks.

Third-party packs (pierre/pipenetwork/ddalcu/mlx-community mixed 2-bit conversions,
TensorFold MTP packs) declare quantization in non-uniform ways: per-module dicts keyed by
checkpoint or module names, extra top-level fields (``expert_bits``), no ``quantization``
block at all (native mx fp4/fp8), or a custom codec stored beside ordinary tensors.
The packed tensors themselves are the ground truth: with the module's true input width,
``weight[-1]`` and ``scales[-1]`` determine ``bits`` and ``group_size``.  Pure functions
over shapes, so they are testable without MLX.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import cast

_AFFINE_BITS = (2, 3, 4, 5, 6, 8)
_AFFINE_GROUPS = (32, 64, 128)


def infer_quantization(
    in_dim: int,
    weight_shape: Sequence[int],
    scales_shape: Sequence[int],
    *,
    scales_dtype: str,
    has_biases: bool,
) -> dict | None:
    """Quantization parameters implied by packed tensor shapes, or None if inconsistent.

    ``in_dim`` is the unquantized input width of the module (``weight.shape[-1]`` of the
    freshly built bf16 module).  Packed ``weight`` is uint32 with ``in_dim * bits / 32``
    columns; ``scales`` has ``in_dim / group_size`` columns.
    """
    wlast, slast = int(weight_shape[-1]), int(scales_shape[-1])
    if in_dim <= 0 or slast <= 0 or in_dim % slast or (wlast * 32) % in_dim:
        return None
    bits = wlast * 32 // in_dim
    group = in_dim // slast
    if has_biases:
        if bits in _AFFINE_BITS and group in _AFFINE_GROUPS:
            return {"bits": bits, "group_size": group, "mode": "affine"}
        return None
    if scales_dtype.endswith("uint8"):
        if group == 32 and bits == 4:
            return {"bits": 4, "group_size": 32, "mode": "mxfp4"}
        if group == 32 and bits == 8:
            return {"bits": 8, "group_size": 32, "mode": "mxfp8"}
        if group == 16 and bits == 4:
            return {"bits": 4, "group_size": 16, "mode": "nvfp4"}
    return None


def quantization_matches(declared: dict | None, derived: dict) -> bool:
    if not declared:
        return False
    return cast(
        bool,
        declared.get("bits") == derived["bits"]
        and declared.get("group_size") == derived["group_size"]
        and declared.get("mode", "affine") == derived["mode"],
    )


_CODEC_MARKERS = ("tq_packed", "tq_norms", "tq_bits", "mxtq", "jangtq")


def unsupported_pack_reason(config: dict, tensor_names: Iterable[str]) -> str | None:
    """Reason a pack needs a custom runtime Yunshu does not ship, else None.

    Custom expert codecs (TurboQuant/JANGTQ ``tq_packed`` tensors plus a sidecar runtime)
    would otherwise load as an un-quantized model with the codec tensors silently dropped.
    """
    q = config.get("quantization")
    if isinstance(q, dict):
        for key in q:
            if str(key).lower() in ("routed_expert_bit_plan", "mxtq_bits", "mxtq_seed"):
                return f"custom expert codec declared in quantization config ({key})"
    for name in tensor_names:
        last = name.rsplit(".", 1)[-1]
        if last in _CODEC_MARKERS:
            return f"custom expert codec tensor {name!r} (needs the pack's own runtime)"
    return None


def tensor_quantization_override(
    weights, path: str, module_weight_shape, top_level: dict, per_module
) -> dict | None:
    """Tensor-derived quantization for ``path`` when the config disagrees with the tensors.

    Returns the derived parameters only when the checkpoint holds ``path.scales`` and the
    declared (per-module, else top-level) parameters do not reproduce the packed shapes;
    returns None when the declaration is right or the tensors cannot decide.
    """
    scales = weights.get(f"{path}.scales")
    weight = weights.get(f"{path}.weight")
    if scales is None or weight is None or not module_weight_shape:
        return None
    derived = infer_quantization(
        int(module_weight_shape[-1]),
        weight.shape,
        scales.shape,
        scales_dtype=str(scales.dtype),
        has_biases=f"{path}.biases" in weights,
    )
    if derived is None:
        return None
    declared = per_module if isinstance(per_module, dict) else None
    if declared is None and per_module is not False:
        declared = top_level
    if declared and quantization_matches({"mode": "affine", **declared}, derived):
        return None
    return derived
