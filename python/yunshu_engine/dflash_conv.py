# Upstream (derived): Blaizzy/mlx-vlm (MIT) dflash2._grouped_dynamic_convolve @ v0.7.4
"""Private DFlash2 convolution: one dispatch, explicit rounding boundaries.

The target and shared readout are untouched. Unsupported operand layouts or
precision keep the upstream proposal implementation.
"""

from typing import Any, cast

import mlx.core as mx

_SOURCE = r"""
    uint idx = thread_position_in_grid.x;
    if (idx >= B * L * C) return;
    uint c = idx % C, t = (idx / C) % L, batch = idx / (C * L);
    uint group = c / GS;
    InT acc = InT(0);
    for (int offset = 0; offset < K; ++offset) {
        InT x = t >= uint(offset) ? hidden[batch * hidden_strides[0] + (t - offset) * hidden_strides[1] + c * hidden_strides[2]] : InT(0);
        InT base_value = InT(base[offset * base_strides[0] + c * base_strides[1]]);
        InT coefficient = InT(float(base_value) + float(dynamic[batch * dynamic_strides[0] + t * dynamic_strides[1] + offset * dynamic_strides[2] + group * dynamic_strides[3]]));
        InT product = InT(float(coefficient) * float(x));
        acc = InT(float(acc) + float(product));
    }
    output[idx] = acc;
"""
_KERNEL: Any = None


def grouped_conv(
    hidden: mx.array, dynamic: mx.array, base: mx.array, group_size: int
) -> mx.array:
    global _KERNEL
    if hidden.dtype != dynamic.dtype:
        raise ValueError("prototype requires matching activation/kernel dtype")
    batch, length, channels = hidden.shape
    if (
        group_size <= 0
        or channels % group_size
        or base.ndim != 2
        or base.shape[1] != channels
        or dynamic.shape != (batch, length, base.shape[0], channels // group_size)
    ):
        raise ValueError("invalid grouped convolution shape")
    if _KERNEL is None:
        _KERNEL = mx.fast.metal_kernel(
            name="wide3_draft_conv_direct",
            input_names=["hidden", "dynamic", "base"],
            output_names=["output"],
            source=_SOURCE,
            ensure_row_contiguous=False,
        )
    return cast(
        mx.array,
        _KERNEL(
            inputs=[hidden, dynamic, base],
            template=[
                ("InT", hidden.dtype),
                ("B", batch),
                ("L", length),
                ("C", channels),
                ("G", channels // group_size),
                ("GS", group_size),
                ("K", base.shape[0]),
            ],
            grid=(batch * length * channels, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[hidden.shape],
            output_dtypes=[hidden.dtype],
        )[0],
    )


def install(drafter: Any) -> int:
    """Replace only the private conv instance methods; return installed count."""
    from mlx_vlm.speculative.drafters.dflash2.dflash2 import (
        GroupedDynamicCausalConv,
        _grouped_dynamic_convolve,
    )

    count = 0
    for layer in getattr(drafter, "layers", []):
        for name in ("attention_conv", "mlp_conv"):
            conv = getattr(layer, name, None)
            if not isinstance(conv, GroupedDynamicCausalConv):
                continue
            if getattr(conv, "_yunshu_direct_conv", False):
                continue
            if conv.base_kernel.ndim != 3 or conv.base_kernel.shape[:2] != (
                2,
                conv.kernel_size,
            ):
                continue

            def convolve(hidden, dynamic, base, group_size):
                supported = (
                    hidden.ndim == 3
                    and hidden.dtype in (mx.bfloat16, mx.float32)
                    and dynamic.dtype == hidden.dtype
                    and group_size > 0
                    and hidden.shape[-1] % group_size == 0
                    and base.ndim == 2
                    and base.shape[-1] == hidden.shape[-1]
                    and base.shape[0] >= 1
                    and dynamic.shape
                    == (
                        *hidden.shape[:-1],
                        base.shape[0],
                        hidden.shape[-1] // group_size,
                    )
                )
                if not supported:
                    return _grouped_dynamic_convolve(hidden, dynamic, base, group_size)
                return grouped_conv(hidden, dynamic, base, group_size)

            def prepare(hidden, module=conv):
                groups = hidden.shape[-1] // module.group_size
                dynamic = module.kernel_projection(hidden).reshape(
                    *hidden.shape[:-1], 2, module.kernel_size, groups
                )
                return (
                    convolve(
                        hidden,
                        dynamic[..., 0, :, :],
                        module.base_kernel[0],
                        module.group_size,
                    ),
                    dynamic[..., 1, :, :],
                )

            def finish(hidden, dynamic, module=conv):
                return convolve(
                    hidden, dynamic, module.base_kernel[1], module.group_size
                )

            object.__setattr__(conv, "prepare", prepare)
            object.__setattr__(conv, "finish", finish)
            object.__setattr__(conv, "_yunshu_direct_conv", True)
            count += 1
    return count
