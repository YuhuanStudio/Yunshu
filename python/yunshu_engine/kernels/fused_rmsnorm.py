# Patches upstream (MIT): mlx-vlm qwen4_exp Qwen4ExpRMSNorm.__call__ (one fused Metal kernel replaces the
# AsType + Square + Sum + elementwise chain; tracked in vendor.json; Blaizzy/mlx-vlm).
"""One Metal kernel for Qwen4-Exp's RMSNorm ((1 + weight) centered, optional per-group statistics).

Upstream compiles the elementwise parts but a reduction ends a fusion region, so every norm call is
AsType + Square + Sum + a compiled elementwise node (4 kernels; 136 norm calls per Flash-Next decode step,
each kernel costing ~3.5 us of launch/dependency latency on M5).  This kernel does the fp32 statistics and
the scaling of one (row, group) in a single threadgroup with a fixed reduction order (per-thread strided
partial sums, simd_sum, then the simdgroups in order), so a row's result never depends on the other rows or
on how many rows are in flight: decode, verify and prefill agree with each other.  Values can differ from
the unfused chain in the last fp32 bit of the mean (then rarely in one bf16 ulp); that is a different but
equally valid rounding, applied everywhere.
"""

from __future__ import annotations

from functools import cache
from typing import Any

import mlx.core as mx

_STATE: dict[str, Any] = {"original": None, "installed": False}

_SOURCE = """
    const uint gid = threadgroup_position_in_grid.x;
    const uint t = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const size_t base = size_t(gid) * GS;
    const size_t wbase = size_t(gid % NG) * GS;
    float acc = 0.0f;
    for (uint i = t; i < GS; i += TG) {
        const float v = float(X[base + i]);
        acc += v * v;
    }
    acc = simd_sum(acc);
    threadgroup float part[TG / 32];
    if (lane == 0) part[sg] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    for (uint k = 0; k < TG / 32; k++) total += part[k];
    const float rinv = metal::rsqrt(total / float(GS) + EPS);
    for (uint i = t; i < GS; i += TG) {
        float y = float(X[base + i]) * rinv;
        y = y * (1.0f + float(W[wbase + i]));
        OUT[base + i] = static_cast<T>(y);
    }
"""


def _threads(group: int) -> int:
    if group >= 256:
        return 256
    return max(32, ((group + 31) // 32) * 32)


@cache
def _kernel(eps: float):
    # eps is baked into the source: one kernel per distinct eps (only 1e-6 in practice).
    return mx.fast.metal_kernel(
        name=f"yunshu_qwen4_rmsnorm_{abs(hash(eps)) % 10**8}",
        input_names=["X", "W"],
        output_names=["OUT"],
        source=_SOURCE.replace("EPS", repr(float(eps)) + "f"),
    )


def rms_norm(x: Any, weight: Any, group_size: int | None, eps: float) -> Any:
    """(x / rms) * (1 + weight) with statistics per ``group_size`` slice of the last axis (whole axis if None)."""
    width = x.shape[-1]
    group = width if group_size is None else group_size
    n_groups = width // group
    rows = x.size // width
    tg = _threads(group)
    (out,) = _kernel(eps)(
        inputs=[x.reshape(-1), weight.reshape(-1)],
        template=[("T", x.dtype), ("GS", group), ("NG", n_groups), ("TG", tg)],
        grid=(tg * rows * n_groups, 1, 1),
        threadgroup=(tg, 1, 1),
        output_shapes=[(x.size,)],
        output_dtypes=[x.dtype],
    )
    return out.reshape(x.shape)


def install() -> bool:
    """Route ``Qwen4ExpRMSNorm`` through the fused kernel for bf16/fp16 inputs (idempotent)."""
    try:
        from mlx_vlm.models.qwen4_exp import language
    except Exception:  # noqa: BLE001 - mlx_vlm without qwen4_exp
        return False
    cls = language.Qwen4ExpRMSNorm
    if _STATE["installed"]:
        return True
    _STATE["original"] = cls.__call__

    def __call__(self, x):
        if x.dtype in (mx.bfloat16, mx.float16) and self.weight.size == (
            x.shape[-1]
            if self.group_size is None
            else self.group_size * (x.shape[-1] // self.group_size)
        ):
            return rms_norm(x, self.weight, self.group_size, self.eps)
        return _STATE["original"](self, x)

    cls.__call__ = __call__
    _STATE["installed"] = True
    return True


def uninstall() -> None:
    if not _STATE["installed"]:
        return
    from mlx_vlm.models.qwen4_exp import language

    language.Qwen4ExpRMSNorm.__call__ = _STATE["original"]
    _STATE["installed"] = False
