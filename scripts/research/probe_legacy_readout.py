# Upstream (derived): ml-explore/mlx (MIT), quantized.h @ v0.32.4.
# Copyright © 2023-2024 Apple Inc. (upstream template call contract; MIT).
# Diagnostic only. Never imported or installed by serving.
"""Call installed stock qmm_t_impl with T partial output, then the old strided sum.

No new multiply/add order is written here. Match actual MLX0.32.3 replay outputs
on every fixed input before considering an end-to-end diagnostic. This is not an
adopted kernel or a performance claim.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import re
from functools import cache
from pathlib import Path

SOURCE = r"""
  constexpr int BM = 32, BK = 32, BN = 32;
  const int K = META[0], N = META[1], M = META[2];
  const int KP = META[3], stride = META[4];
  const uint3 tid = threadgroup_position_in_grid;
  constexpr int pf = get_pack_factor<BITS, 8>();
  constexpr int bp = get_bytes_per_pack<BITS>();
  const int ks = int(tid.z) * KP;
  threadgroup T Xs[BM * (BK + 16 / sizeof(T))];
  threadgroup T Ws[BN * (BK + 16 / sizeof(T))];
  const device uchar* wp = (const device uchar*)W + ks * bp / pf;
  qmm_t_impl<T, GS, BITS, false, T, BM, BK, BN>(
      (const device uint32_t*)wp, S + ks / GS, B + ks / GS,
      X + ks, Y + int(tid.z) * stride, Xs, Ws,
      K, N, M, KP, tid, thread_index_in_threadgroup,
      simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
"""


def geometry(m, n, k, group=64, bits=4):
    """Narrow to the observed readout ABI, not arbitrary quantized projections."""
    if not all(
        isinstance(v, int) and not isinstance(v, bool) for v in (m, n, k, group, bits)
    ):
        raise ValueError("integer dimensions required")
    if m != 15 or n not in (5, 8) or k != 5120 or group != 64 or bits != 4:
        raise ValueError("outside captured ABI")
    # MLX0.32.4 quantized.cpp: BM=BN=32, 512 target threadgroups, K aligned to GS/BK.
    split = min(512 // (((n + 31) // 32) * ((m + 31) // 32)), k // max(32, group))
    while split > 1 and k % (split * max(32, group)):
        split -= 1
    if split <= 1:
        raise ValueError("no split staging")
    return split, k // split


def build_header(root):
    root = Path(root)
    seen = {"mlx/backend/metal/kernels/utils.h"}  # custom kernels prepend MLX utils

    def expand(name):
        if name in seen:
            return ""
        seen.add(name)
        text = (root / name).read_text()
        return re.sub(
            r'^\s*#include "([^"]+)"', lambda m: expand(m[1]), text, flags=re.M
        ).replace("#pragma once", "")

    result = expand("mlx/backend/metal/kernels/quantized.h")
    if "typename U = T" not in result or "METAL_FUNC void qmm_t_impl(" not in result:
        raise ValueError("installed qmm partial-output ABI changed")
    # Private device-buffer dimensions are values, not Metal constant-address-space refs.
    return re.sub(
        r"const constant int& (K|N|M|K_eff)\b(?!\s*\[\[buffer)", r"const int \1", result
    )


@cache
def kernel():
    import mlx.core as mx

    if importlib.metadata.version("mlx") != "0.32.4":
        raise RuntimeError(
            "prototype is qualified only against the inspected 0.32.4 header"
        )
    root = importlib.metadata.distribution("mlx").locate_file("mlx/include")
    return mx.fast.metal_kernel(
        name="probe_legacy_readout_partials_v1",
        input_names=["W", "S", "B", "X", "META"],
        output_names=["Y"],
        source=SOURCE,
        header=build_header(root),
    )


def matmul(x, w, s, b, *, group_size=64, bits=4):
    import mlx.core as mx

    if (
        x.ndim != 2
        or w.ndim != 2
        or any(a.dtype != mx.bfloat16 for a in (x, s, b))
        or w.dtype != mx.uint32
    ):
        raise ValueError("expected contiguous BF16 readout / packed uint32 weights")
    m, k = map(int, x.shape)
    n = int(w.shape[0])
    split, kp = geometry(m, n, k, group_size, bits)
    if w.shape != (n, k // 8) or s.shape != (n, k // group_size) or b.shape != s.shape:
        raise ValueError("weight/group shape mismatch")
    meta = mx.array([k, n, m, kp, m * n], dtype=mx.int32)
    partial = kernel()(
        inputs=[w, s, b, x, meta],
        template=[("T", mx.bfloat16), ("GS", group_size), ("BITS", bits)],
        grid=(((n + 31) // 32) * 32, ((m + 31) // 32) * 2, split * 2),
        threadgroup=(32, 2, 2),
        output_shapes=[(split, m, n)],
        output_dtypes=[mx.bfloat16],
    )[0]
    return mx.sum(partial, axis=0)


def fixture_inputs():
    """Exact dyadic inputs: old SK80 staging predicts63, float partials predict63.25."""
    import numpy as np

    x = np.zeros((15, 5120), np.float32)
    x[:, : 63 * 64 : 64] = 1.0
    x[:, 1 : 63 * 64 : 64] = 2**-8
    return (
        x,
        np.full((8, 640), 0x11111111, np.uint32),
        np.ones((8, 80), np.float32),
        np.zeros((8, 80), np.float32),
    )


def reference_inputs(directory, reference):
    from probe_draft_readout import digest, read_manifest

    root = Path(directory)
    capture = read_manifest(root)
    if reference.get("complete") is not True or reference.get("mlx") != "0.32.3":
        raise ValueError("reference must be a complete actual MLX0.32.3 replay")
    if reference.get("manifest_sha256") != digest(root / "manifest.json"):
        raise ValueError("reference input manifest differs")
    refs = reference.get("frames", [])
    if len(refs) != len(capture["frames"]):
        raise ValueError("reference frame coverage differs")
    for frame, ref in zip(capture["frames"], refs, strict=True):
        m, k = frame["shape"]
        geometry(m, frame["width"], k, frame["group_size"], frame["bits"])
        if frame["file"] != ref["file"] or frame["sha256"] != ref["input_sha256"]:
            raise ValueError("reference frame inputs differ")
        shape = [m, frame["width"]]
        if (
            ref["shape"] != shape
            or len(ref["logits"]) != m
            or any(len(row) != shape[1] for row in ref["logits"])
        ):
            raise ValueError("reference logits shape differs")
        if any(not math.isfinite(v) for row in ref["logits"] for v in row):
            raise ValueError("nonfinite reference")
    return capture, refs


def run(args):
    import mlx.core as mx
    from probe_draft_readout import write_result

    reference = json.loads(Path(args.reference).read_text())
    capture, refs = reference_inputs(args.directory, reference)
    device = mx.metal.device_info()
    if device.get("architecture") != "applegpu_g17s" or any(
        reference["device"].get(k) != device.get(k)
        for k in ("device_name", "architecture", "memory_size")
    ):
        raise ValueError("not the same qualified M5 device")
    x, w, s, b = (
        mx.array(a, dtype=mx.uint32 if i == 1 else mx.bfloat16)
        for i, a in enumerate(fixture_inputs())
    )
    legacy = matmul(x, w, s, b)
    stock = mx.quantized_matmul(x, w, s, b, group_size=64, bits=4, transpose=True)
    mx.eval(legacy, stock)
    pilot = {
        "legacy_min": float(mx.min(legacy).item()),
        "legacy_max": float(mx.max(legacy).item()),
        "stock_min": float(mx.min(stock).item()),
        "stock_max": float(mx.max(stock).item()),
    }
    if pilot != {
        "legacy_min": 63.0,
        "legacy_max": 63.0,
        "stock_min": 63.25,
        "stock_max": 63.25,
    }:
        raise RuntimeError(f"analytic staging pilot failed: {pilot}")
    rows = []
    for frame, ref in zip(capture["frames"], refs, strict=True):
        a = mx.load(str(Path(args.directory) / frame["file"]))
        out = matmul(
            a["x"],
            a["w"],
            a["s"],
            a["b"],
            group_size=frame["group_size"],
            bits=frame["bits"],
        )
        want = mx.array(ref["logits"], dtype=mx.bfloat16)
        mx.eval(out, want)
        rows.append(
            {
                "file": frame["file"],
                "bit_mismatches": int(
                    mx.sum(out.view(mx.uint16) != want.view(mx.uint16)).item()
                ),
                "max_abs_old_delta": float(
                    mx.max(
                        mx.abs(out.astype(mx.float32) - want.astype(mx.float32))
                    ).item()
                ),
            }
        )
    result = {
        "complete": True,
        "passed": all(r["bit_mismatches"] == 0 for r in rows),
        "diagnostic_only": True,
        "mlx": importlib.metadata.version("mlx"),
        "reference_mlx": reference["mlx"],
        "manifest_sha256": reference["manifest_sha256"],
        "device": device,
        "analytic_pilot": pilot,
        "frames": rows,
    }
    write_result(args.out, result)
    print(json.dumps(result), flush=True)
    return 0 if result["passed"] else 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory", required=True)
    p.add_argument("--reference", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    try:
        return run(args)
    except Exception as exc:
        from probe_draft_readout import write_result

        write_result(
            args.out,
            {
                "complete": True,
                "passed": False,
                "diagnostic_only": True,
                "error": str(exc),
            },
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
