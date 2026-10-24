"""Real FFN micro A/B including GPU layout conversion, shared-event handoff and join.

Not serving integration. Uses real checkpoint weights and resident MLX inputs;
no host copies occur on the candidate hot path. All work runs under gpuq.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from yunshu_engine.ane import convert
from yunshu_engine.ane.native import NativeSession
from yunshu_engine.ane.native.mil import mlp_program


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model", default="/Volumes/P5Plus/models/Jundot/Qwen3.8-27B-oQ4e-mtp"
    )
    p.add_argument("--rows", type=int, default=1024)
    p.add_argument("--channels", type=int, default=2176)
    p.add_argument("--mode", choices=["fp16", "int8"], default="fp16")
    p.add_argument("--reps", type=int, default=7)
    p.add_argument("--inner", type=int, default=5)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    return p


def valid_shape(rows, channels, inter):
    if rows < 1 or channels <= 0 or channels >= inter or channels % 64:
        raise ValueError("positive rows and proper64-aligned channel slice required")
    return max(128, (rows + 63) // 64 * 64)


def main():
    a = parser().parse_args()
    raw = convert.load_mlp_layer(a.model, 40)
    inter = raw["gate_proj"]["weight"].shape[0]
    program_rows = valid_shape(a.rows, a.channels, inter)
    if min(a.reps, a.inner) < 1:
        raise ValueError("positive reps/inner required")
    import mlx.core as mx

    hidden = raw["gate_proj"]["weight"].shape[1] * 32 // raw["gate_proj"]["bits"]
    n_gpu = inter - a.channels
    quant = {
        k: {
            "w": mx.array(v["weight"]),
            "s": mx.array(v["scales"]).astype(mx.bfloat16),
            "b": mx.array(v["biases"]).astype(mx.bfloat16),
            "bits": v["bits"],
            "gs": v["group_size"],
        }
        for k, v in raw.items()
    }
    x = mx.array(
        np.random.default_rng(0).normal(0, 0.7, (a.rows, hidden)).astype(np.float32)
    ).astype(mx.bfloat16)
    mx.eval(x)

    def qmm(values, proj, n=None, axis=0):
        p = quant[proj]
        w, s, b = p["w"], p["s"], p["b"]
        if n is not None:
            if axis == 0:
                w, s, b = w[:n], s[:n], b[:n]
            else:
                w, s, b = (
                    w[:, : n * p["bits"] // 32],
                    s[:, : n // p["gs"]],
                    b[:, : n // p["gs"]],
                )
        return mx.quantized_matmul(
            values, w, s, b, transpose=True, group_size=p["gs"], bits=p["bits"]
        )

    def gpu(n=None):
        gate = qmm(x, "gate_proj", n)
        up = qmm(x, "up_proj", n)
        return qmm((gate * mx.sigmoid(gate)) * up, "down_proj", n, axis=1)

    for _ in range(25):
        mx.eval(gpu())
    reference = np.array(gpu().astype(mx.float32))
    mil, blob = mlp_program(
        *convert.mlp_slices(raw, n_gpu, a.mode), program_rows, a.mode, 6
    )
    session, reason = NativeSession.try_create(
        mil,
        blob,
        [(hidden, program_rows)],
        (hidden, program_rows),
        a.cache,
        mlx_root=Path(mx.__file__).parent,
    )
    if session is None:
        raise RuntimeError(reason)
    with session:
        inp, reason = session.try_alias(0)
        if inp is None:
            raise RuntimeError(reason)
        out, reason = session.try_alias(1)
        if out is None:
            raise RuntimeError(reason)
        producer = mx.fast.metal_kernel(
            name="anenative_ffn_pack",
            input_names=["x", "surface"],
            output_names=["marker"],
            source="""uint i = thread_position_in_grid.x;
if (i < H*P) {
  uint channel = i/P, row = i%P;
  const_cast<device half*>(surface)[i] = row < R ? half(x[row*H+channel]) : half(0);
}
if (i == 0) marker[0] = 1;
""",
        )
        join = mx.fast.metal_kernel(
            name="anenative_ffn_join",
            input_names=["gpu", "ane"],
            output_names=["out"],
            source="""uint i = thread_position_in_grid.x;
if (i < R*H) {
  uint row = i/H, channel = i%H;
  out[i] = bfloat16_t(float(gpu[i]) + float(ane[channel*P+row]));
}
""",
        )
        template = [("H", hidden), ("R", a.rows), ("P", program_rows)]

        def candidate():
            marker = producer(
                inputs=[x, inp],
                template=template,
                grid=(hidden * program_rows, 1, 1),
                threadgroup=(256, 1, 1),
                output_shapes=[(1,)],
                output_dtypes=[mx.uint32],
            )[0]
            mx.async_eval(marker)
            ticket, why = session.try_submit_mlx()
            if ticket is None:
                raise RuntimeError(why)
            with ticket:
                part = gpu(n_gpu)
                mx.async_eval(part)
                if not ticket.wait_mlx():
                    raise RuntimeError(ticket.reason)
                result = join(
                    inputs=[part, out],
                    template=template,
                    grid=(hidden * a.rows, 1, 1),
                    threadgroup=(256, 1, 1),
                    output_shapes=[(a.rows, hidden)],
                    output_dtypes=[mx.bfloat16],
                )[0]
                mx.eval(result)
                if not ticket.finish():
                    raise RuntimeError(ticket.reason)
            return result, part

        result, part = candidate()
        # Check join arithmetic against ordinary MLX fp32 add then bf16 rounding.
        expected = (
            part.astype(mx.float32) + out.T[: a.rows].astype(mx.float32)
        ).astype(mx.bfloat16)
        actual = np.array(result.astype(mx.float32))
        expected_np = np.array(expected.astype(mx.float32))
        assert np.array_equal(actual.view(np.uint32), expected_np.view(np.uint32)), (
            "join bits differ"
        )
        rel = float(np.linalg.norm(actual - reference) / np.linalg.norm(reference))
        if not np.isfinite(actual).all() or rel > 0.10:
            raise ValueError(f"invalid handoff output relL2={rel}")
        for _ in range(10):
            candidate()
        a.out.parent.mkdir(parents=True, exist_ok=True)
        with a.out.open("w") as stream:

            def emit(record):
                stream.write(json.dumps(record) + "\n")
                stream.flush()
                print(json.dumps(record), flush=True)

            emit(
                {
                    "kind": "meta",
                    "rows": a.rows,
                    "program_rows": program_rows,
                    "channels": a.channels,
                    "mode": a.mode,
                    "layer": 40,
                    "device": "m5",
                    "host_copies_included": False,
                    "host_copies_required": False,
                    "gpu_pack_included": True,
                    "gpu_join_included": True,
                    "shared_events": True,
                    "serving_engaged": False,
                    "join_bit_check": True,
                    "rel_l2_vs_gpu": rel,
                }
            )
            for arm in ["base", "cand", "cand-r1", "base-r1", "cand-r2", "base-r2"]:
                times = []
                for _ in range(a.reps):
                    start = time.perf_counter()
                    for _ in range(a.inner):
                        if arm.startswith("cand"):
                            candidate()
                        else:
                            mx.eval(gpu())
                    times.append((time.perf_counter() - start) * 1000 / a.inner)
                emit(
                    {
                        "kind": "handoff",
                        "arm": arm,
                        "rows": a.rows,
                        "channels": a.channels,
                        "mode": a.mode,
                        "ms": times,
                        "median_ms": float(np.median(times)),
                        "cached": False,
                    }
                )
            emit({"complete": True})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
