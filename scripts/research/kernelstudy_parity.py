# Upstream (inspired): ml-explore/mlx (MIT), sdpa_vector.h @ 36969c094354086e15d2a327a9c50feb20b9cf11
"""Rejudge TK4 numerics and canonical per-row adaptation; run only via gpuq.

The adaptation launches the pinned research TK4 body once per causal query row.
It is deliberately not wired into serving. Cache test covers tensor restoration,
not end-to-end APC/GDN state or the paired accuracy gate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def canonical_lengths(end, count):
    if not 1 <= count <= 7 or end < count:
        raise ValueError("invalid canonical query window")
    return tuple(range(end - count + 1, end + 1))


def validate(records):
    expected = {(n, t) for n in (32768, 65536, 131072) for t in range(1, 8)}
    if len(records) != 21 or {(r["context"], r["T"]) for r in records} != expected:
        raise ValueError("missing or duplicate correctness cells")
    return all(
        r["ours"]["within_existing_atol_4e_3"]
        and r["external"]["within_existing_atol_4e_3"]
        and r["deterministic"]
        and r["verify_equals_decode"]
        and r["cache_tensor_hit_equals_miss"]
        for r in records
    )


def error_metrics(output, reference):
    import numpy as np

    delta = np.abs(output.astype(np.float64) - reference)
    return {
        "max_abs": float(delta.max()),
        "max_rel": float((delta / np.maximum(np.abs(reference), 1e-12)).max()),
        # Existing ragged attention contract: tests/unit/test_ragged_kv.py:54.
        "within_existing_atol_4e_3": bool(
            np.isfinite(delta).all() and delta.max() < 4e-3
        ),
    }


def moe_check(out):
    import mlx.core as mx
    import numpy as np
    from kernelstudy_micro import external_module

    from yunshu_engine.kernels.omlx import moe_verify_gather as local

    other, provenance = external_module("moe_verify_gather")
    mx.random.seed(181026)
    w = mx.random.normal((512, 640, 2560)).astype(mx.bfloat16)
    weight, scales, biases = mx.quantize(w, group_size=64, bits=4)
    mx.eval(weight, scales, biases)
    del w

    class Linear(dict):
        bits = 4
        group_size = 64

    linear = Linear(weight=weight, scales=scales, biases=biases)
    x = mx.random.normal((7, 2560)).astype(mx.bfloat16)
    mx.eval(x)
    reference = np.empty((7, 10, 640), np.float64)
    xx = np.array(x.astype(mx.float32)).astype(np.float64)
    for expert in range(10):
        packed = np.array(weight[expert])
        quant = (
            (packed[..., None] >> (np.arange(8, dtype=np.uint32) * 4)) & 15
        ).reshape(640, 2560)
        ss = np.repeat(
            np.array(scales[expert].astype(mx.float32)).astype(np.float64), 64, axis=-1
        )
        bb = np.repeat(
            np.array(biases[expert].astype(mx.float32)).astype(np.float64), 64, axis=-1
        )
        dequant = quant.astype(np.float64) * ss + bb
        reference[:, expert] = xx @ dequant.T
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as stream:

        def emit(row):
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            print(json.dumps(row), flush=True)

        emit(
            {
                "kind": "metadata",
                "candidate": "moe",
                "reference": "fp64 affine dequant and dot",
                **provenance,
            }
        )
        for count in range(1, 8):
            picks = mx.tile(mx.arange(10, dtype=mx.uint32), count)
            results = {}
            for name, module in (("ours", local), ("external", other)):
                y = module.gather_qmv(linear, x[:count], picks, 10)
                again = module.gather_qmv(linear, x[:count], picks, 10)
                single = mx.concatenate(
                    [
                        module.gather_qmv(linear, x[r : r + 1], picks[:10], 10)
                        for r in range(count)
                    ]
                )
                mx.eval(y, again, single)
                yy = np.array(y.astype(mx.float32)).reshape(count, 10, 640)
                metrics = error_metrics(yy, reference[:count])
                # Attention's tolerance does not apply to unnormalised MoE dots.
                metrics.pop("within_existing_atol_4e_3")
                results[name] = {
                    **metrics,
                    "deterministic": bool(mx.array_equal(y, again).item()),
                    "verify_equals_decode": bool(mx.array_equal(y, single).item()),
                }
            emit({"T": count, **results})
        emit({"complete": True, "rows": 7, "serving_qualified": False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--candidate", choices=("attention", "moe"), default="attention"
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        print(
            json.dumps(
                {
                    "contexts": [32768, 65536, 131072],
                    "rows": list(range(1, 8)),
                    "gpu_executed": False,
                }
            )
        )
        return
    if args.candidate == "moe":
        return moe_check(args.out)
    import mlx.core as mx
    import numpy as np
    from kernelstudy_micro import latest_vector

    from yunshu_engine.kernels.ragged_attention import ragged_decode_attention

    mx.random.seed(181027)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    records = []
    with args.out.open("w") as stream:

        def emit(row):
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            print(json.dumps(row), flush=True)

        emit(
            {
                "kind": "metadata",
                "engaged": "MLX0324 TK4 canonical single-row adaptation; fp64 numpy reference",
                "serving_qualified": False,
            }
        )
        for context in (32768, 65536, 131072):
            # All T windows end at the same absolute position. Each query row
            # always uses its own absolute prefix length and split ladder.
            q = mx.random.normal((1, 24, 7, 256)).astype(mx.bfloat16)
            k = mx.random.normal((1, 4, context, 256)).astype(mx.bfloat16)
            v = mx.random.normal(k.shape).astype(mx.bfloat16)
            mx.eval(q, k, v)
            k_np, v_np, q_np = [np.array(x.astype(mx.float32)) for x in (k, v, q)]
            reference = np.empty(q.shape, np.float64)
            for h in range(24):
                kk = k_np[0, h // 6].astype(np.float64)
                vv = v_np[0, h // 6].astype(np.float64)
                scores = q_np[0, h].astype(np.float64) @ kk.T * 0.0625
                for r in range(7):
                    scores[r, context - 6 + r :] = -np.inf
                scores -= scores.max(axis=-1, keepdims=True)
                weights = np.exp(scores)
                weights /= weights.sum(axis=-1, keepdims=True)
                reference[0, h] = weights @ vv
            del k_np, v_np, q_np, kk, vv, scores, weights

            def adapted(query, keys, values, end):
                count = query.shape[2]
                outputs = []
                for row, n in enumerate(canonical_lengths(end, count)):
                    # Canonical contiguous row and prefix: same geometry in
                    # decode, verify, and restored-cache paths.
                    outputs.append(
                        latest_vector(
                            mx,
                            mx.contiguous(query[:, :, row : row + 1]),
                            mx.contiguous(keys[:, :, :n]),
                            mx.contiguous(values[:, :, :n]),
                            n,
                        )()
                    )
                return mx.concatenate(outputs, axis=2)

            singles = adapted(q, k, v, context)
            mx.eval(singles)
            for count in range(1, 8):
                query = q[:, :, 7 - count :]
                out = adapted(query, k, v, context)
                repeat = adapted(query, k, v, context)
                # Force a materialised prefix-cache copy with different backing.
                split = context - 32
                restored_k = mx.concatenate([k[:, :, :split], k[:, :, split:]], axis=2)
                restored_v = mx.concatenate([v[:, :, :split], v[:, :, split:]], axis=2)
                restored = adapted(query, restored_k, restored_v, context)
                old = ragged_decode_attention(
                    query,
                    k,
                    v,
                    mx.array([context], mx.int32),
                    0.0625,
                    max_length=context,
                    impl="tile",
                    row_lengths=[context],
                )
                mx.eval(out, repeat, restored, old)
                target = reference[:, :, 7 - count :]
                row = {
                    "context": context,
                    "T": count,
                    "ours": error_metrics(np.array(old.astype(mx.float32)), target),
                    "external": error_metrics(np.array(out.astype(mx.float32)), target),
                    "deterministic": bool(mx.array_equal(out, repeat).item()),
                    "verify_equals_decode": bool(
                        mx.array_equal(out, singles[:, :, 7 - count :]).item()
                    ),
                    "cache_tensor_hit_equals_miss": bool(
                        mx.array_equal(out, restored).item()
                    ),
                    "end_to_end_apc_qualified": False,
                }
                records.append(row)
                emit(row)
            del q, k, v, singles, out, repeat, restored, restored_k, restored_v, old
            mx.synchronize()
            mx.clear_cache()
        passed = validate(records)
        emit(
            {
                "complete": True,
                "rows": len(records),
                "component_pass": passed,
                "serving_qualified": False,
            }
        )


if __name__ == "__main__":
    main()
