"""Fail-closed split-invariance screen. Run all MLX work inside gpuq."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cuts = [64, 128, 256, 512, 1024, 1984]
    if args.dry_run:
        print(json.dumps({"cuts": cuts, "families": ["gemm", "attention", "gdn"]}))
        return
    import mlx.core as mx

    from yunshu_engine.kernels import attn_prefill as attn
    from yunshu_engine.kernels import gdn_prefill as gdn
    from yunshu_engine.kernels import nax_prefill as nax
    from yunshu_engine.kernels.lane_linear import LaneLinear

    args.out.parent.mkdir(parents=True, exist_ok=True)
    failures = []

    def record(family, cut, **checks):
        row = dict(family=family, cut=cut, **checks)
        with args.out.open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
        if not all(checks.values()):
            failures.append(row)

    def equal(a, b):
        mx.eval(a, b)
        return bool(
            mx.array_equal(
                a.view(mx.uint16 if a.dtype == mx.bfloat16 else mx.uint32),
                b.view(mx.uint16 if b.dtype == mx.bfloat16 else mx.uint32),
            ).item()
        )

    mx.random.seed(1022)
    assert nax.enable() and attn.supported_device(), "unqualified device/ABI"
    # Real target projection shapes and quantizations, one resident at a time.
    for kdim, ndim, bits in [(5120, 17408, 4), (17408, 5120, 5), (5120, 6144, 4)]:
        weight, scales, biases = mx.quantize(
            mx.random.normal((ndim, kdim)).astype(mx.bfloat16), group_size=64, bits=bits
        )
        layer = LaneLinear(weight, scales, biases, bits, 64)
        x = mx.random.normal((2048, kdim)).astype(mx.bfloat16)
        plain = nax.matmul(x, layer.weight, layer.sbt, bits=bits)
        mx.eval(plain)
        for cut in cuts:
            split = mx.concatenate(
                [
                    nax.split_matmul(x[:cut], layer.weight, layer.sbt, bits=bits),
                    nax.split_matmul(x[cut:], layer.weight, layer.sbt, bits=bits),
                ]
            )
            record(f"gemm-{kdim}-{ndim}-{bits}", cut, rows=equal(plain, split))
        for rows in [1, 2, 31, 63, 65, 127, 511]:
            record(
                f"gemm-tail-{bits}",
                rows,
                rows=equal(
                    plain[:rows],
                    nax.split_matmul(x[:rows], layer.weight, layer.sbt, bits=bits),
                ),
            )
        del layer, weight, scales, biases, x, plain, split
        mx.clear_cache()
    for prefix in [0, 30720]:
        q = mx.random.normal((1, 24, 2048, 256)).astype(mx.bfloat16)
        k = mx.random.normal((1, 4, prefix + 2048, 256)).astype(mx.bfloat16)
        v = mx.random.normal(k.shape).astype(mx.bfloat16)
        plain = attn.attend(q, k, v)
        mx.eval(plain)
        for cut in cuts:
            split = mx.concatenate(
                [
                    attn.attend(
                        q[:, :, :cut], k[:, :, : prefix + cut], v[:, :, : prefix + cut]
                    ),
                    attn.attend(q[:, :, cut:], k, v),
                ],
                axis=2,
            )
            record(f"attention-{prefix}", cut, rows=equal(plain, split))
        for rows in [1, 2, 31, 63, 65, 127, 511]:
            record(
                f"attention-tail-{prefix}",
                rows,
                rows=equal(
                    plain[:, :, :rows],
                    attn.attend(
                        q[:, :, :rows],
                        k[:, :, : prefix + rows],
                        v[:, :, : prefix + rows],
                    ),
                ),
            )
    q = mx.random.normal((1, 2048, 16, 128)).astype(mx.bfloat16) * 0.05
    k = mx.random.normal(q.shape).astype(mx.bfloat16) * 0.05
    v = mx.random.normal((1, 2048, 48, 128)).astype(mx.bfloat16)
    g = mx.exp(-mx.random.uniform(shape=(1, 2048, 48)))
    beta = mx.random.uniform(shape=g.shape).astype(mx.bfloat16)
    state = mx.random.normal((1, 48, 128, 128))
    plain, final = gdn.chunked_padded(q, k, v, g, beta, state)
    mx.eval(plain, final)
    for cut in cuts:
        a, mid = gdn.chunked_padded(
            q[:, :cut], k[:, :cut], v[:, :cut], g[:, :cut], beta[:, :cut], state
        )
        b, end = gdn.chunked_padded(
            q[:, cut:], k[:, cut:], v[:, cut:], g[:, cut:], beta[:, cut:], mid
        )
        record(
            "gdn",
            cut,
            rows=equal(plain, mx.concatenate([a, b], axis=1)),
            state=equal(final, end),
        )
    # Identity padding must preserve the state at the real tail exactly. Compare
    # an explicit 64-row identity suffix, independent of the padding helper.
    for rows in [1, 2, 31, 63, 65, 127, 511]:
        out, tail = gdn.chunked_padded(
            q[:, :rows], k[:, :rows], v[:, :rows], g[:, :rows], beta[:, :rows], state
        )
        record("gdn-tail", rows, rows=equal(out, plain[:, :rows]))

        def zeros(shape, dtype):
            return mx.zeros(shape, dtype=dtype)

        _, after = gdn.chunked_padded(
            zeros((1, 64, 16, 128), q.dtype),
            zeros((1, 64, 16, 128), k.dtype),
            zeros((1, 64, 48, 128), v.dtype),
            mx.ones((1, 64, 48), dtype=g.dtype),
            zeros((1, 64, 48), beta.dtype),
            tail,
        )
        record("gdn-identity", rows, state=equal(tail, after))
    with args.out.open("a") as stream:
        stream.write(json.dumps({"complete": True, "failures": len(failures)}) + "\n")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
