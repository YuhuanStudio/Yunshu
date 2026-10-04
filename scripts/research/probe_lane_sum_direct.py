"""Bit/dispatch gate for projection input-sum reuse, GPU queue only."""

import argparse
import json
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    if a.dry_run:
        print(json.dumps(dict(dry_run=True, cases=72)))
        return
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
    import mlx.core as mx
    import mlx.nn as nn
    from lane_sum_direct import install

    from yunshu_engine.kernels.lane_linear import LaneLinear
    from yunshu_engine.kernels.tensorfold import lane_qmm as q

    activate = install()
    factory = q._kernel
    calls = [0]

    def counted(name):
        kernel = factory(name)
        if name != "xsum":
            return kernel

        def run(**kw):
            calls[0] += 1
            return kernel(**kw)

        return run

    q._kernel = counted
    mx.random.seed(617)
    records = []
    for bits, group in ((4, 32), (4, 64), (8, 64)):
        for k, outputs in (
            (512, (48, 64, 128, 256)),
            (2048, (48, 64, 256, 512)),
            (5120, (10240, 6144, 48, 48)),
        ):
            members = []
            for n in outputs:
                linear = nn.Linear(k, n, bias=False)
                linear.weight = linear.weight.astype(mx.bfloat16)
                members.append(
                    LaneLinear.from_quantized(
                        nn.QuantizedLinear.from_linear(
                            linear, group_size=group, bits=bits
                        )
                    )
                )
            mx.eval([m.parameters() for m in members])
            for rows in (1, 8, 16, 32):
                for dtype in (mx.bfloat16, mx.float32):
                    x = mx.random.normal((1, rows, k)).astype(dtype)
                    q._xs_cache.clear()
                    calls[0] = 0
                    activate(False)
                    expected = tuple(m(x) for m in members)
                    mx.eval(expected)
                    baseline_calls = calls[0]
                    q._xs_cache.clear()
                    calls[0] = 0
                    activate(True)
                    got = tuple(m(x) for m in members)
                    mx.eval(got)
                    assert all(
                        bool(mx.array_equal(r, g))
                        for r, g in zip(expected, got, strict=True)
                    ), (bits, group, k, rows, str(dtype))
                    assert calls[0] == (
                        1 if dtype == mx.bfloat16 else baseline_calls
                    ), (bits, group, k, rows, calls[0])
                    assert len(q._xs_cache) <= 4
                    records.append(
                        dict(
                            bits=bits,
                            group=group,
                            k=k,
                            rows=rows,
                            dtype=str(dtype),
                            bit_equal=True,
                            baseline_xsum_calls=baseline_calls,
                            xsum_calls=calls[0],
                        )
                    )
    records.append(dict(complete=True, success=True, cases=len(records)))
    a.output.write_text("".join(json.dumps(r) + "\n" for r in records))
    print(json.dumps(records[-1]), flush=True)


if __name__ == "__main__":
    main()
