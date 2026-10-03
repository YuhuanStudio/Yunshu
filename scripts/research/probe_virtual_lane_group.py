"""Numerical/ownership gate for virtual column stacking; GPU queue only."""

import argparse
import hashlib
import json
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    if a.dry_run:
        print(json.dumps(dict(dry_run=True, cases=48)))
        return
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
    import mlx.core as mx
    import mlx.nn as nn
    from virtual_lane_group import grouped_linears

    from yunshu_engine.kernels.lane_linear import LaneLinear

    mx.random.seed(616)
    records = []
    for group in (32, 64):
        for k, n0, n1 in (
            (512, 64, 96),
            (2048, 512, 512),
            (4096, 1024, 1024),
            (5120, 17408, 17408),
        ):
            members = []
            for n in (n0, n1):
                linear = nn.Linear(k, n, bias=False)
                linear.weight = linear.weight.astype(mx.bfloat16)
                quant = nn.QuantizedLinear.from_linear(linear, group_size=group, bits=4)
                members.append(LaneLinear.from_quantized(quant))
            originals = [(m.weight, m.sbt) for m in members]
            mx.eval([m.parameters() for m in members])
            for rows in (1, 8, 16, 32, 48, 128):
                x = mx.random.normal((1, rows, k)).astype(mx.bfloat16)
                expected = tuple(m(x) for m in members)
                result = grouped_linears(members, x)
                assert result is not None
                mx.eval(expected, result)
                assert all(
                    bool(mx.array_equal(r, g))
                    for r, g in zip(expected, result, strict=True)
                ), (group, k, n0, n1, rows)
                assert all(
                    m.weight is w and m.sbt is s
                    for m, (w, s) in zip(members, originals, strict=True)
                )
                records.append(
                    dict(
                        group=group,
                        k=k,
                        n0=n0,
                        n1=n1,
                        rows=rows,
                        bit_equal=True,
                        parameters_unchanged=True,
                    )
                )
    records.append(
        dict(
            complete=True,
            success=True,
            cases=len(records),
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            kernel_sha256=hashlib.sha256(
                Path(__file__).with_name("virtual_lane_group.py").read_bytes()
            ).hexdigest(),
        )
    )
    a.output.write_text("".join(json.dumps(r) + "\n" for r in records))
    print(json.dumps(records[-1]), flush=True)


if __name__ == "__main__":
    main()
