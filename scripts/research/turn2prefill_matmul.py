"""Check whether padded native tiled-weight QMM equals the existing short stock QMM."""

import argparse
import json
import sys
from pathlib import Path


def padded_rows(rows):
    return ((rows + 127) // 128) * 128


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    shapes = [(5120, 17408), (17408, 5120), (5120, 12288), (6144, 5120)]
    rows = (513, 534, 535, 536, 600, 1023, 2047)
    if args.dry_run:
        print(json.dumps({"shapes": shapes, "rows": rows, "out": str(args.out)}))
        return
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
    import mlx.core as mx
    import mlx.nn as nn

    from yunshu_engine.kernels import lane_linear, nax_prefill

    assert nax_prefill.enable(), "unqualified GPU/MLX"
    mx.random.seed(1021)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    all_equal = True
    with args.out.open("w") as out:
        for k, n in shapes:
            for bits in (4, 5, 8):
                linear = nn.QuantizedLinear(k, n, bias=False, group_size=64, bits=bits)
                linear.scales = linear.scales.astype(mx.bfloat16)
                linear.biases = linear.biases.astype(mx.bfloat16)
                lane = lane_linear.LaneLinear.from_quantized(linear)
                mx.eval(lane.parameters(), linear.parameters())
                for m in rows:
                    x = mx.random.normal((m, k)).astype(mx.bfloat16)
                    ref = linear(x)
                    xp = mx.pad(x, [(0, padded_rows(m) - m), (0, 0)])
                    got = nax_prefill.matmul(xp, lane.weight, lane.sbt, bits=bits)[:m]
                    mx.eval(ref, got)
                    equal = bool(mx.array_equal(ref, got).item())
                    all_equal &= equal
                    record = dict(
                        k=k,
                        n=n,
                        bits=bits,
                        rows=m,
                        bit_equal=equal,
                        mismatches=int(mx.sum(ref != got).item()),
                    )
                    out.write(json.dumps(record) + "\n")
                    out.flush()
                    print(json.dumps(record), flush=True)
                del linear, lane
                mx.clear_cache()
        out.write(json.dumps({"complete": True, "all_equal": all_equal}) + "\n")
    if not all_equal:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
