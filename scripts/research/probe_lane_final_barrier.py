"""Paired bit gate and timings for an arithmetic-preserving final barrier."""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[2] / "python")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--variant", choices=("barrier", "paired32"), default="barrier")
    args = parser.parse_args()
    import mlx.core as mx

    if args.variant == "paired32":
        from lane_paired32 import install
    else:
        from lane_final_barrier import install

    from yunshu_engine.kernels.tensorfold import lane_qmm

    lane_qmm.MAX_ROWS = 512
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as out:
        for bits in (2, 3, 4, 5, 6, 8):
            k, n = (1024, 64) if args.smoke else (5120, 17408)
            weight, scales, biases = mx.quantize(
                mx.random.normal((n, k)).astype(mx.bfloat16), bits=bits, group_size=64
            )
            tiled = bits == 4
            if tiled:
                weight = lane_qmm.tile_weight(weight, bits=bits)
            sbt = lane_qmm.pack_scales(scales, biases)
            for rows in (1, 17, 33, 49, 66, 282) if args.smoke else (66, 282):
                x = mx.random.normal((rows, k)).astype(mx.bfloat16)
                mx.eval(x, weight, sbt)

                def run():
                    return lane_qmm.lane_matmul(
                        x, weight, sbt, tiled=tiled, group=64, row_limit=512
                    )

                expected = run()
                mx.eval(expected)
                undo = install()
                actual = run()
                mx.eval(actual)
                equal = bool(mx.array_equal(expected, actual).item())
                undo()
                if not equal:
                    raise RuntimeError((bits, rows, "not bit-equal"))
                for rep in range(1 if args.smoke else 3):
                    for mode in (
                        ("main", "candidate") if rep != 1 else ("candidate", "main")
                    ):
                        undo = install() if mode == "candidate" else None
                        mx.eval(run())
                        times = []
                        for _ in range(1 if args.smoke else 8):
                            mx.synchronize()
                            begin = time.perf_counter()
                            mx.eval(run())
                            times.append(1000 * (time.perf_counter() - begin))
                        if undo:
                            undo()
                        row = dict(
                            bits=bits,
                            split_k=lane_qmm.split_k(n, k),
                            rows=rows,
                            rep=rep,
                            mode=mode,
                            ms=statistics.median(times),
                            equal=equal,
                        )
                        out.write(json.dumps(row) + "\n")
                        out.flush()
                        print(json.dumps(row), flush=True)
        out.write(json.dumps(dict(phase="complete", success=True)) + "\n")


if __name__ == "__main__":
    main()
