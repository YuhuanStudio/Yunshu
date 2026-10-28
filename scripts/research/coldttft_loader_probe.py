"""Actual-shape bit identity and paired kernel cost; execute through gpuq only."""

import argparse
import json
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
    import coldttft_loader
    import mlx.core as mx

    from yunshu_engine.kernels import nax_prefill
    from yunshu_engine.kernels.tensorfold import lane_qmm

    original = nax_prefill.lane_header
    assert nax_prefill.enable(), "unqualified device"
    with open(args.out, "w") as output:

        def emit(record):
            print(json.dumps(record), flush=True)
            output.write(json.dumps(record) + "\n")
            output.flush()

        for k, n, bits in (
            (5120, 17408, 4),
            (17408, 5120, 5),
            (5120, 10240, 4),
            (6144, 5120, 4),
            (5120, 12288, 4),
        ):
            mx.random.seed(1022)
            w = mx.random.normal((n, k)).astype(mx.bfloat16)
            q, scale, bias = mx.quantize(w, group_size=64, bits=bits)
            tiled = lane_qmm.tile_weight(q, lane_qmm.NT, 64, bits=bits)
            sb = lane_qmm.pack_scales(scale, bias)
            mx.eval(tiled, sb)
            del w, q, scale, bias
            for rows in (1, 63, 64, 65, 127, 128, 513, 2048):
                x = mx.random.normal((rows, k)).astype(mx.bfloat16)
                mx.eval(x)
                runs = []
                values = []
                for candidate in (False, True, True, False):
                    nax_prefill.lane_header = (
                        (lambda: coldttft_loader.induction_header(original()))
                        if candidate
                        else original
                    )
                    nax_prefill.compiled.cache_clear()

                    def fn():
                        if rows % 128 == 0 and rows > 512:
                            return nax_prefill.matmul(x, tiled, sb, bits=bits)
                        return nax_prefill.split_matmul(x, tiled, sb, bits=bits)

                    y = fn()
                    mx.eval(y)
                    values.append(y.view(mx.uint16))
                    start = time.perf_counter()
                    for _ in range(4):
                        mx.eval(fn())
                    runs.append((time.perf_counter() - start) / 4)
                equal = all(bool(mx.array_equal(values[0], v)) for v in values[1:])
                emit(
                    dict(
                        k=k,
                        n=n,
                        bits=bits,
                        rows=rows,
                        bit_equal=equal,
                        order=["base", "cand", "cand", "base"],
                        seconds=runs,
                    )
                )
                if not equal:
                    raise RuntimeError("bit mismatch")
            mx.clear_cache()
        emit(dict(complete=True))


if __name__ == "__main__":
    main()
