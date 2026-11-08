"""Lane projection time versus row count at the 27B MLP shapes (GB/s of weight bytes)."""

import json
import sys
import time

ROWS = (1, 4, 8, 12, 16, 24, 32)


def gbs(nbytes: int, ms: float) -> float:
    return round(nbytes / ms / 1e6, 1)


def main():
    import mlx.core as mx
    import mlx.nn as nn

    from yunshu_engine.kernels import lane_linear

    mx.random.seed(0)
    reps = 8
    for k, n in ((5120, 17408), (17408, 5120), (5120, 10240)):
        lin = nn.Linear(k, n, bias=False)
        lin.weight = (mx.random.normal((n, k)) * 0.02).astype(mx.bfloat16)
        q = nn.QuantizedLinear.from_linear(lin, group_size=64, bits=4)
        q.scales = q.scales.astype(mx.bfloat16)
        q.biases = q.biases.astype(mx.bfloat16)
        lane = lane_linear.LaneLinear.from_quantized(q)
        mx.eval(lane.parameters())
        nbytes = q.weight.nbytes + q.scales.nbytes + q.biases.nbytes
        for rows in ROWS:
            x = mx.random.normal((1, rows, k)).astype(mx.bfloat16)
            mx.eval(x)

            def run():
                outs = [lane(x + i) for i in range(reps)]
                mx.eval(*outs)

            run()
            t = time.perf_counter()
            for _ in range(10):
                run()
            ms = (time.perf_counter() - t) / 10 / reps * 1e3
            print(
                json.dumps(
                    dict(k=k, n=n, rows=rows, ms=round(ms, 3), GBs=gbs(nbytes, ms))
                ),
                flush=True,
            )


if __name__ == "__main__":
    if sys.argv[1:] != ["--dry"]:
        main()
