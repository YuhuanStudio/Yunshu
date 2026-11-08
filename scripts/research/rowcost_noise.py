"""Keyed noise draw: graph vs fused kernel at the verify (31 x 248320) and draft shapes."""

import json
import time


def per_call_ms(fn, reps=20):
    import mlx.core as mx

    mx.eval(fn())
    t = time.perf_counter()
    for _ in range(reps):
        mx.eval(fn())
    return round((time.perf_counter() - t) / reps * 1e3, 3)


def main():
    import mlx.core as mx
    import numpy as np

    from yunshu_engine import keyed_sampling as ks

    vocab = 248320
    for n in (1, 10, 15, 31):
        row = (mx.random.normal((n, vocab)) * 3).astype(mx.float32)
        pos = mx.arange(n, dtype=mx.uint32) + 100
        bases = mx.array(np.full((n,), ks.seed_base(17), dtype=np.uint64))
        mx.eval(row, pos, bases)

        def graph():
            noise = ks._gumbel_rows_graph(bases, pos, vocab)
            return mx.argmax(mx.where(row == -mx.inf, -mx.inf, row + noise), axis=-1)

        def fused():
            return mx.argmax(ks.noised_rows(row, bases, pos), axis=-1)

        assert np.array_equal(np.array(graph()), np.array(fused()))
        print(
            json.dumps(
                dict(n=n, graph_ms=per_call_ms(graph), fused_ms=per_call_ms(fused))
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
