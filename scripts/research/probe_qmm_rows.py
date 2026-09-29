"""Are stock quantized-matmul rows independent of the call's row count?

For 27B projection shapes and bit widths: the first ``m`` rows of a call with
``M`` rows against a call with just those ``m`` rows, bitwise, and the time per
row. If rows at or above some ``m`` never change, prompt spans of at least that
length can share one matmul without losing per-prompt invariance.
"""

import json
import time

import mlx.core as mx

SHAPES = {
    "up": (5120, 17408),
    "down": (17408, 5120),
    "gdn_qkv": (5120, 10240),
    "gdn_z": (5120, 6144),
    "o": (6144, 5120),
    "q": (5120, 12288),
    "kv": (5120, 1024),
    "ab": (5120, 48),
}
BITS = (4, 5, 6, 8)
BASE = (16, 32, 64, 128, 200, 512)
BIG = (512, 1024, 2048, 4096)


def run(x, w, s, b, bits):
    return mx.quantized_matmul(
        x, w, s, b, transpose=True, group_size=64, bits=bits
    )


def main():
    mx.random.seed(1)
    for name, (k, n) in SHAPES.items():
        for bits in BITS:
            w = mx.random.normal((n, k)).astype(mx.bfloat16)
            wq, sc, bi = mx.quantize(w, group_size=64, bits=bits)
            x = (mx.random.normal((max(BIG), k)) * 0.5).astype(mx.bfloat16)
            mx.eval(wq, sc, bi, x)
            rec = {"shape": name, "bits": bits, "differs": {}}
            for m in BASE:
                alone = run(x[:m], wq, sc, bi, bits)
                bad = []
                for big in BIG:
                    if big <= m:
                        continue
                    pooled = run(x[:big], wq, sc, bi, bits)[:m]
                    if not bool(mx.array_equal(alone, pooled).item()):
                        bad.append(big)
                rec["differs"][str(m)] = bad
            t0 = time.perf_counter()
            for big in (512, 2048):
                mx.eval([run(x[:big], wq, sc, bi, bits) for _ in range(3)])
            print(json.dumps(rec), flush=True)


main()
