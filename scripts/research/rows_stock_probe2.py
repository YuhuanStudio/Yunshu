"""Which row counts R keep the batch-axis stock call bit-identical to R stock M=1 calls? (all R = 2..8)"""

import sys

import mlx.core as mx
import mlx.nn as nn


class gi:  # the batch-axis forms the probe checks (the module that held them was removed)
    @staticmethod
    def _dense_batched(w):
        wt = w.T
        return lambda xs: mx.matmul(xs, mx.broadcast_to(wt, (xs.shape[0], *wt.shape)))

    @staticmethod
    def _quant_batched(mod, w, scales, biases):
        def batched(xs):
            r = xs.shape[0]
            bw, bs, bb = (
                mx.broadcast_to(a, (r, *a.shape)) for a in (w, scales, biases)
            )
            return mx.quantized_matmul(
                xs, bw, bs, bb, transpose=True, group_size=mod.group_size, bits=mod.bits
            )

        return batched


mx.random.seed(0)
bad = []
total = 0
DENSE = [
    (2560, 2048),
    (2560, 512),
    (2048, 2560),
    (2560, 10240),
    (10240, 2560),
    (2560, 262144),
    (2560, 4096),
    (256, 2560),
    (2560, 10752),
]
QUANT = [
    (1536, 2048),
    (1536, 256),
    (2048, 1536),
    (1536, 6144),
    (6144, 1536),
    (1536, 12288),
    (1536, 4096),
    (2048, 4096),
]
for kind in ("dense", "q4g64", "q8g64", "q4g32"):
    shapes = DENSE if kind == "dense" else QUANT
    for k, n in shapes:
        x = mx.random.normal((8, 1, k)).astype(mx.bfloat16)
        w = (mx.random.normal((n, k)) * 0.05).astype(mx.bfloat16)
        if kind == "dense":
            lin = nn.Linear(k, n, bias=False)
            lin.weight = w
            batched = gi._dense_batched(w)
        else:
            bits, group = int(kind[1]), int(kind.split("g")[1])
            lin = nn.QuantizedLinear(k, n, bias=False, group_size=group, bits=bits)
            lin.weight, lin.scales, lin.biases = mx.quantize(
                w, group_size=group, bits=bits
            )
            batched = gi._quant_batched(lin, lin.weight, lin.scales, lin.biases)
        mx.eval(lin.parameters())
        ref = [lin(x[r : r + 1]) for r in range(8)]
        mx.eval(ref)
        fails = []
        for rows in range(2, 9):
            got = batched(x[:rows])
            ok = all(
                bool(mx.array_equal(got[r], ref[r][0]).item()) for r in range(rows)
            )
            total += 1
            if not ok:
                fails.append(rows)
        print(f"{kind} K={k} N={n}: rows not row-exact = {fails}", flush=True)
        if fails:
            bad.append((kind, k, n, fails))
print("TOTAL cells", total, "failing shapes", len(bad))
sys.exit(0)
