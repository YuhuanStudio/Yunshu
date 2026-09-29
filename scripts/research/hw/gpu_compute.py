"""GPU compute peaks: dense matmul TFLOPS per dtype and quantized matmul (qmm/qmv).

Also reports the M at which decode-shaped quantized matmul (K=5120, N=17408, the
Qwen3.8-27B MLP up/gate shape) leaves the weight-bandwidth-bound regime.

Run twice to A/B the M5 tensor unit (NAX): once as-is, once with
MLX_METAL_GPU_ARCH=applegpu_g16s (forces the non-NAX kernels; --tag nonax).

    PYTHONPATH=scripts/research/hw python scripts/research/hw/gpu_compute.py [--tag nax]
"""

import argparse

import mlx.core as mx
from _common import Out, timeit


def sync_loop(fn, reps):
    """Time `reps` back-to-back dispatches inside one eval; returns seconds per op.

    fn takes the rep index so callers can cycle through distinct weight copies (a single
    weight smaller than the SLC would otherwise be re-read from cache, not DRAM).
    """

    def run():
        mx.eval([fn(i) for i in range(reps)])

    t, _ = timeit(run, 5, 2)
    return t / reps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="nax")
    a = ap.parse_args()
    out = Out(f"gpu_compute_{a.tag}")
    from yunshu_engine.kernels.omlx import is_nax_available

    out(
        kind="env",
        nax_available=is_nax_available(),
        tag=a.tag,
        mlx=mx.__version__,
        arch=mx.device_info()["architecture"],
    )

    # Dense square matmul peak
    for dt in (mx.float32, mx.float16, mx.bfloat16):
        for n in (2048, 4096, 8192):
            x = (mx.random.normal((n, n)) * 0.1).astype(dt)
            y = (mx.random.normal((n, n)) * 0.1).astype(dt)
            mx.eval(x, y)
            reps = 8 if n >= 8192 else 24
            s = sync_loop(lambda i, x=x, y=y: x @ y, reps)
            out(
                kind="dense_square",
                dtype=str(dt).split(".")[-1],
                n=n,
                ms=round(s * 1e3, 3),
                TFLOPS=round(2 * n**3 / s / 1e12, 2),
            )

    # Dense bf16 skinny (decode-like) M sweep with an unquantized MLP-size weight
    K, N = 5120, 17408
    NC = 4
    w = (mx.random.normal((K, N)) * 0.02).astype(mx.bfloat16)
    ws = [w + mx.array(0, dtype=mx.bfloat16) * j for j in range(NC)]
    mx.eval(ws)
    for M in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 2048, 8192):
        x = mx.random.normal((M, K)).astype(mx.bfloat16)
        mx.eval(x)
        s = sync_loop(lambda i, x=x: x @ ws[i % NC], 24)
        out(
            kind="dense_bf16_mlp",
            M=M,
            us=round(s * 1e6, 1),
            TFLOPS=round(2 * M * K * N / s / 1e12, 2),
            weight_GBps=round(K * N * 2 / s / 1e9, 1),
        )

    # Quantized matmul, weight-stationary decode shapes and prefill shapes
    for bits in (4, 5, 8):
        gs = 64
        wq, sc, bi = mx.quantize(
            w.T.astype(mx.float32).astype(mx.bfloat16), group_size=gs, bits=bits
        )  # (N, K)
        cp = [(wq + j * 0, sc + j * 0, bi + j * 0) for j in range(8)]
        mx.eval(cp)
        wbytes = wq.nbytes + sc.nbytes + bi.nbytes
        for M in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192):
            x = mx.random.normal((M, K)).astype(mx.bfloat16)
            mx.eval(x)
            reps = 32 if M <= 64 else 8
            s = sync_loop(
                lambda i, x=x: mx.quantized_matmul(
                    x, *cp[i % 8], transpose=True, group_size=gs, bits=bits
                ),
                reps,
            )
            out(
                kind="qmm",
                bits=bits,
                M=M,
                K=K,
                N=N,
                us=round(s * 1e6, 1),
                TFLOPS=round(2 * M * K * N / s / 1e12, 2),
                weight_GBps=round(wbytes / s / 1e9, 1),
            )


if __name__ == "__main__":
    main()
