"""CPU capabilities relevant to drafting / sampling next to a busy GPU.

  * P-core / E-core topology and SME features (sysctl, read only)
  * Accelerate (numpy) sgemm TFLOPS (SME-backed on M4+/M5) and skinny drafter matmuls
  * CPU streaming read/copy bandwidth
  * sampling-type work over a 248320 vocab: argmax, top-k (argpartition), softmax, sort

Pure CPU; does not touch the GPU.

    python scripts/research/hw/cpu_probe.py
"""

import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from _common import Out, timeit  # noqa: E402


def sysctl(k):
    return subprocess.run(
        ["sysctl", "-n", k], capture_output=True, text=True
    ).stdout.strip()


def main():
    out = Out("cpu_probe")
    out(
        kind="topology",
        **{
            k: sysctl(k)
            for k in (
                "hw.perflevel0.physicalcpu",
                "hw.perflevel0.name",
                "hw.perflevel1.physicalcpu",
                "hw.perflevel1.name",
                "hw.perflevel0.l2cachesize",
                "hw.perflevel1.l2cachesize",
                "hw.optional.arm.FEAT_SME2",
                "hw.optional.arm.sme_max_svl_b",
            )
        },
        numpy=np.__version__,
    )

    for n in (1024, 2048, 4096):
        a = np.random.rand(n, n).astype(np.float32)
        b = np.random.rand(n, n).astype(np.float32)
        t, _ = timeit(lambda: a @ b, 10, 2)
        out(
            kind="sgemm_square",
            n=n,
            ms=round(t * 1e3, 2),
            TFLOPS=round(2 * n**3 / t / 1e12, 3),
        )

    for K, N, name in ((5120, 5120, "attn_proj"), (5120, 17408, "mlp_up")):
        w = np.random.rand(K, N).astype(np.float32)
        for M in (1, 2, 4, 8, 16):
            x = np.random.rand(M, K).astype(np.float32)
            t, _ = timeit(lambda: x @ w, 20, 3)
            out(
                kind="skinny_sgemm",
                shape=name,
                M=M,
                K=K,
                N=N,
                us=round(t * 1e6, 1),
                weight_GBps=round(K * N * 4 / t / 1e9, 1),
                GFLOPS=round(2 * M * K * N / t / 1e9, 1),
            )

    big = np.random.rand(2**29).astype(np.float32)  # 2 GiB
    t, _ = timeit(lambda: big.sum(), 5, 1)
    out(kind="cpu_stream_read_sum", GBps=round(big.nbytes / t / 1e9, 1))
    dst = np.empty_like(big)
    t, _ = timeit(lambda: np.copyto(dst, big), 5, 1)
    out(kind="cpu_stream_copy", GBps=round(2 * big.nbytes / t / 1e9, 1))
    big = dst = None

    V = 248320
    for B in (1, 4, 16):
        lg = np.random.randn(B, V).astype(np.float32)
        t, _ = timeit(lambda: lg.argmax(-1), 200, 10)
        out(kind="argmax", B=B, V=V, us=round(t * 1e6, 1))
        t, _ = timeit(lambda: np.argpartition(lg, -50, axis=-1)[:, -50:], 100, 5)
        out(kind="topk50_argpartition", B=B, V=V, us=round(t * 1e6, 1))

        def softmax():
            e = np.exp(lg - lg.max(-1, keepdims=True))
            return e / e.sum(-1, keepdims=True)

        t, _ = timeit(softmax, 100, 5)
        out(kind="softmax", B=B, V=V, us=round(t * 1e6, 1))
        t, _ = timeit(lambda: np.sort(lg, axis=-1), 20, 2)
        out(kind="full_sort", B=B, V=V, us=round(t * 1e6, 1))

    # Latency of a Python thread hand-off (relevant for CPU-side overlap work)
    import queue
    import threading

    q1, q2 = queue.Queue(), queue.Queue()

    def echo():
        while True:
            v = q1.get()
            if v is None:
                return
            q2.put(v)

    th = threading.Thread(target=echo)
    th.start()
    s = []
    for _ in range(2000):
        t0 = time.perf_counter()
        q1.put(1)
        q2.get()
        s.append(time.perf_counter() - t0)
    q1.put(None)
    th.join()
    s.sort()
    out(kind="thread_roundtrip", us_median=round(s[1000] * 1e6, 1))


if __name__ == "__main__":
    main()
