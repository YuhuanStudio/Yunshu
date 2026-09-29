"""Background load for the GPU-concurrency experiments (ANE model loop or CPU hog).

Protocol: prints READY after warmup, loops until it reads a line on stdin, then prints one
JSON line with the wall-clock (start, end) of every call so the parent can slice a window.

    ane      : CoreML predict loop on CPU_AND_NE  (run with the ane venv python)
    cpu_gemv : numpy M=1 fp32 gemv over a big weight (bandwidth hog)
    cpu_gemm : numpy fp32 sgemm 2048 (compute hog, SME)
"""

import argparse
import json
import sys
import threading
import time

import numpy as np


def make_ane(path):
    import coremltools as ct

    m = ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_AND_NE)
    d = m.get_spec().description.input[0]
    x = np.random.randn(*tuple(d.type.multiArrayType.shape)).astype(np.float32)
    return lambda: m.predict({d.name: x})


def make_cpu_gemv():
    w = np.random.rand(5120, 17408).astype(np.float32)
    x = np.random.rand(1, 5120).astype(np.float32)
    return lambda: x @ w


def make_cpu_gemm():
    a = np.random.rand(2048, 2048).astype(np.float32)
    return lambda: a @ a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True)
    ap.add_argument("--model")
    a = ap.parse_args()
    fn = {"ane": lambda: make_ane(a.model), "cpu_gemv": make_cpu_gemv,
          "cpu_gemm": make_cpu_gemm}[a.kind]()
    for _ in range(5):
        fn()
    stop = threading.Event()
    threading.Thread(target=lambda: (sys.stdin.readline(), stop.set()), daemon=True).start()
    print("READY", flush=True)
    calls = []
    while not stop.is_set():
        t0 = time.time()
        fn()
        calls.append((t0, time.time()))
    print(json.dumps(calls), flush=True)


if __name__ == "__main__":
    main()
