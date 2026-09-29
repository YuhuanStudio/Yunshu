"""Can decode keep running while another request prefills on the same GPU?

Decode proxy (27B-like weight traffic, ~11.5 GB / step) on the main thread, a prefill proxy
either (a) on a second thread with its own MLX stream (own MTLCommandQueue), or (b)
interleaved on the same thread, one prefill chunk per decode step (what a single-queue
scheduler does today). Prefill chunk size is swept: its command-buffer length bounds how
long a decode step can be delayed.

    PYTHONPATH=scripts/research/hw python scripts/research/hw/gpu_stream_overlap.py
"""

import argparse
import threading
import time

import mlx.core as mx
from _common import Out
from _proxy import DecodeProxy, PrefillProxy, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=8)
    a = ap.parse_args()
    out = Out("gpu_stream_overlap")
    dec = DecodeProxy(stream=mx.new_stream(mx.gpu))
    out(kind="proxy", weight_GB=round(dec.nbytes / 1e9, 2))

    base = stats(dec.run(a.seconds))
    out(kind="decode_alone", GBps=round(dec.nbytes / (base["ms_median"] / 1e3) / 1e9, 1), **base)

    for M, layers in ((128, 4), (512, 8), (2048, 8), (2048, 32)):
        pa = PrefillProxy(M, layers)
        n, dt = pa.run(a.seconds / 2)
        pre_tps = n * M / dt
        out(kind="prefill_alone", M=M, layers=layers, chunk_ms=round(dt / n * 1e3, 1),
            TFLOPS=round(pa.flops * n / dt / 1e12, 1))

        # (a) two threads, two streams
        pre = PrefillProxy(M, layers)
        res = {}

        def bg(pre=pre, res=res):
            pre.stream = mx.new_stream(mx.gpu)
            res["n"], res["dt"] = pre.run(a.seconds)

        th = threading.Thread(target=bg)
        th.start()
        time.sleep(0.5)
        d = stats(dec.run(a.seconds - 1))
        th.join()
        out(kind="decode_plus_prefill", mode="two_streams_two_threads", M=M, layers=layers,
            decode_tok_s_ratio=round(d["tok_s"] / base["tok_s"], 3),
            prefill_tok_s_ratio=round(res["n"] * M / res["dt"] / pre_tps, 3), **d)

        # (b) same thread: one decode step then one prefill chunk, repeated
        pre = PrefillProxy(M, layers)
        times = []
        chunks = 0
        with mx.stream(dec.stream):
            t_end = time.perf_counter() + a.seconds
            last = time.perf_counter()
            while time.perf_counter() < t_end:
                y = dec.step(dec.x)
                mx.eval(y)
                now = time.perf_counter()
                times.append(now - last)
                mx.eval(pre.chunk())
                chunks += 1
                last = time.perf_counter()
        d = stats(times)
        out(kind="decode_plus_prefill", mode="interleaved_single_queue", M=M, layers=layers,
            decode_tok_s_ratio=round(len(times) / a.seconds / base["tok_s"], 3),
            prefill_tok_s_ratio=round(chunks * M / a.seconds / pre_tps, 3), **d)


if __name__ == "__main__":
    main()
