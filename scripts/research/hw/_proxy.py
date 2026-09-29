"""Decode-step and prefill-chunk proxies with 27B-like weight traffic.

The decode proxy chains 64 layers of (gate, up, down, two attention-sized) 4-bit matmuls at
M=1 plus a handful of elementwise ops, so one step streams ~14.5 GB of distinct weights like
the real Qwen3.8-27B decode (no SLC reuse), without loading the checkpoint.
"""

import time

import mlx.core as mx

K, INTER = 5120, 17408
GS, BITS = 64, 4


def _qw(n, k, seed):
    w = mx.random.normal((n, k), key=mx.random.key(seed)).astype(mx.bfloat16)
    q, s, b = mx.quantize(w, group_size=GS, bits=BITS)
    mx.eval(q, s, b)
    return q, s, b


def _mm(x, w):
    return mx.quantized_matmul(x, *w, transpose=True, group_size=GS, bits=BITS)


class DecodeProxy:
    def __init__(self, layers=64, stream=None):
        self.stream = stream
        base = [_qw(INTER, K, 1), _qw(INTER, K, 2), _qw(K, INTER, 3), _qw(K, K, 4), _qw(K, K, 5)]
        # distinct buffers per layer (cheap copies, so DRAM traffic is not SLC-resident)
        self.layers = []
        for i in range(layers):
            self.layers.append([tuple(t + 0 * i for t in w) for w in base])
        mx.eval([t for l in self.layers for w in l for t in w])
        self.nbytes = sum(t.nbytes for l in self.layers for w in l for t in w)
        self.x = mx.random.normal((1, K)).astype(mx.bfloat16)

    def step(self, x):
        for wg, wu, wd, wa, wb in self.layers:
            a = _mm(x, wa)
            x = x + _mm(a, wb)
            h = x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + 1e-6)
            m = mx.sigmoid(_mm(h, wg)) * _mm(h, wu)
            x = x + _mm(m, wd)
        return x

    def run(self, seconds, log=None):
        """Pipelined steps (async_eval one ahead) for `seconds`; returns per-step times."""
        ctx = mx.stream(self.stream) if self.stream is not None else _Null()
        times = []
        with ctx:
            y = self.step(self.x)
            mx.async_eval(y)
            end = time.perf_counter() + seconds
            last = time.perf_counter()
            while time.perf_counter() < end:
                nxt = self.step(y * 0.01 + self.x)
                mx.async_eval(nxt)
                mx.eval(y)
                now = time.perf_counter()
                times.append(now - last)
                last = now
                y = nxt
            mx.eval(y)
        return times


class _Null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class PrefillProxy:
    """A prefill 'chunk' = M tokens through `layers` MLP-like qmm triples."""

    def __init__(self, M=512, layers=8, stream=None):
        self.M, self.layers, self.stream = M, layers, stream
        self.w = [[_qw(INTER, K, 11), _qw(INTER, K, 12), _qw(K, INTER, 13)] for _ in range(1)]
        self.x = mx.random.normal((M, K)).astype(mx.bfloat16)
        mx.eval(self.x)
        self.flops = 2 * M * K * INTER * 3 * layers

    def chunk(self):
        x = self.x
        wg, wu, wd = self.w[0]
        for _ in range(self.layers):
            h = x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + 1e-6)
            m = mx.sigmoid(_mm(h, wg)) * _mm(h, wu)
            x = x + _mm(m, wd)
        return x

    def run(self, seconds):
        ctx = mx.stream(self.stream) if self.stream is not None else _Null()
        n = 0
        t0 = time.perf_counter()
        with ctx:
            while time.perf_counter() - t0 < seconds:
                mx.eval(self.chunk())
                n += 1
        return n, time.perf_counter() - t0


def stats(times):
    times = sorted(times)
    if not times:
        return {}
    n = len(times)
    return {"steps": n, "ms_median": round(times[n // 2] * 1e3, 2),
            "ms_p95": round(times[int(n * 0.95)] * 1e3, 2), "ms_max": round(times[-1] * 1e3, 2),
            "tok_s": round(n / sum(times), 2)}
