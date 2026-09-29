"""Qwen3.8-27B decode step on the stock mlx-vlm model: dispatch count, launch-bound share,
mx.compile on a decoder layer, wired-memory residency / step variance.

    PYTHONPATH=python:scripts/research/hw python scripts/research/hw/decode_step_model.py $M

Stock mlx-vlm layers (no Yunshu invariant kernels), so absolute times differ from the engine;
the dispatch structure and the compile / wiring deltas are what this measures.
"""

import argparse
import collections
import re
import resource
import subprocess
import tempfile
import time

import mlx.core as mx
from _common import Out
from mlx_vlm import load


def pct(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(len(v) * p))]


def vm_wired_gb():
    o = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    m = re.search(r"Pages wired down:\s+(\d+)", o)
    return int(m.group(1)) * 16384 / 2**30 if m else None


def graph_ops(*outs):
    with tempfile.NamedTemporaryFile("w+", suffix=".dot") as f:
        mx.export_to_dot(f, *outs)
        f.seek(0)
        txt = f.read()
    labels = re.findall(r'label ?= ?"([^"]+)"', txt)
    return txt, labels


def decode_steps(lm, cache, y, n, pipelined=True):
    times = []

    def step(t):
        return mx.argmax(lm(t, cache=cache).logits[:, -1, :], axis=-1, keepdims=True)

    if pipelined:
        nxt = step(y)
        mx.async_eval(nxt)
        last = time.perf_counter()
        for _ in range(n):
            cur = nxt
            nxt = step(cur)
            mx.async_eval(nxt)
            mx.eval(cur)
            now = time.perf_counter()
            times.append(now - last)
            last = now
        mx.eval(nxt)
        return times, nxt
    for _ in range(n):
        t0 = time.perf_counter()
        y = step(y)
        mx.eval(y)
        times.append(time.perf_counter() - t0)
    return times, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--prefill", type=int, default=512)
    a = ap.parse_args()
    out = Out("decode_step_model")

    out(kind="pre_load", wired_GB=vm_wired_gb(), wired_limit_GB=None)
    model, proc = load(a.model)
    lm = model.language_model
    mx.eval(lm.parameters())
    out(kind="post_load", wired_GB=round(vm_wired_gb(), 1),
        active_GB=round(mx.get_active_memory() / 2**30, 2))

    cache = lm.make_cache()
    ids = mx.array([[(i * 7919) % 200000 + 1000 for i in range(a.prefill)]])
    for s in range(0, a.prefill, 256):
        lm(ids[:, s : s + 256], cache=cache)
        mx.eval([c.state for c in cache])
    y = mx.array([[1234]])

    # ---- 1. baseline decode step
    decode_steps(lm, cache, y, 10)
    t, y = decode_steps(lm, cache, y, a.steps)
    out(kind="decode_pipelined", ms_median=round(pct(t, 0.5) * 1e3, 2),
        ms_p95=round(pct(t, 0.95) * 1e3, 2), ms_max=round(max(t) * 1e3, 2))
    t2, y = decode_steps(lm, cache, y, 40, pipelined=False)
    out(kind="decode_sync", ms_median=round(pct(t2, 0.5) * 1e3, 2))

    # ---- 1b. ablation: where the step time goes (stub one component, time the rest)
    import mlx.nn as nn
    from mlx_vlm.models.qwen3_5 import language as qlang

    def ablate(name, patch, undo):
        patch()
        try:
            tt, _ = decode_steps(lm, cache, y, 40)
        finally:
            undo()
        out(kind="ablation", stub=name, ms_median=round(pct(tt, 0.5) * 1e3, 2))

    orig_gdn, orig_sdpa, orig_ql = (
        qlang.gated_delta_update, qlang.scaled_dot_product_attention, nn.QuantizedLinear.__call__)

    def stub_gdn(q, k, v, *a, **kw):
        return mx.zeros_like(v) + q[..., :1, :1].sum() * 0, None

    ablate("gated_delta_kernel",
           lambda: setattr(qlang, "gated_delta_update", stub_gdn),
           lambda: setattr(qlang, "gated_delta_update", orig_gdn))
    ablate("attention_sdpa",
           lambda: setattr(qlang, "scaled_dot_product_attention",
                           lambda q, k, v, cache=None, scale=1.0, mask=None: mx.zeros_like(q)),
           lambda: setattr(qlang, "scaled_dot_product_attention", orig_sdpa))

    def stub_ql(self, x):
        n = self.scales.shape[0]
        return mx.broadcast_to(x[..., :1], (*x.shape[:-1], n)) + 0

    ablate("all_quantized_matmuls",
           lambda: setattr(nn.QuantizedLinear, "__call__", stub_ql),
           lambda: setattr(nn.QuantizedLinear, "__call__", orig_ql))
    t3, y = decode_steps(lm, cache, y, 40)
    out(kind="ablation", stub="none_again", ms_median=round(pct(t3, 0.5) * 1e3, 2))

    # ---- 2. graph size / CPU build time of one step
    tb = []
    for _ in range(5):
        t0 = time.perf_counter()
        logits = lm(y, cache=cache).logits[:, -1, :]
        tb.append(time.perf_counter() - t0)
        mx.eval(logits)
    txt, labels = graph_ops(logits)
    # graph of one step was evaluated already; rebuild for a clean lazy graph
    logits = lm(y, cache=cache).logits[:, -1, :]
    txt, labels = graph_ops(logits)
    hist = collections.Counter(labels)
    out(kind="graph", build_ms_median=round(pct(tb, 0.5) * 1e3, 2), nodes=len(labels),
        edges=txt.count(" -> "), top=hist.most_common(14))
    mx.eval(logits)

    # ---- 3. mx.compile on decoder layers (functional cache wrapper)
    x0 = mx.random.normal((1, 1, 5120)).astype(mx.bfloat16)
    mx.eval(x0)
    for li in (0, 3):
        layer = lm.model.layers[li]
        c = cache[li]
        R = 24

        def eager():
            h = x0
            for _ in range(R):
                h = layer(h, None, c) if layer.is_linear else layer(h, mask=None, cache=c)
            return h

        def timed(fn, n=15):
            for _ in range(3):
                mx.eval(fn())
            s = []
            for _ in range(n):
                t0 = time.perf_counter()
                mx.eval(fn())
                s.append(time.perf_counter() - t0)
            return pct(s, 0.5) / R

        te = timed(eager)
        row = {"layer": li, "linear": layer.is_linear, "eager_us": round(te * 1e6, 1)}
        # MLP block alone (no cache): norm + swiglu + residual
        mlp_fn = lambda h: h + layer.mlp(layer.post_attention_layernorm(h))  # noqa: E731
        mlp_c = mx.compile(mlp_fn)

        def run_mlp(f):
            def g():
                h = x0
                for _ in range(R):
                    h = f(h)
                return h

            return g

        row["mlp_eager_us"] = round(timed(run_mlp(mlp_fn)) * 1e6, 1)
        row["mlp_compiled_us"] = round(timed(run_mlp(mlp_c)) * 1e6, 1)
        _, lab = graph_ops(mlp_fn(x0))
        row["mlp_nodes"] = len(lab)
        if layer.is_linear:
            cls = type(c)

            def fn(h, s0, s1):
                cc = cls(size=2)
                cc.cache = [s0, s1]
                r = layer(h, None, cc)
                return r, cc.cache[0], cc.cache[1]

            fc = mx.compile(fn)

            def comp():
                h, s0, s1 = x0, c.cache[0], c.cache[1]
                for _ in range(R):
                    h, s0, s1 = fc(h, s0, s1)
                return h

            try:
                row["layer_compiled_us"] = round(timed(comp) * 1e6, 1)
                _, lab = graph_ops(layer(x0, None, c))
                row["layer_nodes"] = len(lab)
            except Exception as e:  # noqa: BLE001
                row["layer_compiled_err"] = str(e)[:160]
        out(kind="compile_layer", **row)

    # ---- 4. wired-memory residency and step variance
    def variance(tag):
        ru0 = resource.getrusage(resource.RUSAGE_SELF)
        tt, _ = decode_steps(lm, cache, y, 300)
        ru1 = resource.getrusage(resource.RUSAGE_SELF)
        out(kind="variance", tag=tag, ms_median=round(pct(tt, 0.5) * 1e3, 2),
            ms_p95=round(pct(tt, 0.95) * 1e3, 2), ms_p99=round(pct(tt, 0.99) * 1e3, 2),
            ms_max=round(max(tt) * 1e3, 2), majflt=ru1.ru_majflt - ru0.ru_majflt,
            minflt=ru1.ru_minflt - ru0.ru_minflt, wired_GB=round(vm_wired_gb(), 1))

    variance("default")
    info = mx.device_info()
    wl = int(info["max_recommended_working_set_size"])
    old = mx.set_wired_limit(wl)
    out(kind="set_wired_limit", requested_GB=round(wl / 2**30, 1), previous=old)
    variance("wired_limit_max_recommended")
    mx.set_wired_limit(old)
    mx.set_cache_limit(0)
    variance("cache_limit_0")


if __name__ == "__main__":
    main()
