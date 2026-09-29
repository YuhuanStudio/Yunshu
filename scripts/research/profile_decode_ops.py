"""Per-component cost of one decode step (T rows) on a Qwen3.5-family target.

Serving stack of the speculative lane (exact oMLX kernels, batch-invariant packed
projections, ragged lane attention); one request, a prefilled cache of --context
tokens, T-row windows (T=1: a plain decode step; T=7: an MTP verify window).

Three views of the same step:

  total     the step pipelined the way BatchGenerator runs it (async_eval on the
            next step before reading the previous), no barriers: what the user gets
  barrier   every projection / norm is wrapped; a barrier (mx.eval) before and
            after each attributes GPU time to it; work between two wrapped modules
            (conv + recurrence, rope + attention, swiglu ...) lands in "pre:<next>".
            The barriers themselves cost ``sync_us`` each (measured), so the sum
            exceeds ``total``; the split is the point
  kernel    each projection group run alone over its real per-layer inputs and
            weights (one eval per group, layers streamed so weights miss the SLC):
            GPU time and achieved GB/s against bytes read, next to the same
            weights on MLX's stock ``quantized_matmul``

    python scripts/research/profile_decode_ops.py MODEL_DIR --context 1024 --rows 1 7 \
        --output runs/decode-ops.jsonl
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

FILLER = "".join(
    f"Log line {i}: sensor {i % 17} reading {i * 37 % 1000}.\n" for i in range(40000)
)


class Barrier:
    """Global attribution clock."""

    def __init__(self):
        self.on = False
        self.acc = defaultdict(float)
        self.cnt = defaultdict(int)
        self.last = 0.0
        self.inputs: dict[str, list] = defaultdict(list)
        self.capture = False

    def start(self):
        mx.synchronize()
        self.last = time.perf_counter()

    def enter(self, label, x):
        if not self.on:
            return
        mx.eval(x)
        t = time.perf_counter()
        self.acc["pre:" + label] += t - self.last
        self.cnt["pre:" + label] += 1
        self.last = t

    def leave(self, label, out):
        if not self.on:
            return
        mx.eval(out)
        t = time.perf_counter()
        self.acc[label] += t - self.last
        self.cnt[label] += 1
        self.last = t


B = Barrier()


class Timed(nn.Module):
    def __init__(self, inner, label, kind):
        super().__init__()
        self.inner = inner
        self._label = label
        self._kind = kind

    def __call__(self, x, *a, **k):
        B.enter(self._label, x)
        if B.capture and self._kind == "linear":
            B.inputs[self._label].append((self.inner, x))
        out = self.inner(x, *a, **k)
        B.leave(self._label, out)
        return out


VIEW_OPS = {
    "Broadcast",
    "Squeeze",
    "ExpandDims",
    "Reshape",
    "Transpose",
    "Slice",
    "AsStrided",
    "StopGradient",
}


def count_graph(out) -> dict:
    """Primitives of the lazy graph ending at ``out`` (the step's launch count)."""
    import re
    import tempfile
    from collections import Counter

    with tempfile.NamedTemporaryFile("w+", suffix=".dot") as f:
        mx.export_to_dot(f.name, out)
        text = open(f.name).read()
    names = Counter(re.findall(r'label ="([^"]+)", shape=rectangle', text))
    kernels = sum(v for k, v in names.items() if k not in VIEW_OPS)
    return {
        "kernels": kernels,
        "nodes": sum(names.values()),
        "top": dict(names.most_common(14)),
    }


def bits_of(m):
    for name in ("bits", "_bits"):
        b = getattr(m, name, None)
        if isinstance(b, int):
            return b
    store = getattr(m, "_store", None)
    if store is not None:
        return 4
    return 0


def weight_bytes(m):
    """Bytes a projection streams per call (codes + scales + biases)."""
    tot = 0
    seen = set()
    stack = [m.parameters()]
    while stack:
        t = stack.pop()
        if isinstance(t, dict):
            stack.extend(t.values())
        elif isinstance(t, (list, tuple)):
            stack.extend(t)
        elif isinstance(t, mx.array):
            if id(t) not in seen:
                seen.add(id(t))
                tot += t.nbytes
    store = getattr(m, "_store", None)
    if store is not None:
        # packed store shared by adjacent projections: attribute this layer's share
        share = m.output_dims / store.N
        tot = int((store.w.nbytes + store.sc.nbytes + store.bi.nbytes) * share)
    return tot


def wrap_model(lm):
    """Replace the leaf modules of interest by Timed wrappers; return the list."""
    wrapped = []
    for i, layer in enumerate(lm.model.layers):
        kind = "gdn" if layer.is_linear else "attn"

        def w(parent, name, label, k="other"):
            inner = getattr(parent, name)
            t = Timed(inner, label, k)
            setattr(parent, name, t)
            wrapped.append((i, label, t))

        w(layer, "input_layernorm", "norm.in")
        w(layer, "post_attention_layernorm", "norm.post")
        if layer.is_linear:
            g = layer.linear_attn
            for n in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"):
                w(g, n, f"gdn.{n}", "linear")
        else:
            a = layer.self_attn
            for n in ("q_proj", "k_proj", "v_proj", "o_proj"):
                w(a, n, f"attn.{n}", "linear")
        for n in ("gate_proj", "up_proj", "down_proj"):
            w(layer.mlp, n, f"mlp.{n}", "linear")
    return wrapped


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model_dir")
    ap.add_argument("--context", type=int, nargs="+", default=[1024])
    ap.add_argument("--rows", type=int, nargs="+", default=[1, 7])
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--stock", action="store_true", help="skip the lane kernels")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument(
        "--lane-linear",
        action="store_true",
        help="TensorFold lane matmul for every projection",
    )
    ap.add_argument("--packed-geometry", default=None, choices=("target", "few"))
    a = ap.parse_args()

    from mlx_vlm import load

    from yunshu_engine.kernels import omlx, ragged_kv
    from yunshu_engine.kernels.omlx import qwen35_packed_linear

    if a.packed_geometry:
        qwen35_packed_linear.GEOMETRY = a.packed_geometry
    from yunshu_engine.kernels.batch_invariant import install as install_invariant
    from yunshu_engine.kernels.batch_invariant import set_active

    model, processor = load(a.model_dir)
    lm = model.language_model
    if not a.stock:
        omlx.apply(row_exact=False)
        if a.lane_linear:
            from yunshu_engine.kernels import lane_linear

            print(
                json.dumps({"lane_linear": lane_linear.convert(lm)["converted"]}),
                flush=True,
            )
        install_invariant(
            lm, model=model, packed=omlx.is_nax_available() and not a.lane_linear
        )
        set_active(True)
        ragged_kv.install()
        ragged_kv.set_dense_lane(True)
    tok = processor.tokenizer
    ids_all = tok.encode(FILLER, add_special_tokens=False)
    args = lm.args
    print(json.dumps({"layers": args.num_hidden_layers}), flush=True)
    wrapped = wrap_model(lm)

    def fresh_cache(ctx):
        cache = lm.make_cache()
        B.on = False
        ids = mx.array(ids_all[:ctx])[None]
        step = 2048
        for s in range(0, ctx, step):
            lm(ids[:, s : s + step], cache=cache)
            mx.eval([c.state for c in cache])
        return cache

    def run_step(cache, toks):
        out = lm(toks, cache=cache)
        logits = out.logits[:, -1]
        return mx.argmax(logits, axis=-1)

    a.output.parent.mkdir(parents=True, exist_ok=True)
    for ctx in a.context:
        cache = fresh_cache(ctx)
        for T in a.rows:
            toks = mx.array(ids_all[ctx : ctx + T])[None]
            # --- total (pipelined) ---
            B.on = False

            def trim(n):
                for c in cache:
                    if hasattr(c, "trim") and getattr(c, "keys", None) is not None:
                        c.trim(n)

            # GDN caches cannot be trimmed: measure on a snapshot per step instead
            # by running steps forward (context grows by T per step; fine).
            for _ in range(3):
                mx.eval(run_step(cache, toks))
            t0 = time.perf_counter()
            nxt = run_step(cache, toks)
            mx.async_eval(nxt)
            for _ in range(a.steps):
                cur = nxt
                nxt = run_step(cache, toks)
                mx.async_eval(nxt)
                mx.eval(cur)
            mx.eval(nxt)
            total_ms = (time.perf_counter() - t0) / (a.steps + 1) * 1e3
            # --- launch count (graph of one step) ---
            graph = count_graph(run_step(cache, toks))
            build_t = time.perf_counter()
            nxg = run_step(cache, toks)
            build_ms = (time.perf_counter() - build_t) * 1e3
            mx.eval(nxg)
            # --- sync overhead ---
            z = mx.zeros((8,))
            mx.eval(z)
            t0 = time.perf_counter()
            for _ in range(200):
                z = z + 1
                mx.eval(z)
            sync_us = (time.perf_counter() - t0) / 200 * 1e6
            # --- barrier ---
            B.acc.clear()
            B.cnt.clear()
            B.inputs.clear()
            B.on = True
            B.capture = True
            n_b = 4
            for r in range(n_b):
                if r == 1:
                    B.capture = False
                    B.acc.clear()
                    B.cnt.clear()
                    n_meas = n_b - 1
                B.start()
                nx = run_step(cache, toks)
                B.enter("lm_head+argmax_done", nx) if False else None
                mx.eval(nx)
                t = time.perf_counter()
                B.acc["tail(final norm..argmax)"] += t - B.last
            B.on = False
            barrier = {
                k: {"ms": v / n_meas * 1e3, "calls": B.cnt[k] // n_meas}
                for k, v in sorted(B.acc.items(), key=lambda kv: -kv[1])
            }
            barrier_total = sum(v["ms"] for v in barrier.values())
            # --- kernel view per projection group ---
            kernel = {}
            groups = defaultdict(list)
            for label, items in B.inputs.items():
                groups[label] = items
            for label, items in groups.items():
                mods = [m for m, _ in items]
                # bit width groups within a label
                by_bits = defaultdict(list)
                for m, x in items:
                    by_bits[bits_of(m)].append((m, x))
                for bits, its in by_bits.items():
                    nbytes = sum(weight_bytes(m) for m, _ in its)
                    xs = [x for _, x in its]
                    mx.eval(xs)

                    def once(its=its):
                        return [m(x) for m, x in its]

                    def stock(its=its):
                        outs = []
                        for m, x in its:
                            w = getattr(m, "weight", None)
                            if w is None or not hasattr(m, "scales"):
                                return None
                            outs.append(
                                mx.quantized_matmul(
                                    x,
                                    m.weight,
                                    m.scales,
                                    m.biases,
                                    transpose=True,
                                    group_size=m.group_size,
                                    bits=m.bits,
                                )
                            )
                        return outs

                    def tm(fn, reps=6):
                        mx.eval(fn())
                        best = 1e9
                        for _ in range(reps):
                            mx.synchronize()
                            t = time.perf_counter()
                            mx.eval(fn())
                            best = min(best, time.perf_counter() - t)
                        return best

                    dt = tm(once)
                    entry = {
                        "layers": len(its),
                        "MB": round(nbytes / 1e6, 1),
                        "ms": round(dt * 1e3, 3),
                        "GBps": round(nbytes / dt / 1e9, 1),
                    }
                    s = stock()
                    if s is not None:
                        dts = tm(stock)
                        entry["stock_ms"] = round(dts * 1e3, 3)
                        entry["stock_GBps"] = round(nbytes / dts / 1e9, 1)
                    kernel[f"{label}[{bits}b]"] = entry
            row = {
                "context": ctx,
                "T": T,
                "total_ms": round(total_ms, 2),
                "sync_us": round(sync_us, 1),
                "graph": graph,
                "build_ms": round(build_ms, 2),
                "barrier_total_ms": round(barrier_total, 2),
                "barrier": {
                    k: {"ms": round(v["ms"], 3), "calls": v["calls"]}
                    for k, v in barrier.items()
                },
                "kernel": kernel,
            }
            with a.output.open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
            mx.clear_cache()


if __name__ == "__main__":
    main()
