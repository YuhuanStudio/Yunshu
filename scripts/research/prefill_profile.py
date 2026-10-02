"""Cold-prefill profile of a Qwen3.5-family checkpoint, by op class.

    prefill_profile.py --engine stock|yunshu --ntok 8192 [--chunk 2048] [--mode plain|classes|peak] [--out F.json]

``stock``  : mlx_lm.load + the stock forward (what TensorFold's prefill runs).
``yunshu`` : the engine's own load (verify kernels / lane projections installed, as in the server).
``plain``  : per-chunk wall time, one eval per chunk (like the engine's prefill step).
``classes``: every projection / norm / GDN core / SDPA call is timed behind a sync; reports share by class and the
             achieved matmul throughput. The syncs cost something, so the total is compared with the plain pass.
``peak``   : the achievable tensor throughput on this GPU (bf16 matmul, quantized matmul at prefill shapes).
Fails closed: a missing piece is an exception, not a zero.
"""

import argparse
import asyncio
import collections
import json
import os
import sys
import time
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

M = "/Volumes/P5Plus/models/Jundot/Qwen3.8-27B-oQ4e-mtp"
PROMPTS = Path("/Volumes/P5Plus/yunshu-build/tfnew/prompts")


def prompt_ids(tok, ntok, kind="prose"):
    text = (PROMPTS / f"{kind}-{32768 if ntok > 8192 else 8192}.txt").read_text()
    msgs = [{"role": "user", "content": text}]
    ids = tok.apply_chat_template(
        msgs, tokenize=True, add_generation_prompt=True, enable_thinking=False
    )
    if isinstance(ids, dict) or hasattr(ids, "input_ids"):
        ids = ids["input_ids"]
    ids = list(ids)
    if len(ids) < ntok:
        raise RuntimeError(f"prompt has {len(ids)} tokens, need {ntok}")
    return ids[:ntok]


def load_stock():
    from mlx_lm import load

    model, tok = load(M)
    return model, tok


def load_yunshu():
    from yunshu_engine.vlm_engine import VLMEngine

    engine = VLMEngine(M)
    asyncio.run(engine.start())
    return engine, engine._processor.tokenizer


def inner_of(model):
    """(decoder with .layers, make_cache owner)."""
    for path in ("language_model.model", "model", "language_model"):
        obj = model
        ok = True
        for part in path.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                ok = False
                break
        if ok and hasattr(obj, "layers"):
            owner = model
            for cand in ("language_model", None):
                o = getattr(model, cand, None) if cand else model
                if o is not None and hasattr(o, "make_cache"):
                    owner = o
                    break
            return obj, owner
    raise RuntimeError("no decoder with layers")


def run_chunks(inner, owner, ids, chunk, *, clear):
    cache = owner.make_cache()
    times = []
    x = mx.array(ids)[None]
    t_all = time.perf_counter()
    for s in range(0, len(ids), chunk):
        t0 = time.perf_counter()
        inner(x[:, s : s + chunk], cache=cache)
        mx.eval([c.state for c in cache])
        times.append(time.perf_counter() - t0)
        if clear:
            mx.clear_cache()
    return times, time.perf_counter() - t_all


def dims(m):
    if hasattr(m, "input_dims") and hasattr(m, "output_dims"):
        return int(m.input_dims), int(m.output_dims)
    w = m.get("weight") if hasattr(m, "get") else None
    if w is None:
        return None
    if w.ndim != 2:
        return None
    if "scales" in m:
        bits = int(getattr(m, "bits", 4))
        return int(w.shape[-1] * 32 // bits), int(w.shape[0])
    return int(w.shape[1]), int(w.shape[0])


class Timers:
    def __init__(self):
        self.t = collections.defaultdict(float)
        self.n = collections.Counter()
        self.flops = collections.defaultdict(float)
        self.rows = 0

    def sync_cost(self):
        x = mx.ones((8,))
        mx.eval(x)
        t0 = time.perf_counter()
        for _ in range(200):
            mx.eval(x + 1)
        return (time.perf_counter() - t0) / 200


def instrument(inner, T):
    """Wrap leaf calls with sync timers."""
    try:
        from mlx_vlm.models.qwen3_5 import language as mod

        if type(inner.layers[0]).__module__ != mod.__name__:
            raise ImportError
    except ImportError:
        from mlx_lm.models import qwen3_5 as mod

    def timed(key, fn, flops=None):
        def run(*a, **k):
            mx.synchronize()
            t0 = time.perf_counter()
            out = fn(*a, **k)
            mx.eval(out)
            T.t[key] += time.perf_counter() - t0
            T.n[key] += 1
            if flops:
                T.flops[key] += flops(*a)
            return out

        return run

    def wrap_module(mod, key):
        cls = type(mod)
        d = dims(mod)
        base_call = cls.__call__

        def call(self, x, *a, **k):
            mx.synchronize()
            t0 = time.perf_counter()
            out = base_call(self, x, *a, **k)
            mx.eval(out)
            T.t[key] += time.perf_counter() - t0
            T.n[key] += 1
            if d:
                rows = 1
                for s in x.shape[:-1]:
                    rows *= s
                T.flops[key] += 2.0 * rows * d[0] * d[1]
            return out

        mod.__class__ = type(cls.__name__ + "T", (cls,), {"__call__": call})

    for layer in inner.layers:
        wrap_module(layer.input_layernorm, "norm")
        wrap_module(layer.post_attention_layernorm, "norm")
        if layer.is_linear:
            la = layer.linear_attn
            for name in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"):
                wrap_module(getattr(la, name), "gdn.in_proj")
            wrap_module(la.out_proj, "gdn.out_proj")
            wrap_module(la.norm, "gdn.gated_norm")
        else:
            at = layer.self_attn
            for name in ("q_proj", "k_proj", "v_proj"):
                wrap_module(getattr(at, name), "attn.qkv_proj")
            wrap_module(at.o_proj, "attn.o_proj")
        ml = layer.mlp
        wrap_module(ml.gate_proj, "mlp.gate_up")
        wrap_module(ml.up_proj, "mlp.gate_up")
        wrap_module(ml.down_proj, "mlp.down")
    mod.gated_delta_update = timed("gdn.core(conv-excluded)", mod.gated_delta_update)
    sdpa = mx.fast.scaled_dot_product_attention
    mx.fast.scaled_dot_product_attention = timed("attn.sdpa", sdpa)
    return mod


def classes(inner, owner, ids, chunk, clear):
    T = Timers()
    sync = T.sync_cost()
    run_chunks(inner, owner, ids[:chunk], chunk, clear=clear)  # warm kernels
    instrument(inner, T)
    T.t.clear(), T.n.clear(), T.flops.clear()
    times, wall = run_chunks(inner, owner, ids, chunk, clear=clear)
    tot = sum(T.t.values())
    layer_wall = wall
    out = {}
    for k in sorted(T.t, key=lambda k: -T.t[k]):
        out[k] = dict(
            s=round(T.t[k], 3),
            n=T.n[k],
            tflops=round(T.flops[k] / T.t[k] / 1e12, 1) if T.flops[k] else None,
            pct_wall=round(100 * T.t[k] / layer_wall, 1),
        )
    out["_other(rope/conv/cache/residual/host)"] = dict(
        s=round(layer_wall - tot, 3),
        pct_wall=round(100 * (layer_wall - tot) / layer_wall, 1),
    )
    return dict(wall_s=round(layer_wall, 3), sync_us=round(sync * 1e6, 1), classes=out)


def peak():
    res = {}
    for m, k, n in ((2048, 5120, 17408), (2048, 5120, 5120), (2048, 17408, 5120)):
        a = mx.random.normal((m, k)).astype(mx.bfloat16)
        w = mx.random.normal((k, n)).astype(mx.bfloat16)
        mx.eval(a, w)
        for _ in range(3):
            mx.eval(a @ w)
        t0 = time.perf_counter()
        for _ in range(10):
            mx.eval(a @ w)
        dt = (time.perf_counter() - t0) / 10
        res[f"bf16 {m}x{k}x{n}"] = round(2 * m * k * n / dt / 1e12, 1)
        wq, sc, bi = mx.quantize(w.T, group_size=64, bits=4)
        mx.eval(wq, sc, bi)
        for _ in range(3):
            mx.eval(
                mx.quantized_matmul(
                    a, wq, sc, bi, transpose=True, group_size=64, bits=4
                )
            )
        t0 = time.perf_counter()
        for _ in range(10):
            mx.eval(
                mx.quantized_matmul(
                    a, wq, sc, bi, transpose=True, group_size=64, bits=4
                )
            )
        dt = (time.perf_counter() - t0) / 10
        res[f"q4g64 {m}x{k}x{n}"] = round(2 * m * k * n / dt / 1e12, 1)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=("stock", "yunshu"), default="stock")
    ap.add_argument("--ntok", type=int, default=8192)
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--mode", choices=("plain", "classes", "peak"), default="plain")
    ap.add_argument("--clear", action="store_true")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out")
    a = ap.parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    if a.mode == "peak":
        res = dict(mode="peak", **peak())
    else:
        if a.engine == "stock":
            model, tok = load_stock()
        else:
            eng, tok = load_yunshu()
            model = eng._model
        ids = prompt_ids(tok, a.ntok)
        inner, owner = inner_of(model)
        if a.mode == "plain":
            run_chunks(inner, owner, ids[: a.chunk], a.chunk, clear=a.clear)  # warm
            runs = []
            for _ in range(a.reps):
                times, wall = run_chunks(inner, owner, ids, a.chunk, clear=a.clear)
                runs.append(
                    dict(wall_s=round(wall, 3), chunks=[round(t, 3) for t in times])
                )
            res = dict(mode="plain", runs=runs)
        else:
            res = dict(mode="classes", **classes(inner, owner, ids, a.chunk, a.clear))
    res.update(engine=a.engine, ntok=a.ntok, chunk=a.chunk, clear=a.clear)
    print(json.dumps(res, indent=1))
    if a.out:
        with open(a.out, "a") as f:
            f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
