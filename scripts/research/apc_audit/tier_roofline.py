"""Roofline for an APC WARM tier on the real model (GPU: run through gpuq).

Prefills a real corpus document (the accuracy corpus: code / prose, 32K tokens) with the stock mlx-vlm
path, then measures, on the resulting exact checkpoint:

  composition  attention KV bytes vs GDN recurrent state bytes (per token / per checkpoint)
  lossless     lz4 / zstd levels, optionally after a bf16 byte-plane shuffle: ratio, compress and
               decompress GB/s (1 thread and 8 threads over 4 MiB chunks)
  lossy        KV int8 / 4-bit (mx.quantize, affine, per-group) and the GDN state cast to bf16 / int8:
               size, dequantize time, KLD(exact || restored) and top-1 agreement over teacher-forced
               real continuation tokens
  restore      HOT clone time, lossless-WARM decode time, lossy dequantize time

    tier_roofline.py --model M --doc long_code_32k --prefix 32000 --out roofline.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import mlx.core as mx
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "accuracy"))
import common as C  # noqa: E402, N812

MiB = 1 << 20
GB = 1e9


def arrays_of(c):
    """(kind, arrays) of one prompt-cache layer through the state contract."""
    st = c.state
    name = type(c).__name__
    arrs = [a for a in (st if isinstance(st, (list, tuple)) else [st]) if a is not None]
    return name, arrs


def raw_bytes(a: mx.array) -> np.ndarray:
    mx.eval(a)
    # copy: a zero-copy view would dangle once the temporary mx view is freed
    if a.dtype == mx.bfloat16:
        return np.array(a.view(mx.uint16)).reshape(-1).view(np.uint8)
    return np.array(a).reshape(-1).view(np.uint8)


def clone_state(c, transform=None):
    """A new cache layer from c.state (arrays passed through ``transform(kind, idx, a)``)."""
    name, _ = arrays_of(c)
    st = c.state
    seq = isinstance(st, (list, tuple))
    items = list(st) if seq else [st]
    new = [None if a is None else mx.contiguous(mx.array(a)) for a in items]
    if transform:
        new = [None if a is None else transform(name, i, a) for i, a in enumerate(new)]
    new = tuple(new) if isinstance(st, tuple) else (new if seq else new[0])
    out = type(c).__new__(type(c))
    from_state = getattr(type(c), "from_state", None)
    if callable(from_state):
        return from_state(new, c.meta_state)
    out.state = new
    out.meta_state = c.meta_state
    return out


def plane_shuffle(buf: np.ndarray) -> np.ndarray:
    """bf16 byte planes: all low bytes, then all high bytes."""
    u = buf.reshape(-1, 2)
    return np.concatenate([u[:, 0], u[:, 1]])


def comp_bench(name, fn_c, fn_d, data: np.ndarray, threads: int):
    chunk = 4 * MiB
    parts = [data[i : i + chunk].tobytes() for i in range(0, data.size, chunk)]
    t0 = time.perf_counter()
    if threads == 1:
        comp = [fn_c(p) for p in parts]
    else:
        with ThreadPoolExecutor(threads) as ex:
            comp = list(ex.map(fn_c, parts))
    tc = time.perf_counter() - t0
    t0 = time.perf_counter()
    if threads == 1:
        dec = [fn_d(p) for p in comp]
    else:
        with ThreadPoolExecutor(threads) as ex:
            dec = list(ex.map(fn_d, comp))
    td = time.perf_counter() - t0
    assert b"".join(dec) == data.tobytes()
    csize = sum(len(p) for p in comp)
    return {
        "codec": name,
        "threads": threads,
        "ratio": round(data.size / csize, 3),
        "comp_GBps": round(data.size / tc / GB, 2),
        "decomp_GBps": round(data.size / td / GB, 2),
    }


def sample(arrs, limit):
    out, n = [], 0
    for a in arrs:
        b = raw_bytes(a)
        take = b[: max(0, limit - n)]
        out.append(take)
        n += take.size
        if n >= limit:
            break
    return np.concatenate(out)


def kld_stats(ref_logits, cand_logits):
    """KLD(ref || cand) per position and top-1 agreement. Logits [T, V] float32."""
    lp = ref_logits - mx.logsumexp(ref_logits, axis=-1, keepdims=True)
    lq = cand_logits - mx.logsumexp(cand_logits, axis=-1, keepdims=True)
    kl = mx.sum(mx.exp(lp) * (lp - lq), axis=-1)
    top = mx.mean(
        (mx.argmax(ref_logits, -1) == mx.argmax(cand_logits, -1)).astype(mx.float32)
    )
    kl = np.array(kl)
    return {
        "kld_mean": float(kl.mean()),
        "kld_p99": float(np.percentile(kl, 99)),
        "kld_max": float(kl.max()),
        "top1": float(top.item()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--doc", default="long_code_32k")
    ap.add_argument("--prefix", type=int, default=32000)
    ap.add_argument("--cont", type=int, default=256)
    ap.add_argument("--sample-mb", type=int, default=384)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--cpu", action="store_true", help="debug on the CPU device (no GPU work)"
    )
    a = ap.parse_args()
    if a.cpu:
        mx.set_default_device(mx.cpu)

    import lz4.frame
    import zstandard as zstd

    corpus = C.load_corpus()
    seq = next(s for s in corpus["seqs"] if s["name"] == a.doc)
    ids = [int(t) for t in seq["ids"]]
    prefix, cont = ids[: a.prefix], ids[a.prefix : a.prefix + a.cont]
    model, _tok = C.load_model(a.model)
    lm = model.language_model
    res: dict = {"doc": a.doc, "prefix": a.prefix, "cont": len(cont)}

    def prefill(tokens):
        cache = lm.make_cache()
        t0 = time.perf_counter()
        for i in range(0, len(tokens), 2048):
            lm(mx.array([tokens[i : i + 2048]]), cache=cache)
            mx.eval(*[x for c in cache for x in arrays_of(c)[1]])
        return cache, time.perf_counter() - t0

    def compose(cache, ntok):
        kv = gdn = 0
        kinds: dict[str, dict] = {}
        for c in cache:
            name, arrs = arrays_of(c)
            nb = sum(x.nbytes for x in arrs)
            d = kinds.setdefault(
                name, {"layers": 0, "bytes": 0, "dtypes": [], "shapes": []}
            )
            d["layers"] += 1
            d["bytes"] += nb
            if not d["shapes"]:
                d["shapes"] = [list(x.shape) for x in arrs]
                d["dtypes"] = [str(x.dtype) for x in arrs]
            if name == "KVCache":
                kv += nb
            else:
                gdn += nb
        return {
            "tokens": ntok,
            "kv_bytes": kv,
            "gdn_bytes": gdn,
            "kv_bytes_per_token": kv / ntok,
            "gdn_bytes_total": gdn,
            "kinds": kinds,
        }

    print("prefill", len(prefix), flush=True)
    cache, tpf = prefill(prefix)
    res["prefill_s"] = round(tpf, 1)
    res["prefill_tok_s"] = round(len(prefix) / tpf, 1)
    res["composition"] = compose(cache, len(prefix))
    print(json.dumps(res["composition"]), flush=True)

    # smaller checkpoint for the composition fit (KV ~ linear, GDN constant)
    c8, _ = prefill(prefix[:8192])
    res["composition_8k"] = compose(c8, 8192)
    del c8

    # ── lossless compressibility ──────────────────────────────────────────
    kv_arrs = [
        x for c in cache if type(c).__name__ == "KVCache" for x in arrays_of(c)[1]
    ]
    gdn_arrs = [
        x for c in cache if type(c).__name__ != "KVCache" for x in arrays_of(c)[1]
    ]
    lim = a.sample_mb * MiB
    kv_s = sample(kv_arrs[::7] if len(kv_arrs) > 14 else kv_arrs, lim)
    gdn_s = sample(gdn_arrs, lim)
    lossless = []
    for label, data, planes in (("kv", kv_s, True), ("gdn", gdn_s, False)):
        variants = [(label, data)]
        if planes:
            variants.append((label + "+planes", plane_shuffle(data)))
        for vname, d in variants:
            row = {"data": vname, "bytes": int(d.size)}
            outs = []
            for threads in (1, 8):
                outs.append(
                    comp_bench(
                        "lz4",
                        lambda p: lz4.frame.compress(p, compression_level=0),
                        lz4.frame.decompress,
                        d,
                        threads,
                    )
                )
                for lvl in (1, 3, 9):
                    # a (de)compressor object is not thread-safe: one per chunk
                    outs.append(
                        comp_bench(
                            f"zstd{lvl}",
                            lambda p, lvl=lvl: zstd.ZstdCompressor(level=lvl).compress(
                                p
                            ),
                            lambda p: zstd.ZstdDecompressor().decompress(p),
                            d,
                            threads,
                        )
                    )
            row["results"] = outs
            lossless.append(row)
            print(json.dumps(row), flush=True)
    res["lossless"] = lossless

    # ── restore costs ─────────────────────────────────────────────────────
    t0 = time.perf_counter()
    cl = [clone_state(c) for c in cache]
    mx.eval(*[x for c in cl for x in arrays_of(c)[1]])
    res["hot_clone_s"] = round(time.perf_counter() - t0, 3)
    del cl

    # ── lossy variants + KLD ──────────────────────────────────────────────
    def qdq(bits, group):
        def f(name, i, x):
            if name != "KVCache":
                return x
            q = mx.quantize(x, group_size=group, bits=bits)
            return mx.dequantize(*q, group_size=group, bits=bits)

        return f

    def gdn_cast(kind):
        def f(name, i, x):
            if name == "KVCache":
                return x
            if kind == "bf16":
                return x.astype(mx.bfloat16).astype(x.dtype)
            if x.ndim >= 2 and x.shape[-1] % 32 == 0:
                q = mx.quantize(x.astype(mx.float32), group_size=32, bits=8)
                return mx.dequantize(*q, group_size=32, bits=8).astype(x.dtype)
            return x

        return f

    def compose_t(*fs):
        def f(name, i, x):
            for g in fs:
                x = g(name, i, x)
            return x

        return f

    variants = {
        "kv_int8_g32": qdq(8, 32),
        "kv_int8_g64": qdq(8, 64),
        "kv_4bit_g64": qdq(4, 64),
        "kv_4bit_g32": qdq(4, 32),
        "gdn_bf16": gdn_cast("bf16"),
        "gdn_int8": gdn_cast("int8"),
        "kv_int8_g32+gdn_bf16": compose_t(qdq(8, 32), gdn_cast("bf16")),
        "kv_4bit_g64+gdn_bf16": compose_t(qdq(4, 64), gdn_cast("bf16")),
    }
    toks = mx.array([cont])

    def run(cl):
        out = lm(toks, cache=cl)
        lg = out.logits if hasattr(out, "logits") else out
        lg = lg[0].astype(mx.float32)
        mx.eval(lg)
        return lg

    ref_cache = [clone_state(c) for c in cache]
    ref = run(ref_cache)
    del ref_cache
    # control: a second exact restore (the run-to-run floor)
    ctl = run([clone_state(c) for c in cache])
    res["control"] = kld_stats(ref, ctl)
    res["lossy"] = {}
    for name, tf in variants.items():
        t0 = time.perf_counter()
        cl = [clone_state(c, tf) for c in cache]
        mx.eval(*[x for c in cl for x in arrays_of(c)[1]])
        t_restore = time.perf_counter() - t0
        st = kld_stats(ref, run(cl))
        st["restore_s"] = round(t_restore, 3)
        res["lossy"][name] = st
        print(name, json.dumps(st), flush=True)
        del cl

    # bytes of the lossy forms (affine: codes + scale + bias bf16 per group)
    def qbytes(bits, group):
        tot = 0
        for x in kv_arrs:
            n = x.size
            tot += n * bits // 8 + (n // group) * 4
        return tot

    res["lossy_bytes"] = {
        "kv_bf16": sum(x.nbytes for x in kv_arrs),
        "kv_int8_g32": qbytes(8, 32),
        "kv_int8_g64": qbytes(8, 64),
        "kv_4bit_g64": qbytes(4, 64),
        "kv_4bit_g32": qbytes(4, 32),
        "gdn": sum(x.nbytes for x in gdn_arrs),
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print("done", a.out, flush=True)


if __name__ == "__main__":
    main()
