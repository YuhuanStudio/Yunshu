"""Engine-path decode step vs the bandwidth roofline, and per-projection kernel efficiency.

Loads the 27B, times (1) stock mlx-vlm one-row decode, (2) decode after Yunshu's verify /
batch-invariant / NAX-packed kernels are installed (1 row, invariant active), (3) the same
with the invariant path switched off. Then, for every distinct projection shape of the
model (type, bits, K, N), times one call at 1/2/4/8/16 rows cycling through all layers'
weights (so nothing is SLC-resident) and reports the achieved weight GB/s.

    PYTHONPATH=python:scripts/research/hw python scripts/research/hw/decode_step_engine.py $M
"""

import argparse
import collections
import time

import mlx.core as mx
import mlx.nn as nn
from _common import Out
from decode_step_model import decode_steps, pct
from mlx_vlm import load


def nbytes(mod):
    n = 0
    for _, v in nn.utils.tree_flatten(mod.parameters()):
        n += v.nbytes
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--steps", type=int, default=60)
    a = ap.parse_args()
    out = Out("decode_step_engine")

    model, _ = load(a.model)
    lm = model.language_model
    mx.eval(lm.parameters())
    cache = lm.make_cache()
    ids = mx.array([[(i * 7919) % 200000 + 1000 for i in range(512)]])
    for s in range(0, 512, 256):
        lm(ids[:, s : s + 256], cache=cache)
        mx.eval([c.state for c in cache])
    y = mx.array([[1234]])

    decode_steps(lm, cache, y, 10)
    t, y = decode_steps(lm, cache, y, a.steps)
    weights_gb = sum(v.nbytes for _, v in nn.utils.tree_flatten(lm.parameters())) / 1e9
    out(
        kind="decode_stock",
        ms_median=round(pct(t, 0.5) * 1e3, 2),
        weight_GB=round(weights_gb, 2),
        GBps=round(weights_gb / pct(t, 0.5), 1),
    )

    from yunshu_engine.kernels.batch_invariant import install, set_active
    from yunshu_engine.kernels.omlx import apply, is_nax_available

    apply(row_exact=False)
    info = install(lm, model=model, packed=is_nax_available())
    out(kind="installed", **info)
    cache = lm.make_cache()
    for s in range(0, 512, 256):
        lm(ids[:, s : s + 256], cache=cache)
        mx.eval([c.state for c in cache])
    for active in (True, False):
        set_active(active)
        decode_steps(lm, cache, y, 10)
        t, y = decode_steps(lm, cache, y, a.steps)
        out(
            kind="decode_installed",
            invariant_active=active,
            ms_median=round(pct(t, 0.5) * 1e3, 2),
            ms_p95=round(pct(t, 0.95) * 1e3, 2),
        )
    set_active(True)

    # per-projection efficiency at 1..16 rows
    groups = collections.defaultdict(list)
    for name, m in lm.named_modules():
        if type(m).__name__ in ("QuantizedLinear", "PackedLinear") and "layers" in name:
            if hasattr(m, "input_dims"):
                k, n = m.input_dims, m.output_dims
            else:
                n, k = m.scales.shape[0], m.scales.shape[1] * m.group_size
            groups[(type(m).__name__, getattr(m, "bits", 4), k, n)].append(m)
    mixes = collections.Counter()
    for (cls, bits, k, n), mods in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        mixes[cls] += len(mods)
        use = mods[:: max(1, len(mods) // 24)][:24]
        gb = sum(nbytes(m) for m in use) / len(use)
        row = {
            "cls": cls,
            "bits": bits,
            "K": k,
            "N": n,
            "count": len(mods),
            "MB_each": round(gb / 1e6, 1),
        }
        for rows in (1, 2, 4, 8, 16):
            x = mx.random.normal((1, rows, k)).astype(mx.bfloat16)
            mx.eval(x)
            for _ in range(2):
                mx.eval([m(x) for m in use])
            ts = []
            for _ in range(6):
                t0 = time.perf_counter()
                mx.eval([m(x) for m in use])
                ts.append((time.perf_counter() - t0) / len(use))
            ts.sort()
            row[f"us_{rows}"] = round(ts[len(ts) // 2] * 1e6, 1)
            row[f"GBps_{rows}"] = round(gb / ts[len(ts) // 2] / 1e9, 0)
        out(kind="projection", **row)
    out(kind="projection_mix", **dict(mixes))


if __name__ == "__main__":
    main()
