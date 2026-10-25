"""Lane projection bandwidth by shape, row count and K-split (gpuq only).

Every projection shape of a model is run at the served row counts through the same
``LaneLinear._rows`` path the verify round uses. ``copies`` distinct weights are
chained (``mx.depends``) in one eval so the weights stream from DRAM rather than the
system cache; GB/s counts the weight + scale/bias bytes. For each shape the output
also names the kernel, tile, grid and threadgroup. Row invariance is checked
bitwise: row i of an M-row call equals row i of the 32-row call, for every split.

    probe_verify_proj.py --preset 27b --rows 1 8 16 32 --sks default 1 2 4 8 16 --out x.jsonl
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

PRESETS = {
    # name, n, k, bits, group, count per model
    "27b": [
        ("gdn.qkv", 10240, 5120, 4, 64, 48),
        ("gdn.z4", 6144, 5120, 4, 64, 27),
        ("gdn.z5", 6144, 5120, 5, 64, 21),
        ("gdn.b4", 48, 5120, 4, 64, 18),
        ("gdn.b5", 48, 5120, 5, 64, 30),
        ("gdn.out", 5120, 6144, 5, 64, 48),
        ("mlp.gate", 17408, 5120, 4, 64, 64),
        ("mlp.down4", 5120, 17408, 4, 64, 37),
        ("mlp.down5", 5120, 17408, 5, 64, 27),
        ("attn.q", 12288, 5120, 4, 64, 16),
        ("attn.k4", 1024, 5120, 4, 64, 14),
        ("attn.v4", 1024, 5120, 4, 64, 16),
        ("attn.o4", 5120, 6144, 4, 64, 9),
        ("attn.o5", 5120, 6144, 5, 64, 7),
    ],
    # Flash-Next oQ3e-mtp dense side (Jundot/Qwen3.8-Flash-Next-oQ3e-mtp): 8-bit g64 mixers and attention
    # (out_proj is 5-bit g128 and shared experts 8-bit g128: groups the lane kernel does not read)
    "oq3e": [
        ("gdn.qkv", 10240, 2560, 8, 64, 36),
        ("gdn.z", 6144, 2560, 8, 64, 36),
        ("gdn.b", 48, 2560, 8, 64, 36),
        ("gdn.a", 48, 2560, 8, 64, 36),
        ("attn.q", 12288, 2560, 8, 64, 12),
        ("attn.k", 512, 2560, 8, 64, 12),
        ("attn.v", 512, 2560, 8, 64, 12),
        ("attn.o", 2560, 6144, 8, 64, 12),
    ],
    # Flash-Next dense side (group 32, 4-bit): GDN/attention/shared-expert/router projections
    "flashnext": [
        ("gdn.qkv", 10240, 2560, 4, 32, 36),
        ("gdn.z", 6144, 2560, 4, 32, 36),
        ("gdn.b", 48, 2560, 4, 32, 36),
        ("gdn.out", 2560, 6144, 4, 32, 36),
        ("shared.gate", 640, 2560, 4, 32, 48),
        ("shared.down", 2560, 640, 4, 32, 48),
    ],
}


def row_block(m, n):
    """LaneLinear._rows: 16-row blocks for 17..48 rows on projections under 100K wide."""
    return 16 if 16 < m <= 48 and n < 100_000 else None


def geometry(n, k, bits, m, group, sk=None, tiled=True, nt=32):
    """Kernel name, tile and dispatch geometry lane_matmul uses for one call."""
    from yunshu_engine.kernels.tensorfold import lane_qmm as q

    mp = 16 * ((m + 15) // 16)
    block = row_block(m, n) or (mp if mp <= q.ROW_BLOCK else q.ROW_BLOCK)
    s = int(sk) if sk else q.split_k(n, k)
    if bits == 4:
        name = "main_tiled" if tiled else "main"
    else:
        name = "lowbit" if bits < 4 else "bytes"
        if (
            bits == 8
            and tiled
            and q._lane_schedule(bits, n, k, m, group, nt) == "direct8"
        ):
            name = "bytes8"
    tg = (-(-n // nt) * 32 * s, -(-mp // block), 1)
    return {
        "kernel": name,
        "rows_per_tg": block,
        "col_tile": nt,
        "sk": s,
        "grid": list(tg),
        "threadgroup": [32 * s, 1, 1],
        "threadgroups": (-(-n // nt)) * (-(-mp // block)),
        "simdgroups": (-(-n // nt)) * (-(-mp // block)) * s,
    }


def copies_for(nbytes, target_bytes):
    return max(2, min(64, -(-target_bytes // max(1, nbytes))))


def gbps(nbytes, copies, seconds):
    return nbytes * copies / seconds / 1e9


def summarize(samples):
    return {"ms_median": statistics.median(samples) * 1e3, "ms_min": min(samples) * 1e3}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preset", choices=sorted(PRESETS), required=True)
    p.add_argument("--rows", type=int, nargs="+", default=[1, 8, 16, 32])
    p.add_argument("--sks", nargs="+", default=["default"])
    p.add_argument("--only", nargs="*", default=None)
    p.add_argument(
        "--variants", nargs="+", default=["default"], choices=["default", "pipe"]
    )
    p.add_argument(
        "--shape", action="append", default=[], help="extra name:n:k:bits:group[:count]"
    )
    p.add_argument("--reps", type=int, default=7)
    p.add_argument("--target-mb", type=int, default=768)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
    import mlx.core as mx
    import mlx.nn as nn

    from yunshu_engine.kernels import lane_linear
    from yunshu_engine.kernels.tensorfold import lane_qmm as q

    if not q.ready():
        sys.exit("lane kernel not ready on this device")
    mx.random.seed(0)
    first = True
    extra = [
        (f[0], *map(int, f[1:5]), int(f[5]) if len(f) > 5 else 1)
        for f in (x.split(":") for x in a.shape)
    ]
    for name, n, k, bits, group, count in PRESETS[a.preset] + extra:
        if a.only and not any(name.startswith(o) for o in a.only):
            continue
        lin = nn.Linear(k, n, bias=False)
        lin.weight = (mx.random.normal((n, k)) * 0.02).astype(mx.bfloat16)
        base = nn.QuantizedLinear.from_linear(lin, group_size=group, bits=bits)
        base.scales, base.biases = (
            base.scales.astype(mx.bfloat16),
            base.biases.astype(mx.bfloat16),
        )
        proto = lane_linear.LaneLinear.from_quantized(base)
        nbytes = proto.weight.nbytes + proto.sbt.nbytes
        nc = copies_for(nbytes, a.target_mb * 2**20)
        mods = []
        for c in range(nc):
            m = lane_linear.LaneLinear.from_quantized(base)
            m.weight = mx.contiguous(
                proto.weight ^ mx.array(c * 2654435761 % 2**32, dtype=mx.uint32)
            )
            m.sbt = mx.contiguous(proto.sbt * 1)
            mods.append(m)
        mx.eval([m.weight for m in mods] + [m.sbt for m in mods])
        xs = {
            m_: (mx.random.normal((m_, k)) * 0.5).astype(mx.bfloat16) for m_ in a.rows
        }
        mx.eval(list(xs.values()))
        ref32 = None
        for m_ in sorted(set(a.rows) | {32}):
            if m_ not in xs:
                xs[m_] = (mx.random.normal((m_, k)) * 0.5).astype(mx.bfloat16)
        x32 = xs[32]
        ref_default = {}
        for sk in a.sks:
            skv = None if sk == "default" else int(sk)
            for mod in mods:
                mod.sk = skv
            for m_ in a.rows:
                x = x32[:m_]
                try:
                    mods[0]._rows(x)
                except ValueError as exc:
                    print(
                        json.dumps(
                            {
                                "name": name,
                                "rows": m_,
                                "sk": sk,
                                "skipped": str(exc)[:80],
                            }
                        ),
                        flush=True,
                    )
                    continue

                def graph():
                    outs, prev = [], None
                    for mod in mods:
                        xi = x if prev is None else mx.depends(x, prev)
                        prev = mod._rows(xi)
                        outs.append(prev)
                    return outs

                results, samples, graphs = {}, {}, {}
                for variant in a.variants:
                    q.PIPE[0] = variant == "pipe"
                    y_m = mods[0]._rows(x)
                    y_32 = mods[0]._rows(x32)
                    mx.eval(y_m, y_32)
                    results[variant] = (
                        y_m,
                        bool(mx.array_equal(y_m, y_32[:m_]).item()),
                    )
                    if skv is None and variant == "default":
                        ref_default[m_] = y_m
                    for _ in range(2):
                        mx.eval(graph())
                    graphs[variant] = None
                    samples[variant] = []
                for rep in range(a.reps):
                    order = a.variants if rep % 2 == 0 else a.variants[::-1]
                    for variant in order:
                        q.PIPE[0] = variant == "pipe"
                        g = graph()
                        mx.synchronize()
                        t0 = time.perf_counter()
                        mx.eval(g)
                        samples[variant].append((time.perf_counter() - t0) / nc)
                q.PIPE[0] = False
                for variant in a.variants:
                    y_m, invariant = results[variant]
                    ref = ref_default.get(m_)
                    s = summarize(samples[variant])
                    row = {
                        "preset": a.preset,
                        "name": name,
                        "n": n,
                        "k": k,
                        "bits": bits,
                        "group": group,
                        "rows": m_,
                        "sk_req": sk,
                        "variant": variant,
                        "copies": nc,
                        "bytes": nbytes,
                        "us_per_call": s["ms_median"] * 1e3,
                        "GBps": gbps(nbytes, 1, s["ms_median"] / 1e3),
                        "GBps_best": gbps(nbytes, 1, s["ms_min"] / 1e3),
                        "invariant_vs32": invariant,
                        "maxdiff_vs_default": None
                        if ref is None
                        else float(
                            mx.max(
                                mx.abs(y_m.astype(mx.float32) - ref.astype(mx.float32))
                            ).item()
                        ),
                        "bit_equal_default": None
                        if ref is None
                        else bool(mx.array_equal(y_m, ref).item()),
                        "per_round_ms_x_count": s["ms_median"] * count,
                        **geometry(n, k, bits, m_, group, skv),
                    }
                    if first:
                        row["device"] = str(mx.device_info().get("architecture"))
                        first = False
                    with a.out.open("a") as f:
                        f.write(json.dumps(row) + "\n")
                    print(
                        json.dumps(
                            {
                                k_: row[k_]
                                for k_ in (
                                    "name",
                                    "rows",
                                    "sk",
                                    "variant",
                                    "GBps",
                                    "invariant_vs32",
                                    "bit_equal_default",
                                )
                            }
                        ),
                        flush=True,
                    )
                    if not invariant:
                        sys.exit(
                            f"ROW INVARIANCE BROKEN {name} rows={m_} sk={sk} variant={variant}"
                        )


if __name__ == "__main__":
    main()
