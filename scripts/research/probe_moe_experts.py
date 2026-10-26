# ruff: noqa: E731
"""Flash-Next MoE expert kernels (router, gate|up, down) at the served row counts (gpuq only).

Random quantized experts at the real shapes (512 experts, top-10, D=2560, I=640, shared
expert I=640) with random router logits, so each row picks its own experts. GB/s counts
the bytes of the *distinct* experts picked (weights + scales + biases) plus the shared
expert. Every stage is chained 24 times in one eval (``mx.depends``), as a layer stack
runs, so the per-eval sync (~100 us) is not charged to a 20-100 us kernel. Row
invariance is checked bitwise against the 16-row call; ``--sweep`` tries launch
geometries whose arithmetic must stay bit-equal to the default.

    probe_moe_experts.py --bits 3 --group 64 --shared-bits 8 --shared-group 128 --rows 1 2 4 16 --out x.jsonl
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace


def picked_bytes(unique_experts, per_expert_bytes, shared_bytes):
    return unique_experts * per_expert_bytes + shared_bytes


def bench(mx, call, x0, n=24, reps=7):
    """Median seconds per call of ``call(x)`` chained n times in one eval."""

    def graph():
        outs, prev = [], None
        for _ in range(n):
            xi = (
                x0
                if prev is None
                else mx.depends(x0, prev[0] if isinstance(prev, tuple) else prev)
            )
            prev = call(xi)
            outs.append(prev)
        return outs

    for _ in range(2):
        mx.eval(graph())
    ts = []
    for _ in range(reps):
        g = graph()
        mx.synchronize()
        t0 = time.perf_counter()
        mx.eval(g)
        ts.append((time.perf_counter() - t0) / n)
    return statistics.median(ts)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bits", type=int, default=4)
    p.add_argument("--group", type=int, default=32)
    p.add_argument("--shared-bits", type=int, default=4)
    p.add_argument("--shared-group", type=int, default=32)
    p.add_argument("--experts", type=int, default=512)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--dims", type=int, default=2560)
    p.add_argument("--inter", type=int, default=640)
    p.add_argument("--rows", type=int, nargs="+", default=[1, 8, 16])
    p.add_argument(
        "--sweep",
        action="store_true",
        help="launch geometry sweep (outputs must stay bit-equal)",
    )
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
    import mlx.core as mx

    from yunshu_engine.kernels.tf_flash_next import experts

    mx.random.seed(1)

    def qlin(n_exp, n, k, bits, group):
        ws, ss, bs = [], [], []
        for lo in range(0, max(n_exp, 1), 64):
            cnt = min(64, max(n_exp, 1) - lo)
            shape = (cnt, n, k) if n_exp else (n, k)
            w = (mx.random.normal(shape) * 0.02).astype(mx.bfloat16)
            wq, s, b = mx.quantize(w, group_size=group, bits=bits)
            mx.eval(wq, s, b)
            (
                ws.append(wq),
                ss.append(s.astype(mx.bfloat16)),
                bs.append(b.astype(mx.bfloat16)),
            )
            if not n_exp:
                break

        def cat(xs):
            return xs[0] if len(xs) == 1 else mx.concatenate(xs)

        return SimpleNamespace(
            weight=cat(ws), scales=cat(ss), biases=cat(bs), bits=bits, group_size=group
        )

    D, I, E, K = a.dims, a.inter, a.experts, a.top_k
    gate, up = qlin(E, I, D, a.bits, a.group), qlin(E, I, D, a.bits, a.group)
    down = qlin(E, D, I, a.bits, a.group)
    sg, su = (
        qlin(0, I, D, a.shared_bits, a.shared_group),
        qlin(0, I, D, a.shared_bits, a.shared_group),
    )
    sd = qlin(0, D, I, a.shared_bits, a.shared_group)
    router_w = (mx.random.normal((E + 1, D)) * 0.05).astype(mx.bfloat16)

    def nb(lin, share=1):
        return (lin.weight.nbytes + lin.scales.nbytes + lin.biases.nbytes) // share

    per_gu, per_d = nb(gate, E) + nb(up, E), nb(down, E)
    shared_gu, shared_d = nb(sg) + nb(su), nb(sd)
    full = {
        m: (mx.random.normal((m, D)) * 0.5).astype(mx.bfloat16) for m in {16, *a.rows}
    }
    mx.eval(list(full.values()), router_w)

    def run(xx, **kw):
        logits = experts.router(xx, router_w)
        act, picks, wts = experts.expert_gateup(
            xx, logits, K, E, gate, up, shared=(sg, su), **kw
        )
        return logits, act, picks, experts.expert_down_y(act, picks, down, sd)

    for m in a.rows:
        x = full[16][:m] if m <= 16 else full[m]
        logits, act, picks, y = run(x)
        mx.eval(logits, act, picks, y)
        y16 = run(full[16])[3]
        mx.eval(y16)
        invariant = m > 16 or bool(mx.array_equal(y, y16[:m]).item())
        uniq = len(set(picks.reshape(-1).tolist()))
        row = {
            "bits": a.bits,
            "group": a.group,
            "rows": m,
            "unique_experts": uniq,
            "invariant_vs16": invariant,
        }
        stages = (
            ("router", lambda xx: experts.router(xx, router_w), router_w.nbytes),
            (
                "gateup",
                lambda xx: experts.expert_gateup(
                    xx, logits, K, E, gate, up, shared=(sg, su)
                ),
                picked_bytes(uniq, per_gu, shared_gu),
            ),
        )
        for stage, fn, byt in stages:
            t = bench(mx, fn, x)
            row[stage] = {"us": t * 1e6, "GBps": byt / t / 1e9, "bytes": byt}
        t = bench(mx, lambda aa: experts.expert_down_y(aa, picks, down, sd), act)
        byt = picked_bytes(uniq, per_d, shared_d)
        row["down"] = {"us": t * 1e6, "GBps": byt / t / 1e9, "bytes": byt}
        if a.sweep:
            sw = {"router": [], "gateup": [], "down": []}
            for threads, split in (
                (128, False),
                (256, False),
                (512, False),
                (256, True),
            ):
                fn = lambda xx, t=threads, s=split: experts.router(
                    xx, router_w, threads=t, split=s
                )  # noqa: E731
                sw["router"].append(
                    {
                        "threads": threads,
                        "split": split,
                        "us": bench(mx, fn, x) * 1e6,
                        "equal": bool(mx.array_equal(fn(x), logits).item()),
                    }
                )
            for rps in (1, 2, 4, 8):
                for sgs in (1, 2, 4, 8):
                    fn = lambda xx, r=rps, g=sgs: experts.expert_gateup(  # noqa: E731
                        xx,
                        logits,
                        K,
                        E,
                        gate,
                        up,
                        shared=(sg, su),
                        rows_per_simdgroup=r,
                        simdgroups=g,
                    )
                    try:
                        eq = all(
                            bool(mx.array_equal(u, v).item())
                            for u, v in zip(fn(x), (act, picks, None), strict=False)
                            if v is not None
                        )
                        t = bench(mx, fn, x)
                        sw["gateup"].append(
                            {
                                "rps": rps,
                                "sg": sgs,
                                "us": t * 1e6,
                                "equal": eq,
                                "GBps": picked_bytes(uniq, per_gu, shared_gu) / t / 1e9,
                            }
                        )
                    except Exception as exc:  # noqa: BLE001
                        sw["gateup"].append(
                            {"rps": rps, "sg": sgs, "error": str(exc)[:60]}
                        )
            for dps in (8, 16, 32, 64):
                for sgs in (1, 2, 4):
                    fn = lambda aa, g=sgs, d=dps: experts.expert_down_y(  # noqa: E731
                        aa, picks, down, sd, simdgroups=g, dims_per_simdgroup=d
                    )
                    try:
                        eq = bool(mx.array_equal(fn(act), y).item())
                        t = bench(mx, fn, act)
                        sw["down"].append(
                            {
                                "dps": dps,
                                "sg": sgs,
                                "us": t * 1e6,
                                "equal": eq,
                                "GBps": picked_bytes(uniq, per_d, shared_d) / t / 1e9,
                            }
                        )
                    except Exception as exc:  # noqa: BLE001
                        sw["down"].append(
                            {"dps": dps, "sg": sgs, "error": str(exc)[:60]}
                        )
            row["sweep"] = sw
        with a.out.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
        if not invariant:
            sys.exit("ROW INVARIANCE BROKEN")


if __name__ == "__main__":
    main()
