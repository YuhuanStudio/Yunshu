"""Flash-Next MoE expert kernels (router, gate|up, down) at the served row counts (gpuq only).

Random quantized experts at the real shapes (512 experts, top-10, D=2560, I=640, shared
expert I=640) with random router logits, so each row picks its own experts. GB/s counts
the bytes of the *distinct* experts picked (weights + scales + biases) plus the shared
expert; row invariance is checked bitwise against the 16-row call.

    probe_moe_experts.py --bits 2 --group 32 --rows 1 8 16 --out x.jsonl
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
    p.add_argument("--reps", type=int, default=15)
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
            w = (
                (mx.random.normal((cnt, n, k)) * 0.02).astype(mx.bfloat16)
                if n_exp
                else (mx.random.normal((n, k)) * 0.02).astype(mx.bfloat16)
            )
            wq, s, b = mx.quantize(w, group_size=group, bits=bits)
            mx.eval(wq, s, b)
            (
                ws.append(wq),
                ss.append(s.astype(mx.bfloat16)),
                bs.append(b.astype(mx.bfloat16)),
            )
            if not n_exp:
                break
        cat = lambda xs: xs[0] if len(xs) == 1 else mx.concatenate(xs)
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

    per_exp_gu = nb(gate, E) + nb(up, E)
    per_exp_d = nb(down, E)
    shared_gu, shared_d = nb(sg) + nb(su), nb(sd)
    full = {
        m: (mx.random.normal((m, D)) * 0.5).astype(mx.bfloat16) for m in (16, *a.rows)
    }
    mx.eval(list(full.values()), router_w)
    for m in a.rows:
        x = full[16][:m] if m <= 16 else full[m]

        def run(xx):
            logits = experts.router(xx, router_w)
            act, picks, wts = experts.expert_gateup(
                xx, logits, K, E, gate, up, shared=(sg, su)
            )
            y = experts.expert_down_y(act, picks, down, sd)
            return logits, act, picks, y

        logits, act, picks, y = run(x)
        mx.eval(logits, act, picks, y)
        _, _, _, y16 = run(full[16])
        mx.eval(y16)
        invariant = bool(mx.array_equal(y, y16[:m]).item())
        uniq = len(set(picks.reshape(-1).tolist()))
        row = {
            "bits": a.bits,
            "group": a.group,
            "rows": m,
            "unique_experts": uniq,
            "invariant_vs16": invariant,
        }
        for stage, fn, byt in (
            ("router", lambda: experts.router(x, router_w), router_w.nbytes),
            (
                "gateup",
                lambda: experts.expert_gateup(
                    x, logits, K, E, gate, up, shared=(sg, su)
                ),
                picked_bytes(uniq, per_exp_gu, shared_gu),
            ),
            (
                "down",
                lambda: experts.expert_down_y(act, picks, down, sd),
                picked_bytes(uniq, per_exp_d, shared_d),
            ),
        ):
            for _ in range(3):
                mx.eval(fn())
            ts = []
            for _ in range(a.reps):
                mx.synchronize()
                t0 = time.perf_counter()
                mx.eval(fn())
                ts.append(time.perf_counter() - t0)
            med = statistics.median(ts)
            row[stage] = {"us": med * 1e6, "GBps": byt / med / 1e9, "bytes": byt}
        a.out.open("a").write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
        if not invariant:
            sys.exit("ROW INVARIANCE BROKEN")


if __name__ == "__main__":
    main()
