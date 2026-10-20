"""Lossless tile-geometry sweep of the NAX lane QMM used by 27B cold prefill (GPU: via gpuq only).

    prefillspeed_gemm.py --out F.jsonl [--m 2048] [--reps 3] [--variants bits,bm,bn,wm,wn ...] [--dry-run]

Every variant keeps BK=64 and the native K loop; only BM/BN and the simdgroup grid (WM, WN) change, so each
output element sees the same accumulation order. Each variant is compared bit-for-bit with the production
tile (BM128, BN64, WM2, WN2) before it is timed; a mismatch is recorded and the variant is not timed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

SHAPES = [  # (K, N, bits): 27B oQ4e projections
    (5120, 17408, 4),
    (17408, 5120, 4),
    (17408, 5120, 5),
    (5120, 10240, 4),
    (5120, 6144, 5),
    (6144, 5120, 5),
    (5120, 12288, 4),
]
BASE = (128, 64, 2, 2)
GRID = [
    (128, 64, 2, 2),
    (64, 64, 2, 2),
    (128, 64, 4, 1),
    (128, 64, 1, 4),
    (128, 64, 2, 4),
    (256, 64, 4, 2),
    (256, 64, 8, 1),
    (128, 32, 4, 1),
    (128, 128, 4, 2),
    (64, 128, 1, 4),
    (64, 64, 1, 2),
    (64, 64, 2, 1),
]


def parse_variants(items: list[str]) -> list[tuple[int, int, int, int]]:
    out = []
    for item in items:
        parts = tuple(int(x) for x in item.split(","))
        if (
            len(parts) != 4
            or parts[0] % 16
            or parts[1] % 16
            or parts[0] % parts[2]
            or parts[1] % parts[3]
        ):
            raise ValueError(f"bad variant {item}: want bm,bn,wm,wn")
        out.append(parts)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--m", type=int, default=2048)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument(
        "--variants", nargs="*", default=[",".join(map(str, g)) for g in GRID]
    )
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)
    variants = parse_variants(a.variants)
    if BASE not in variants:
        variants.insert(0, BASE)
    if a.dry_run:
        print(json.dumps({"variants": len(variants), "shapes": len(SHAPES)}))
        return 0
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
    import mlx.core as mx

    from yunshu_engine.kernels import nax_prefill
    from yunshu_engine.kernels.tensorfold import lane_qmm

    if not nax_prefill.enable():
        print("nax_prefill unsupported on this device", file=sys.stderr)
        return 1
    a.out.parent.mkdir(parents=True, exist_ok=True)
    mx.random.seed(3)
    n = 0
    with a.out.open("w") as f:

        def emit(rec: dict) -> None:
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print(json.dumps(rec), flush=True)

        emit(
            dict(
                type="engagement",
                device=mx.device_info().get("device_name"),
                mlx=mx.__version__,
            )
        )
        for k, nn, bits in SHAPES:
            x = (mx.random.normal((a.m, k)) * 0.5).astype(mx.bfloat16)
            w = (mx.random.normal((nn, k)) * 0.02).astype(mx.bfloat16)
            q, s, b = mx.quantize(w, group_size=64, bits=bits)
            qt = lane_qmm.tile_weight(q, bits=bits)
            sbt = lane_qmm.pack_scales(s, b)
            mx.eval(x, qt, sbt)
            fns = {}
            for bm, bn, wm, wn in variants:
                if a.m % bm or nn % bn:
                    continue
                fns[(bm, bn, wm, wn)] = nax_prefill.make(
                    mx, x, qt, s, b, bits=bits, bm=bm, bn=bn, wm=wm, wn=wn, sbt=sbt
                )
            ref = fns[BASE]()
            mx.eval(ref)
            ok = {}
            for key, fn in fns.items():
                out = fn()
                mx.eval(out)
                ok[key] = bool(mx.array_equal(ref, out).item())
                if not ok[key]:
                    emit(dict(type="mismatch", k=k, n=nn, bits=bits, variant=list(key)))
            times = {key: [] for key in fns if ok[key]}
            for rep in range(a.reps):
                for key in list(times) if rep % 2 == 0 else list(reversed(list(times))):
                    fn = fns[key]
                    mx.eval(fn())
                    mx.synchronize()
                    t0 = time.perf_counter()
                    mx.eval([fn() for _ in range(10)])
                    times[key].append((time.perf_counter() - t0) / 10)
            for key, ts in times.items():
                med = sorted(ts)[len(ts) // 2]
                n += 1
                emit(
                    dict(
                        type="gemm",
                        m=a.m,
                        k=k,
                        n=nn,
                        bits=bits,
                        variant=list(key),
                        tflops=round(2 * a.m * k * nn / med / 1e12, 2),
                    )
                )
            del fns, x, w, qt, sbt
            mx.clear_cache()
        if not n:
            return 1
        emit(dict(type="complete", success=True, records=n))
    return 0


if __name__ == "__main__":
    sys.exit(main())
