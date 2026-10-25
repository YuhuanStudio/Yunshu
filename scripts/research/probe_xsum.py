"""Cost of the lane matmul's separate group-sum launch (xsum) by input width (gpuq only).

Each lane projection call is preceded by one ``xsum`` kernel per distinct input: a thread per
(64-element group, row) summing its group serially. This times that kernel alone, chained 48
times in one eval, so the cost of removing it (summing inside the matmul) can be bounded.
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ks", type=int, nargs="+", default=[2560, 5120, 6144, 17408])
    p.add_argument("--rows", type=int, nargs="+", default=[1, 16, 32])
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
    import mlx.core as mx

    from yunshu_engine.kernels.tensorfold import lane_qmm as q

    q.ready()
    for k in a.ks:
        for m in a.rows:
            kg = k // 64
            mp = 16 * ((m + 15) // 16)
            x0 = (mx.random.normal((m, k)) * 0.5).astype(mx.bfloat16)
            mdims = q._mdims(m, mp)
            mx.eval(x0)

            def graph(x0=x0, k=k, kg=kg, mp=mp, mdims=mdims):
                outs, prev = [], None
                for _ in range(48):
                    xi = x0 if prev is None else mx.depends(x0, prev)
                    prev = q._kernel("xsum")(
                        inputs=[xi, mdims],
                        template=[("K", k), ("GS", 64)],
                        grid=(kg, mp, 1),
                        threadgroup=(min(kg, 256), 1, 1),
                        output_shapes=[(kg, mp)],
                        output_dtypes=[mx.float32],
                    )[0]
                    outs.append(prev)
                return outs

            for _ in range(2):
                mx.eval(graph())
            ts = []
            for _ in range(7):
                g = graph()
                mx.synchronize()
                t0 = time.perf_counter()
                mx.eval(g)
                ts.append((time.perf_counter() - t0) / 48)
            row = {"k": k, "rows": m, "us": statistics.median(ts) * 1e6}
            with a.out.open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
