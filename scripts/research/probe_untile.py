"""Cost of a prefill step's weight untiling: every LaneLinear.stock() of the
model once, evaluated (the copy a prefill step makes per layer)."""

import sys
import time
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sweep_round_driver import load  # noqa: E402


def main():
    model, _, _, _ = load(sys.argv[1], False)
    from yunshu_engine.kernels.lane_linear import LaneLinear

    lm = model.language_model
    lins = [m for _, m in lm.named_modules() if isinstance(m, LaneLinear)]
    for rep in range(3):
        mx.synchronize()
        t0 = time.perf_counter()
        nbytes = 0
        for lin in lins:
            w, s, b = lin.stock()
            mx.eval(w, s, b)
            nbytes += w.nbytes + s.nbytes + b.nbytes
        mx.synchronize()
        dt = time.perf_counter() - t0
        print(
            f"rep {rep}: {len(lins)} projections, {nbytes / 2**30:.2f} GiB stock, {dt * 1e3:.0f} ms",
            flush=True,
        )


main()
