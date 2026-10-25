"""The split's GPU down projection reads the leading K groups of a tiled weight in place (kgw)
and must equal the copy it replaces bit for bit. Also reports the copy's size."""

from __future__ import annotations

import sys

import mlx.core as mx
import numpy as np

from yunshu_engine.ane import ffn_split as fs
from yunshu_engine.kernels import nax_prefill
from yunshu_engine.kernels.tensorfold import lane_qmm


def main():
    if not nax_prefill.enable():
        raise SystemExit("NAX prefill unavailable")
    rng = np.random.default_rng(3)
    for bits in (4, 5):
        out, inp, gs, lead, rows = 2560, 17408, 64, 17408 - 4352, 192
        w = rng.normal(0, 0.05, (out, inp)).astype(np.float32)
        wq, sc, bi = mx.quantize(
            mx.array(w).astype(mx.bfloat16), group_size=gs, bits=bits
        )
        wt = lane_qmm.tile_weight(wq, 32, gs, bits=bits)
        sbt = lane_qmm.pack_scales(sc, bi)
        x = mx.array(rng.normal(0, 1, (rows, lead)).astype(np.float32)).astype(
            mx.bfloat16
        )
        mx.eval(wt, sbt, x)
        p = fs.QProj(wt, None, None, bits, gs, out, inp, True, sbt)
        words = gs * bits // 32
        copy_w = mx.contiguous(
            wt.reshape(out // 32, inp // gs, 32, words)[:, : lead // gs]
        ).reshape(out, lead // gs * words)
        old = fs._nax(x, copy_w, sbt[: lead // gs], bits)
        new = fs._qmm(x, p, fs._slices(p, lead, True))
        mx.eval(old, new)
        same = bool(mx.array_equal(old, new).item())
        print(
            f"bits={bits} identical={same} copy_mib={copy_w.nbytes / 2**20:.1f}",
            flush=True,
        )
        if not same:
            sys.exit(1)


if __name__ == "__main__":
    main()
