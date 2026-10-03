"""Research-only cold-prefill arms. Call install on the idle MLX thread."""

from __future__ import annotations

import json
from collections import Counter

_original = None
_maximum = None
calls = Counter()
observed_rows = Counter()


def install(arm):
    import mlx.core as mx
    from nax_qmm_tiles import make

    from yunshu_engine.kernels import lane_linear
    from yunshu_engine.kernels.tensorfold import lane_qmm

    global _original, _maximum
    if _original is None:
        _original = lane_linear.LaneLinear.__call__
        _maximum = lane_qmm.MAX_ROWS
    lane_linear.LaneLinear.__call__ = _original
    lane_qmm.MAX_ROWS = _maximum
    calls.clear()
    observed_rows.clear()
    if arm == "base":
        return
    if arm not in ("tile128", "lane64", "lane128", "narrow", "combo"):
        raise ValueError(arm)
    if arm in ("narrow", "combo"):
        lane_qmm.MAX_ROWS = 8192

    def run(self, x):
        lead, dtype = x.shape[:-1], x.dtype
        m = 1
        for size in lead:
            m *= size
        if m > 512:
            observed_rows[m] += 1
        common = (
            lane_linear.STOCK_ROWS
            and m > lane_linear.STOCK_ROWS
            and m % 128 == 0
            and m <= 8192
            and self.group_size == 64
        )
        narrow = (
            common
            and self.output_dims < lane_linear.NARROW
            and arm in ("narrow", "combo")
        )
        matrix = (
            common
            and arm in ("tile128", "lane64", "lane128", "combo")
            and lane_linear.NARROW <= self.output_dims < 100_000
            and self.output_dims % 64 == 0
            and self.bits in (4, 5, 8)
        )
        if not narrow and not matrix:
            return _original(self, x)
        x2 = x.reshape(-1, self.input_dims).astype(mx.bfloat16)
        if narrow:
            y = self._rows(x2)
        else:
            bm = 64 if arm == "lane64" or (m > 4096 and self.input_dims > 8192) else 128
            if arm == "tile128":
                w, s, b = self.stock()
                fn = make(mx, x2, w, s, b, bits=self.bits, bm=bm, bn=64)
            else:
                if not self.tiled:
                    raise RuntimeError("matrix lane loader requires tiled weights")
                fn = make(
                    mx,
                    x2,
                    self.weight,
                    None,
                    None,
                    bits=self.bits,
                    bm=bm,
                    bn=64,
                    sbt=self.sbt,
                )
            y = fn()
        if not calls:
            print(
                "NAX_DISPATCH_ENGAGED "
                + json.dumps(
                    dict(
                        arm=arm,
                        calls=1,
                        rows=m,
                        k=self.input_dims,
                        n=self.output_dims,
                        bits=self.bits,
                    )
                ),
                flush=True,
            )
        calls[(arm, self.bits, m, self.input_dims, self.output_dims)] += 1
        if "bias" in self:
            y = y + self["bias"]
        return y.reshape(*lead, self.output_dims).astype(dtype)

    lane_linear.LaneLinear.__call__ = run
