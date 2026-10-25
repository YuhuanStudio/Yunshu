"""Physical IOSurface/MLX alias proof through public DLPack (gpuq only).

CPU synchronization deliberately establishes the reference ordering first.
Shared-event synchronization is a separate next gate; this is not a latency win.
All aliases are destroyed before closing the IOSurface-owning session.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np

from yunshu_engine.ane.native import NativeSession
from yunshu_engine.ane.native.mil import scale_program


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--height", type=int, default=64)
    p.add_argument("--width", type=int, default=128)
    return p


def main():
    a = parser().parse_args()
    if min(a.height, a.width) < 1:
        raise ValueError("positive dimensions required")
    mil, blob = scale_program(a.height, a.width)
    session, reason = NativeSession.try_create(
        mil, blob, [(a.height, a.width)], (a.height, a.width), a.cache
    )
    if session is None:
        raise RuntimeError(reason)
    import mlx.core as mx

    with session:
        # Private view only in this bounded research proof. Do not expose it to serving.
        source = session._view(0)
        target = session._view(1)
        source[:] = 1
        inp = mx.from_dlpack(source, copy=False)
        out = mx.from_dlpack(target, copy=False)
        assert (
            np.asarray(inp).__array_interface__["data"][0]
            == source.__array_interface__["data"][0]
        )
        assert (
            np.asarray(out).__array_interface__["data"][0]
            == target.__array_interface__["data"][0]
        )
        source[0, 0] = 3
        assert float(np.asarray(inp)[0, 0]) == 3
        writer = mx.fast.metal_kernel(
            name="anenative_alias_writer",
            input_names=["values", "surface"],
            output_names=["marker"],
            source="""uint i = thread_position_in_grid.x;
if (i < N) const_cast<device half*>(surface)[i] = half(values[i]);
if (i == 0) marker[0] = 1;
""",
        )
        values = mx.full((a.height, a.width), 2, dtype=mx.float16)
        marker = writer(
            inputs=[values, inp],
            template=[("N", a.height * a.width)],
            grid=(a.height * a.width, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(1,)],
            output_dtypes=[mx.uint32],
        )[0]
        mx.eval(marker)
        assert np.all(source == 2), "GPU did not write IOSurface input"
        if session.run() is None:
            raise RuntimeError(session.reason)
        summed = mx.sum(out.astype(mx.float32))
        mx.eval(summed)
        assert float(summed.item()) == a.height * a.width * 4, (
            "GPU did not consume ANE IOSurface output"
        )
        print(
            json.dumps(
                {
                    "complete": True,
                    "input_alias": True,
                    "output_alias": True,
                    "copy_false": True,
                    "gpu_producer": True,
                    "ane_scale": 2,
                    "gpu_consumer_sum": float(summed.item()),
                    "cpu_sync_reference": True,
                    "shape": [a.height, a.width],
                }
            ),
            flush=True,
        )
        del inp, out, marker, values, summed, source, target
        gc.collect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
