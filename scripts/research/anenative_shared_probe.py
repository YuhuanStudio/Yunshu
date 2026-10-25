"""No-copy GPU→ANE→GPU proof with native shared events (gpuq only)."""

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
    p.add_argument("--loops", type=int, default=8)
    return p


def main():
    a = parser().parse_args()
    if a.loops < 2:
        raise ValueError("at least two reuse cycles required")
    import mlx.core as mx

    shape = (64, 128)
    n = 64 * 128
    mil, blob = scale_program(*shape)
    session, reason = NativeSession.try_create(
        mil, blob, [shape], shape, a.cache, mlx_root=Path(mx.__file__).parent
    )
    if session is None:
        raise RuntimeError(reason)
    with session, mx.stream(mx.new_stream(mx.gpu)):
        inp, reason = session.try_alias(0)
        if inp is None:
            raise RuntimeError(reason)
        out, reason = session.try_alias(1)
        if out is None:
            raise RuntimeError(reason)
        writer = mx.fast.metal_kernel(
            name="anenative_shared_writer",
            input_names=["values", "surface"],
            output_names=["marker"],
            source="""uint i = thread_position_in_grid.x;
if (i < N) const_cast<device half*>(surface)[i] = half(values[i]);
if (i == 0) marker[0] = 1;
""",
        )
        previous = None
        heavy_left = mx.ones((4096, 4096), dtype=mx.float16)
        heavy_right = mx.ones((4096, 4096), dtype=mx.float16)
        for index in range(a.loops):
            values = ((mx.arange(n) % 17).reshape(shape).astype(mx.float16) + index) / 8
            # Delay the producer on a non-default stream. A signal mistakenly
            # inserted on another queue would let ANE read stale input here.
            guard = mx.sum((heavy_left @ heavy_right).astype(mx.float32)) / (2.0**36)
            values = values * guard
            marker = writer(
                inputs=[values, inp],
                template=[("N", n)],
                grid=(n, 1, 1),
                threadgroup=(256, 1, 1),
                output_shapes=[(1,)],
                output_dtypes=[mx.uint32],
            )[0]
            mx.async_eval(marker)
            ticket, reason = session.try_submit_mlx()
            if ticket is None:
                raise RuntimeError(reason)
            try:
                # An old finalized ticket must not clear this new ticket's busy state.
                if previous is not None:
                    assert previous.finish()
                    previous.close()
                    assert session._lib.ys_ane_busy(session._handle) == 1
                if not ticket.wait_mlx():
                    raise RuntimeError(ticket.reason)
                consumer = out.astype(mx.float32)
                mx.eval(consumer)
                if not ticket.finish():
                    raise RuntimeError(ticket.reason)
                expected = np.array((values * 2).astype(mx.float32))
                actual = np.array(consumer)
                assert np.array_equal(actual, expected), (
                    f"stale/incorrect shared surface on cycle {index}"
                )
            except BaseException:
                ticket.close()
                if previous is not None:
                    previous.close()
                raise
            previous = ticket
        previous.close()
        session.close()
        # The aliases own retained native references after parent close.
        after = out.astype(mx.float32)
        mx.eval(after)
        assert np.array_equal(np.array(after), expected)
        print(
            json.dumps(
                {
                    "complete": True,
                    "loops": a.loops,
                    "copy_false": True,
                    "producer_cpu_wait": False,
                    "ane_cpu_wait": False,
                    "shared_events": True,
                    "nondefault_stream_with_delayed_producer": True,
                    "final_consumer_eval": True,
                    "alias_lifetime_after_close": True,
                }
            ),
            flush=True,
        )
        del inp, out, values, marker, consumer, after, ticket, session
        gc.collect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
