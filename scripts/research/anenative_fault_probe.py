"""Instrumented native fault oracle; expected failures must never hang a GPU wait."""

from __future__ import annotations

import argparse
import ctypes as c
import json
import time
from pathlib import Path

from yunshu_engine.ane import native
from yunshu_engine.ane.native.mil import scale_program


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--library", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument(
        "--fault", choices=["selector", "objc", "async", "stalled"], required=True
    )
    return p


def main():
    a = parser().parse_args()
    import mlx.core as mx

    lib = native._configure_library(c.CDLL(str(a.library)), with_mlx=True)
    lib.ys_ane_test_arm.argtypes = [c.c_char_p]
    lib.ys_ane_test_arm.restype = c.c_int
    assert lib.ys_ane_test_arm(a.fault.encode())
    native._library = lambda *args: lib
    shape = (64, 128)
    mil, blob = scale_program(*shape)
    session, reason = native.NativeSession.try_create(
        mil, blob, [shape], shape, a.cache, mlx_root=Path(mx.__file__).parent
    )
    if a.fault == "selector":
        assert session is None and "changed" in reason
        print(
            json.dumps(
                {
                    "fault": a.fault,
                    "unavailable": True,
                    "reason": reason,
                    "complete": True,
                }
            ),
            flush=True,
        )
        return 0
    assert session is not None, reason
    other, reason = native.NativeSession.try_create(
        mil, blob, [shape], shape, a.cache, mlx_root=Path(mx.__file__).parent
    )
    assert other is not None, reason
    with session, other:
        inp, reason = session.try_alias(0)
        assert inp is not None, reason
        out, reason = session.try_alias(1)
        assert out is not None, reason
        writer = mx.fast.metal_kernel(
            name="anenative_fault_writer",
            input_names=["surface"],
            output_names=["marker"],
            source="""uint i = thread_position_in_grid.x;
if (i < 8192) const_cast<device half*>(surface)[i] = half(1);
if (i == 0) marker[0] = 1;
""",
        )
        marker = writer(
            inputs=[inp],
            grid=(8192, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(1,)],
            output_dtypes=[mx.uint32],
        )[0]
        mx.async_eval(marker)
        start = time.perf_counter()
        ticket, reason = session.try_submit_mlx()
        assert ticket is not None, reason
        with ticket:
            assert ticket.wait_mlx()
            consumer = out.astype(mx.float32)
            mx.eval(consumer)
            assert ticket.finish() is False, "fault unexpectedly accepted"
            elapsed = time.perf_counter() - start
            assert elapsed < 4.5, f"GPU wait was not released: {elapsed}s"
            assert lib.ys_ane_available() == 0
            assert other.run() is None, (
                "another session ignored the retired native service"
            )
            print(
                json.dumps(
                    {
                        "fault": a.fault,
                        "complete": True,
                        "unavailable": True,
                        "gpu_wait_released": True,
                        "elapsed_s": elapsed,
                        "reason": ticket.reason,
                        "other_session_declines": True,
                    }
                ),
                flush=True,
            )
        del inp, out, marker, consumer, ticket
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
