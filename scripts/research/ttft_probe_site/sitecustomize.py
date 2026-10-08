"""Stack sampler for ttft_warm_probe.py: loaded in the served process via PYTHONPATH.

Samples every thread's stack every 2 ms into a ring; SIGUSR1 writes it to $TTFT_PROBE_OUT.
Dormant unless TTFT_PROBE_OUT is set.
"""

import json
import os
import signal
import sys
import threading
import time
from collections import deque

_OUT = os.environ.get("TTFT_PROBE_OUT")
if _OUT:
    _ring = deque(maxlen=400000)
    import importlib.metadata

    _gc_policy = os.environ.get("TTFT_PROBE_COPY_INDEX_GC", "auto")
    if _gc_policy not in ("auto", "defer"):
        raise ValueError("TTFT_PROBE_COPY_INDEX_GC must be auto or defer")
    if _gc_policy == "defer":
        from gc_policy import defer_index_gc

        from yunshu_engine.copy_drafter import CopyDrafter

        CopyDrafter._sync_index = defer_index_gc(CopyDrafter._sync_index)
    print(
        f"TTFT probe copy-index GC policy engaged: {_gc_policy}",
        file=sys.stderr,
        flush=True,
    )
    with open(f"{os.path.dirname(_OUT)}/runtime.{os.getpid()}.json", "w") as f:
        json.dump(
            {
                **{
                    key: importlib.metadata.version(key)
                    for key in ("mlx", "mlx-lm", "mlx-vlm")
                },
                "copy_index_gc": _gc_policy,
            },
            f,
        )
    _calls = None
    if os.environ.get("TTFT_PROBE_CALLS") == "1":
        import gc

        import mlx.core as mx
        from spans import CallSpans

        _calls = CallSpans()
        _calls.wrap_native(mx)
        gc.callbacks.append(_calls.gc_event)
        sys.setprofile(_calls.profile)
        threading.setprofile(_calls.profile)

    def _stack(frame, depth=7):
        out = []
        while frame is not None and len(out) < depth:
            c = frame.f_code
            out.append(
                f"{os.path.basename(c.co_filename)}:{c.co_name}:{frame.f_lineno}"
            )
            frame = frame.f_back
        return out

    def _loop():
        me = threading.get_ident()
        names = {}
        while True:
            t = time.perf_counter()
            for tid, fr in sys._current_frames().items():
                if tid == me:
                    continue
                if tid not in names:
                    names[tid] = next(
                        (th.name for th in threading.enumerate() if th.ident == tid),
                        str(tid),
                    )
                _ring.append((t, names[tid], _stack(fr)))
            time.sleep(0.002)

    def _dump(*_):
        with open(f"{_OUT}.{os.getpid()}", "w") as f:
            for r in list(_ring):
                f.write(json.dumps(r) + "\n")
        _ring.clear()
        if _calls is not None:
            with open(f"{_OUT}.spans.{os.getpid()}", "w") as f:
                for r in list(_calls.rows):
                    f.write(json.dumps(r) + "\n")
            _calls.rows.clear()

    signal.signal(signal.SIGUSR1, _dump)
    if os.environ.get("TTFT_PROBE_STACKS", "1") != "0":
        threading.Thread(target=_loop, daemon=True, name="ttft-probe-sampler").start()
