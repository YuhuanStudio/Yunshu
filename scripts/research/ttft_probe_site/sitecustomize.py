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

    signal.signal(signal.SIGUSR1, _dump)
    threading.Thread(target=_loop, daemon=True, name="ttft-probe-sampler").start()
