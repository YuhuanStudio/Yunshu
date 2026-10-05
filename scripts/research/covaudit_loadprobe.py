"""Does stock mlx-vlm load this checkpoint? Prints the full traceback of a failure and stops after
`--seconds` (a successful load of a big model is not needed to answer; fail closed on timeout).

    python covaudit_loadprobe.py --model PATH [--seconds 150]
"""

from __future__ import annotations

import argparse
import faulthandler
import os
import sys
import time
import traceback


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--seconds", type=int, default=150)
    a = ap.parse_args(argv)
    faulthandler.dump_traceback_later(a.seconds, exit=True)
    t0 = time.monotonic()
    try:
        from mlx_vlm import load

        model, _proc = load(a.model, lazy=True)
        print(
            f"RESULT PASS stock mlx-vlm loaded {a.model} (lazy) in {time.monotonic() - t0:.0f}s"
        )
        os._exit(0)
    except BaseException:
        traceback.print_exc()
        print("RESULT FAIL stock mlx-vlm could not load the checkpoint")
        sys.stdout.flush()
        os._exit(1)


if __name__ == "__main__":
    sys.exit(main())
