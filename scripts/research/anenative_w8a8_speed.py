"""ANE program-variant speed probe (random weights): fp16 vs int8-dequantize vs W8A8, segments.

Speed only (no accuracy claim). Run under gpuq; the first variant failing stops the run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from yunshu_engine.ane.native import NativeSession
from yunshu_engine.ane.native.mil import mlp_program, w8a8_program


def variants(hidden, channels, rows, rng):
    g = rng.normal(0, 0.02, (channels, hidden)).astype(np.float32)
    u = rng.normal(0, 0.02, (channels, hidden)).astype(np.float32)
    d = rng.normal(0, 0.02, (hidden, channels)).astype(np.float32)
    yield "fp16", mlp_program(g, u, d, rows, "fp16", 6)
    yield "w8-const", mlp_program(g, u, d, rows, "int8", 6)
    yield "w8a8-seg2560", w8a8_program(g, u, d, rows, 2560, True)
    yield "w8a8-seg5120", w8a8_program(g, u, d, rows, 5120, True)
    yield "w8only-dq-seg2560", w8a8_program(g, u, d, rows, 2560, False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rows", default="1024")
    p.add_argument("--channels", default="2176,4352,8704")
    p.add_argument("--hidden", type=int, default=5120)
    p.add_argument("--reps", type=int, default=7)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    rng = np.random.default_rng(0)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w") as out:
        for rows in map(int, a.rows.split(",")):
            for ch in map(int, a.channels.split(",")):
                for name, (mil, blob) in variants(a.hidden, ch, rows, rng):
                    s, why = NativeSession.try_create(
                        mil, blob, [(a.hidden, rows)], (a.hidden, rows), a.cache
                    )
                    rec = {"variant": name, "rows": rows, "channels": ch}
                    if s is None:
                        rec["error"] = why
                    else:
                        with s:
                            s.copy_input(
                                rng.normal(0, 0.7, (a.hidden, rows)).astype(np.float16)
                            )
                            s.run()
                            ms = [s.run() for _ in range(a.reps)]
                            y = s.copy_output()
                            rec.update(
                                ms=ms,
                                median=float(np.median(ms)),
                                finite=bool(np.isfinite(y).all()),
                            )
                    out.write(json.dumps(rec) + "\n")
                    out.flush()
                    print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
