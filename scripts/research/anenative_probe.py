"""Native ANE pilot. All runtime use (including availability) goes through gpuq."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from yunshu_engine.ane.native import NativeSession
from yunshu_engine.ane.native.mil import mlp_program


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--rows", type=int, default=128)
    p.add_argument("--mode", choices=["fp16", "int8"], default="fp16")
    return p


def main():
    a = parser().parse_args()
    rng = np.random.default_rng(10)
    g, u, d = [rng.normal(0, 0.05, (64, 64)).astype(np.float16) for _ in range(3)]
    x = rng.normal(0, 0.5, (64, a.rows)).astype(np.float16)
    mil, blob = mlp_program(g, u, d, a.rows, a.mode)
    session, reason = NativeSession.try_create(
        mil, blob, [(64, a.rows)], (64, a.rows), a.cache
    )
    if session is None:
        print(json.dumps({"available": False, "reason": reason}), flush=True)
        return 1
    with session:
        session.copy_input(x)
        ms = session.run(3)
        if ms is None:
            raise RuntimeError(session.reason)
        y = session.copy_output()
        gate = g.astype(np.float32) @ x.astype(np.float32)
        ref = d.astype(np.float32) @ (
            (gate / (1 + np.exp(-gate))) * (u.astype(np.float32) @ x.astype(np.float32))
        )
        rel = float(np.linalg.norm(y.astype(np.float32) - ref) / np.linalg.norm(ref))
        assert np.isfinite(y).all() and rel < 0.05, rel
        print(
            json.dumps({"available": True, "ms": ms, "rel_l2": rel, "complete": True}),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
