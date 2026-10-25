"""Native MLP bit oracle for program-row sizes, 64-atoms, padding and column shifts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from yunshu_engine.ane import convert
from yunshu_engine.ane.native import NativeSession
from yunshu_engine.ane.native.mil import mlp_program


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model", default="/Volumes/P5Plus/models/Jundot/Qwen3.8-27B-oQ4e-mtp"
    )
    p.add_argument("--layer", type=int, default=40)
    p.add_argument("--channels", type=int, default=2176)
    p.add_argument("--mode", choices=["fp16", "int8"], default="fp16")
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    return p


def bit_difference(left, right):
    if (
        left.shape != right.shape
        or left.dtype != np.float16
        or right.dtype != np.float16
    ):
        raise ValueError("matching fp16 shapes required")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("nonfinite native MLP output")
    return int(np.count_nonzero(left.view(np.uint16) != right.view(np.uint16)))


def main():
    a = parser().parse_args()
    raw = convert.load_mlp_layer(a.model, a.layer)
    inter = raw["gate_proj"]["weight"].shape[0]
    if a.channels <= 0 or a.channels >= inter or a.channels % 64:
        raise ValueError("proper64-aligned slice required")
    weights = convert.mlp_slices(raw, inter - a.channels, a.mode)
    hidden = weights[0].shape[1]
    x = np.random.default_rng(2).normal(0, 0.7, (hidden, 2048)).astype(np.float16)
    outputs = {}
    sessions = {}
    checks = []
    try:
        for rows in [64, 128, 512, 1024, 2048]:
            mil, blob = mlp_program(*weights, rows, a.mode, 6)
            session, reason = NativeSession.try_create(
                mil, blob, [(hidden, rows)], (hidden, rows), a.cache
            )
            if session is None:
                raise RuntimeError(reason)
            sessions[rows] = session
            session.copy_input(x[:, :rows])
            if session.run() is None:
                raise RuntimeError(session.reason)
            outputs[rows] = session.copy_output()
        for rows in [128, 512, 1024, 2048]:
            checks.append(
                {
                    "kind": "program_rows",
                    "rows": rows,
                    "different_bits": bit_difference(
                        outputs[64], outputs[rows][:, :64]
                    ),
                }
            )
        for rows in [64, 128]:
            session = sessions[rows]
            for valid in [1, 31, 63] if rows == 64 else [64]:
                padded = np.zeros((hidden, rows), dtype=np.float16)
                padded[:, :valid] = x[:, :valid]
                session.copy_input(padded)
                if session.run() is None:
                    raise RuntimeError(session.reason)
                checks.append(
                    {
                        "kind": "padding",
                        "rows": rows,
                        "valid": valid,
                        "different_bits": bit_difference(
                            outputs[rows][:, :valid], session.copy_output()[:, :valid]
                        ),
                    }
                )
        translated = np.zeros((hidden, 128), dtype=np.float16)
        translated[:, :64] = x[:, 64:128]
        sessions[128].copy_input(translated)
        if sessions[128].run() is None:
            raise RuntimeError(sessions[128].reason)
        checks.append(
            {
                "kind": "column_shift64",
                "different_bits": bit_difference(
                    outputs[128][:, 64:128], sessions[128].copy_output()[:, :64]
                ),
            }
        )
        for tile in [64, 128]:
            blocks = []
            for begin in range(0, 1024, tile):
                sessions[tile].copy_input(x[:, begin : begin + tile])
                if sessions[tile].run() is None:
                    raise RuntimeError(sessions[tile].reason)
                blocks.append(sessions[tile].copy_output())
            checks.append(
                {
                    "kind": "tiled_vs_full",
                    "tile": tile,
                    "rows": 1024,
                    "different_bits": bit_difference(
                        outputs[1024], np.concatenate(blocks, axis=1)
                    ),
                }
            )
        result = {
            "checks": checks,
            "variable_rows_admissible": all(c["different_bits"] == 0 for c in checks),
            "canonical64_padding_admissible": all(
                c["different_bits"] == 0
                for c in checks
                if c["kind"] == "padding" and c["rows"] == 64
            ),
            "complete": True,
            "layer": a.layer,
            "channels": a.channels,
            "mode": a.mode,
        }
        a.out.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
        return 0 if result["canonical64_padding_admissible"] else 1
    finally:
        for session in sessions.values():
            session.close()


if __name__ == "__main__":
    raise SystemExit(main())
