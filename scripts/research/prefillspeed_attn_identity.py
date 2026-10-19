"""Bit-identity of kernels.attn_prefill.attend vs mx.fast.scaled_dot_product_attention (GPU: via gpuq only).

    prefillspeed_attn_identity.py --out F.jsonl [--dry-run]

Shapes mirror 27B prefill chunks (24 q heads / 4 kv heads / D256), including unaligned rows and keys,
transposed (non-contiguous) queries and keys/values sliced from a larger cache buffer. Rows below
``MIN_ROWS`` are reported for information only (stock uses other kernels there). Any mismatch at or
above ``MIN_ROWS`` exits 1 at once; the file ends with a complete record only on success.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

ROWS = [1024, 1025, 1500, 1984, 2048, 2051, 3000, 4096]
OFFSETS = [0, 32, 33, 2048, 8199, 30000]
INFO_ROWS = [64, 256, 512, 1000, 1023]
H, HKV, D = 24, 4, 256


def cases(quick: bool = False) -> list[tuple[int, int, str, bool]]:
    """(rows, keys, data, sliced) tuples."""
    out = []
    for rows, off in itertools.product(ROWS, OFFSETS):
        out.append((rows, rows + off, "normal", off % 2 == 0))
    out += [(2048, 2048 + 8192, "peaky", True), (2051, 2051 + 6000, "heavy", True)]
    out += [(r, r + 4096, "normal", False) for r in INFO_ROWS]
    return out[:6] if quick else out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)
    if a.dry_run:
        print(json.dumps({"cases": len(cases(a.quick))}))
        return 0
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
    import mlx.core as mx

    from yunshu_engine.kernels import attn_prefill as ap

    a.out.parent.mkdir(parents=True, exist_ok=True)
    mx.random.seed(1)
    n = 0
    with a.out.open("w") as f:

        def emit(rec: dict) -> None:
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print(json.dumps(rec), flush=True)

        emit(
            dict(
                type="engagement",
                device=mx.device_info().get("device_name"),
                mlx=mx.__version__,
                supported=ap.supported_device(),
                arithmetic=ap.arithmetic_id(),
            )
        )
        if not ap.supported_device():
            return 1
        for rows, keys, data, sliced in cases(a.quick):
            scale = {"normal": 0.5, "peaky": 3.0, "heavy": 1.0}[data]
            qraw = mx.random.normal((1, rows, H, D)) * scale
            if data == "heavy":
                qraw = qraw * mx.exp(mx.random.normal(qraw.shape))
            q = qraw.astype(mx.bfloat16).transpose(
                0, 2, 1, 3
            )  # non-contiguous like the model
            cap = keys + 256 if sliced else keys
            kb = (mx.random.normal((1, HKV, cap, D)) * scale).astype(mx.bfloat16)
            vb = (mx.random.normal((1, HKV, cap, D)) * 0.5).astype(mx.bfloat16)
            k, v = kb[:, :, :keys], vb[:, :, :keys]
            mx.eval(q, k, v)
            ref = mx.fast.scaled_dot_product_attention(
                q, k, v, scale=D**-0.5, mask="causal"
            )
            out = ap.attend(q, k, v)
            mx.eval(ref, out)
            equal = bool(mx.array_equal(ref, out).item())
            finite = bool(mx.all(mx.isfinite(out.astype(mx.float32))).item())
            n += 1
            emit(
                dict(
                    type="case",
                    rows=rows,
                    keys=keys,
                    data=data,
                    sliced=sliced,
                    equal=equal,
                    finite=finite,
                    info=rows < ap.MIN_ROWS,
                )
            )
            if rows >= ap.MIN_ROWS and not (equal and finite):
                emit(dict(type="complete", success=False, records=n))
                return 1
        emit(dict(type="complete", success=True, records=n))
    return 0


if __name__ == "__main__":
    sys.exit(main())
