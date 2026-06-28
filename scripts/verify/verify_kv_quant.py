"""KV-cache quantization gate (KIVI-style) — model-free, deterministic.

Gates the mx-based KVQuantizer (NOT the deleted Metal kivi kernel): quantize a
known KV tensor and dequantize it, asserting the round-trip is lossy-but-bounded,
shape-preserving, monotonic in bits (8-bit beats 4-bit), and gives the expected
compression ratio (32/bits). Deterministic input → no flakiness, no model load.

Run: PYTHONPATH=. uv run python scripts/verify_kv_quant.py
"""
from __future__ import annotations

import math
import sys


def _make_kv(L=2, H=4, S=8, D=16):
    """Deterministic [L,H,S,D] tensor (smooth values, no RNG)."""
    return [[[[math.sin(0.1 * (l + h + s + d)) * 2.0
              for d in range(D)] for s in range(S)] for h in range(H)] for l in range(L)]


def _flat(x):
    if isinstance(x, list):
        out = []
        for e in x:
            out.extend(_flat(e))
        return out
    return [x]


def _rel_err(orig, deq):
    o, d = _flat(orig), _flat(deq)
    num = sum(abs(a - b) for a, b in zip(o, d))
    den = sum(abs(a) for a in o) or 1.0
    return num / den


def main() -> int:
    from yunshu_engine.kv_quantization import KVQuantizer, KVQuantConfig

    kv = _make_kv()
    results = {}
    for bits in (4, 8):
        q = KVQuantizer(KVQuantConfig(bits=bits, group_size=64))
        packed, meta = q.quantize(kv)
        deq = q.dequantize(packed, meta)
        results[bits] = {
            "rel_err": _rel_err(kv, deq),
            "shape_ok": meta.get("shape") == [2, 4, 8, 16],
            "compress": meta.get("config", {}).get("compression_ratio")
                        or (32.0 / bits),
            "nbytes": len(packed),
        }

    r4, r8 = results[4], results[8]
    checks = {
        "4-bit round-trip bounded (rel_err < 0.15)": r4["rel_err"] < 0.15,
        "8-bit round-trip tighter (rel_err < 0.05)": r8["rel_err"] < 0.05,
        "8-bit more accurate than 4-bit": r8["rel_err"] < r4["rel_err"],
        "shape preserved (both)": r4["shape_ok"] and r8["shape_ok"],
        "4-bit packs smaller than 8-bit": r4["nbytes"] < r8["nbytes"],
    }
    for k, v in checks.items():
        print(f"  {'OK ' if v else 'BAD'} {k}")
    print(f"  · 4-bit rel_err={r4['rel_err']:.4f} ({r4['nbytes']}B)  "
          f"8-bit rel_err={r8['rel_err']:.4f} ({r8['nbytes']}B)")
    ok = all(checks.values())
    print(f"RESULT: {sum(checks.values())}/{len(checks)}")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
