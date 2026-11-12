"""Stock ``quantized_matmul`` row-invariance canary and call-site inventory.

Stock MLX picks a different Metal kernel by row count M (qmv, qmv_wide, qmv_mma,
qmm), so row r of an M-row call can differ from the one-row call on the same
row. Upstream mlx 0.32.4 (the pin) routes affine M=2..12 to qmv_wide; mlx HEAD
(ba8c005c1, c6abf0dbb) routes affine M=3..16 to qmv_mma. The two guards below
catch that at an mlx upgrade without anyone reading the mlx diff:

* ``test_stock_qmm_rowinv_matches_golden``: GPU. Per projection shape and bit
  width, counts the elements of rows 0..M-1 of an M-row stock call that differ
  from the one-row call, M = 1..16. Must equal the golden recorded on the pin.
  Regenerate on a new mlx only after the spec and identity gates are re-run:
  ``python tests/unit/test_stock_qmm_rowinv.py --write``.
* ``test_stock_quantized_matmul_callsites_are_classified``: CPU. Every
  ``quantized_matmul`` call in ``python/yunshu_engine`` is listed with its file,
  count and why it is or is not on a decode/verify path.
"""

from __future__ import annotations

import ast
import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
GOLDEN = ROOT / "tests" / "unit" / "data" / "qmm_stock_rowinv_golden.json"
# (K, N): Qwen3.5-27B-class projections (hidden 5120, MLP 17408), down and up.
SHAPES = ((5120, 5120), (5120, 17408), (17408, 5120))
BITS = (4, 8)
MS = tuple(range(1, 17))

# file (relative to python/yunshu_engine) -> (calls, why it is safe for identity)
STOCK_CALLSITES: dict[str, tuple[int, str]] = {
    "ane/ffn_split.py": (
        1,
        "ANE split FFN: lossy opt-in option, not the identity path",
    ),
    "dflash_draft.py": (
        1,
        "drafter projection: changes acceptance only, target verify decides tokens",
    ),
    "draft_vocab.py": (1, "draft head readout: acceptance only"),
    "kernels/int_code_linear.py": (
        1,
        "rows > lane MAX_ROWS (prefill) or one row outside the invariant lane",
    ),
    "blocks/linear.py": (
        2,
        "fallback outside the lane envelope; see lane_linear eligibility",
    ),
    "kernels/omlx/qwen35_gdn_prework.py": (
        1,
        "B1/T1 decode fused-projection fallback: M=1 only",
    ),
    "kernels/omlx_q4/qwen35_gdn_prework.py": (
        2,
        "B1/T1 decode fallback (M=1) and prefill-side fallback",
    ),
    "kernels/omlx_q4/row_exact_qmv.py": (1, "self-check reference: one-row stock call"),
    "kernels/tensorfold/lane_qmm.py": (
        2,
        "self-check reference and rows > MAX_ROWS (128) fallback",
    ),
}


def _inventory() -> dict[str, int]:
    found: dict[str, int] = {}
    for path in sorted((ROOT / "python" / "yunshu_engine").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        n = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                fn = node.func
                name = (
                    fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
                )
                n += name == "quantized_matmul"
        if n:
            found[str(path.relative_to(ROOT / "python" / "yunshu_engine"))] = n
    return found


def test_stock_quantized_matmul_callsites_are_classified() -> None:
    found = _inventory()
    expected = {k: v[0] for k, v in STOCK_CALLSITES.items()}
    assert found == expected, (
        "stock quantized_matmul call sites changed; classify each new one in "
        f"STOCK_CALLSITES (is it on a decode/verify path with M in 3..16?): found={found}"
    )


def _observe(mx, K: int, N: int, bits: int) -> dict[int, int]:
    mx.random.seed(0)
    w = mx.random.normal((N, K)) * 0.05
    wq, s, b = mx.quantize(w, group_size=64, bits=bits)
    x = mx.random.normal((16, K)).astype(mx.bfloat16)
    one = [
        mx.quantized_matmul(
            x[r : r + 1], wq, s, b, transpose=True, group_size=64, bits=bits
        )
        for r in range(16)
    ]
    out: dict[int, int] = {}
    for m in MS:
        y = mx.quantized_matmul(
            x[:m], wq, s, b, transpose=True, group_size=64, bits=bits
        )
        mx.eval(y, *one)
        out[m] = sum(int(mx.sum(y[r : r + 1] != one[r]).item()) for r in range(m))
    return out


def _table(mx) -> dict[str, dict[str, int]]:
    return {
        f"K{K}N{N}b{bits}": {str(m): c for m, c in _observe(mx, K, N, bits).items()}
        for K, N in SHAPES
        for bits in BITS
    }


def _arch(mx) -> str:
    return str(mx.metal.device_info().get("architecture", ""))


def test_stock_qmm_rowinv_matches_golden() -> None:
    mx = pytest.importorskip("mlx.core")
    if not mx.metal.is_available():
        pytest.skip("no Metal device")
    arch = _arch(mx)
    golden = json.loads(GOLDEN.read_text(encoding="utf-8")) if GOLDEN.exists() else {}
    if arch not in golden:
        # the golden is per architecture (M5: applegpu_g17s); another device has nothing to compare with
        pytest.skip(
            f"no stock-qmm golden for {arch!r} (goldens exist for {sorted(golden)})"
        )
    got = _table(mx)
    changed = {
        key: {
            m: (golden[arch][key].get(m), got[key][m])
            for m in got[key]
            if got[key][m] != golden[arch][key].get(m)
        }
        for key in got
    }
    changed = {k: v for k, v in changed.items() if v}
    assert not changed, (
        f"stock quantized_matmul row-invariance changed on {arch} (key -> {{M: (golden, now)}}): {changed}. "
        "A new mlx routes these rows to other kernels; re-verify the spec and identity gates before updating the golden."
    )


if __name__ == "__main__" and "--write" in sys.argv:
    import mlx.core as mx

    data = json.loads(GOLDEN.read_text(encoding="utf-8")) if GOLDEN.exists() else {}
    data[_arch(mx)] = _table(mx)
    GOLDEN.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN.write_text(
        json.dumps(data, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("wrote", GOLDEN, "arch", _arch(mx))
