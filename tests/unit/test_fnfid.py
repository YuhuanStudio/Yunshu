"""Fidelity metrics (CPU) and the fp32 reference attention (M5 GPU)."""

import importlib.util
import math
import sys
from pathlib import Path

import pytest

base = Path(__file__).resolve().parents[2] / "scripts/research"
sys.path.insert(0, str(base))
spec = importlib.util.spec_from_file_location("fnfid", base / "fnfid.py")
fid = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fid)


def top(*pairs):
    return [dict(token=t, logprob=math.log(p)) for t, p in pairs]


def test_kl_is_zero_for_equal_and_positive_otherwise():
    a = top(("x", 0.5), ("y", 0.3), ("z", 0.1))
    assert fid.kl_to_reference(fid.dist(a), fid.dist(a)) == pytest.approx(0.0, abs=1e-9)
    b = top(("x", 0.3), ("y", 0.5), ("z", 0.1))
    assert fid.kl_to_reference(fid.dist(a), fid.dist(b)) > 0.05


def test_missing_token_uses_the_arms_smallest_listed_probability():
    ref = fid.dist(top(("x", 0.6), ("y", 0.4)))
    arm = fid.dist(top(("x", 0.9), ("w", 0.1)))
    assert fid.kl_to_reference(ref, arm) > fid.kl_to_reference(
        ref, fid.dist(top(("x", 0.6), ("y", 0.4)))
    )


def test_agreement():
    a = top(*[(c, 0.1) for c in "abcde"])
    assert fid.agreement(a, a) == (True, 1.0)
    b = top(*[(c, 0.1) for c in "abcxy"])
    assert fid.agreement(a, b) == (True, 0.6)
    assert fid.agreement(a, top(*[(c, 0.1) for c in "bacde"]))[0] is False


def row(i, group, t):
    return dict(id=i, group=group, top=t)


def test_report_says_cand_no_worse_only_when_it_is_not_further_from_the_reference():
    ref = [row(i, "g", top(("x", 0.6), ("y", 0.3), ("z", 0.1))) for i in range(8)]
    near = [row(i, "g", top(("x", 0.58), ("y", 0.31), ("z", 0.11))) for i in range(8)]
    far = [row(i, "g", top(("y", 0.6), ("x", 0.3), ("z", 0.1))) for i in range(8)]
    assert fid.report(far, near, ref)["groups"]["all"]["cand_no_worse"]
    out = fid.report(near, far, ref)["groups"]["all"]
    assert (
        not out["cand_no_worse"]
        and out["main"]["top1"] == 1.0
        and out["cand"]["top1"] == 0.0
    )
    with pytest.raises(RuntimeError):
        fid.report([], near, ref)


def test_reference_attention_matches_a_numpy_reference():
    import mlx.core as mx

    from yunshu_engine.kernels import qsa_sparse_prefill as qsp

    if not (mx.metal.is_available() and qsp.ready()):
        pytest.skip("needs an M5-class GPU")
    sys.path.insert(0, str(Path(__file__).parent))
    import test_qsa_sparse_prefill as k

    scale = k.D**-0.5
    q, kk, v, blocks, ends = k.make(rows=70, keys=1500, seed=5)
    got = fid.reference_attend(q, kk, v, blocks, ends, scale, rows_per_chunk=32)
    ref = k.reference(q, kk, v, blocks, ends, scale)
    assert abs(got.astype(mx.float32).__array__() - ref).max() < 2e-2
