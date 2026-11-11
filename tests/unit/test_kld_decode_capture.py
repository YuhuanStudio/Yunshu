"""The yv kld decode capture: teacher-forced T=1 steps through the round driver's decode forward."""

import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tests.unit.test_round_driver import _tiny  # noqa: E402


@pytest.fixture
def capture(monkeypatch, tmp_path):
    from scripts.verify.kld import capture as cap

    from yunshu_engine.kernels import ragged_attention as ra
    from yunshu_engine.round_driver import driver as rd

    for obj, name in (
        (rd.RoundDriver, "_logits"),
        (rd.RoundDriver, "_draw"),
        (ra, "ragged_decode_attention"),
    ):
        monkeypatch.setattr(obj, name, getattr(obj, name))
    return cap, tmp_path


def _tap(cap, tmp_path, forced, calls=None):
    expect = tmp_path / "_current.json"
    expect.write_text(json.dumps(dict(wid="w0", forced=forced)))
    return cap.DecodeTap(expect, tmp_path, calls or [0]), expect


def _logits(seed, vocab=64):
    return mx.array(np.random.default_rng(seed).normal(size=(1, vocab)), mx.float32)


def test_tap_forces_tokens_and_writes_window(capture):
    cap, out = capture
    forced = [5, 6, 7, 8]
    tap, expect = _tap(cap, out, forced)
    rows = [_logits(i) for i in range(4)]
    got = [int(tap.on_draw(["p"], rows[0], mx.array([1])).item())]
    got += [int(tap.on_draw(["d"], r, mx.array([1])).item()) for r in rows[1:]]
    assert got == forced
    z = np.load(out / "w0.npz")
    assert z["tokens"].tolist() == [5, 6, 7]  # the inputs of the three decode steps
    assert z["ids"].shape == (3, cap.K) and z["lp"].shape == (3, cap.K)
    lp = np.array(rows[1] - mx.logsumexp(rows[1], axis=-1, keepdims=True))[0]
    assert np.isclose(
        z["next_lp"][0], lp[6], atol=1e-5
    )  # logprob of the forced follower
    assert np.isclose(z["lp"][0, 0], lp.max(), atol=1e-5)
    assert not expect.exists()


def test_tap_wide_window_writes_nothing(capture):
    """A speculative window (T > 1) is not a decode step: the window is bad and never written."""
    cap, out = capture
    tap, _ = _tap(cap, out, [5, 6, 7])
    tap.on_draw(["p"], _logits(0), mx.array([1]))
    tap.on_draw(["d"], mx.tile(_logits(1), (2, 1)), mx.array([1, 2]))
    tap.on_draw(["d"], _logits(2), mx.array([1]))
    assert tap.bad and not (out / "w0.npz").exists()
    tap, _ = _tap(cap, out, [5, 6, 7])
    tap.on_draw(["many", "many"], None, mx.array([1]))
    assert tap.bad


def test_tap_idle_without_expect_file(capture):
    cap, out = capture
    tap = cap.DecodeTap(out / "_current.json", out, [0])
    tok = mx.array([9])
    assert tap.on_draw(["d"], _logits(0), tok) is tok


def test_topk_record_next_lp_matches_numpy(capture):
    cap, _ = capture
    lg = _logits(3, 100)
    ids, lp, nxt = cap.topk_record(lg, 17)
    ref = np.array(lg)[0]
    ref = ref - (np.log(np.exp(ref - ref.max()).sum()) + ref.max())
    assert ids.tolist() == np.argsort(-ref)[: cap.K].tolist()
    assert np.allclose(np.array(lp), np.sort(ref)[::-1][: cap.K], atol=1e-5)
    assert np.isclose(nxt, ref[17], atol=1e-5)


def test_reference_decode_attend_matches_loop(capture):
    cap, _ = capture
    rng = np.random.default_rng(0)
    S, HKV, CAP, D, H = 3, 1, 16, 8, 2
    k = mx.array(rng.normal(size=(S, HKV, CAP, D)), mx.float32)
    v = mx.array(rng.normal(size=(S, HKV, CAP, D)), mx.float32)
    q = mx.array(rng.normal(size=(2, H, 1, D)), mx.float32)
    lengths, slots = mx.array([5, 9], mx.int32), mx.array([2, 0], mx.int32)
    calls = [0]
    out = cap.reference_decode_attend(q, k, v, lengths, 0.3, slots, calls)
    assert calls == [1]
    for b in range(2):
        n = int(lengths[b])
        kk, vv = np.array(k[int(slots[b]), 0, :n]), np.array(v[int(slots[b]), 0, :n])
        for h in range(H):
            s = np.array(q[b, h, 0]) @ kk.T * 0.3
            p = np.exp(s - s.max())
            want = (p / p.sum()) @ vv
            assert np.allclose(np.array(out[b, h, 0]), want, atol=1e-4)


@pytest.fixture(scope="module")
def tiny():
    from yunshu_engine.kernels.ragged_attention import tile_ready

    if not mx.metal.is_available() or not tile_ready():
        pytest.skip("needs M5-class tensor ops")
    return _tiny()[0]


@pytest.mark.parametrize("arm", ["cand", "ref"])
def test_driver_decode_is_teacher_forced(tiny, capture, arm):
    """Through the real RoundDriver: the emitted tokens are the forced ones, one position is recorded per
    decode step, and a plain forward over the same tokens gives the same next-token logprobs."""
    from yunshu_engine.round_driver import driver as rd
    from yunshu_engine.round_driver import forward as fwd

    cap, out = capture
    prefix = [(7 * i + 11) % 500 for i in range(200)]
    forced = [
        (13 * i + 5) % 500 for i in range(9)
    ]  # an arbitrary continuation, not the greedy one
    calls = [0]
    expect = out / "_current.json"
    expect.write_text(json.dumps(dict(wid="w0", forced=forced)))
    cap.install_decode(arm, expect, out, calls)

    d = rd.RoundDriver(tiny, stop_tokens=set(), chunk=256)
    d.add(rd.Request(prefix, len(forced), handle=0))
    emitted = []
    while d.busy():
        emitted += [e.token for e in d.step() if e.token is not None]
    assert emitted == forced
    z = np.load(out / "w0.npz")
    assert z["tokens"].tolist() == forced[:-1] and z["ids"].shape == (8, cap.K)
    assert int(z["ref_calls"]) == (0 if arm == "cand" else int(z["ref_calls"]))
    assert arm == "cand" or int(z["ref_calls"]) > 0  # the reference kernel was engaged

    # reference: one full forward over prefix + forced; position len(prefix)+i predicts forced[i+1]
    from yunshu_engine.round_driver.driver import RoundDriver

    spy = []
    inner = rd.lane_forward
    rd.lane_forward = lambda lm, seg, **kw: spy.append(inner(lm, seg, **kw)) or spy[-1]
    try:
        d2 = RoundDriver(tiny, stop_tokens=set(), chunk=4096)
        d2.add(rd.Request(prefix + forced[:-1], 1, handle=1))
        while d2.busy():
            d2.step()
    finally:
        rd.lane_forward = inner
    h = mx.concatenate(spy, axis=0)
    lg = fwd.logits(tiny, h).astype(mx.float32)
    lp = lg - mx.logsumexp(lg, axis=-1, keepdims=True)
    at = len(prefix)
    want = np.array(
        [float(lp[at + i - 1, forced[i]]) for i in range(1, len(forced))], np.float32
    )
    assert np.allclose(z["next_lp"], want, atol=0.15)
