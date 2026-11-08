"""The yv kld capture hooks follow the round driver's prefill (lane_forward segments)."""

import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tests.unit.test_round_driver import _tiny  # noqa: E402


@pytest.fixture(scope="module")
def tiny():
    from yunshu_engine.kernels.ragged_attention import tile_ready

    if not mx.metal.is_available() or not tile_ready():
        pytest.skip("needs M5-class tensor ops")
    return _tiny()[0]


@pytest.fixture
def hooks(monkeypatch, tmp_path):
    from mlx_vlm.generate.ar import PromptProcessingBatch
    from mlx_vlm.models.qwen3_5 import language as l35
    from scripts.verify.kld import capture

    from yunshu_engine.round_driver import driver as rd
    from yunshu_engine.round_driver import flashnext as fn

    monkeypatch.setenv("KLD_CACHE_DIR", str(tmp_path))
    for obj, name in (
        (rd, "lane_forward"),
        (fn, "prefill"),  # install_driver wraps it too: never leave a wrapper behind
        (rd.RoundDriver, "_prefill_step"),
        (PromptProcessingBatch, "generate"),
        (l35.LanguageModel, "__call__"),
        (l35.Qwen3_5Model, "__call__"),
    ):
        monkeypatch.setattr(obj, name, getattr(obj, name))
    return capture, tmp_path


def _prefill(lm, ids, chunk=256):
    from yunshu_engine.round_driver.driver import Request, RoundDriver

    d = RoundDriver(lm, stop_tokens=set(), chunk=chunk)
    d.add(Request(ids, 1, handle=0))
    n = 0
    while d.busy():
        n += len(d.step())
    return d, n


def test_capture_follows_driver_prefill(tiny, hooks):
    from yunshu_engine.round_driver import driver as rd
    from yunshu_engine.round_driver import forward as fwd

    capture, out = hooks
    ids = [(7 * i + 11) % 500 for i in range(700)]  # several prefill segments
    hidden = []
    inner = rd.lane_forward

    def spy(lm, seg, **kw):
        h = inner(lm, seg, **kw)
        hidden.append(h)
        return h

    rd.lane_forward = spy
    capture.install("cand", "qwen3_5")
    (out / "_current.json").write_text(json.dumps(dict(wid="w0")))
    _prefill(tiny, ids)
    got = np.load(out / "w0.npz")
    assert len(hidden) > 1
    assert got["tokens"].tolist() == ids
    assert got["lp"].shape == (len(ids), capture.K)
    h = mx.concatenate(hidden, axis=0)
    logits = fwd.logits(tiny, h).astype(mx.float32)
    lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    want = mx.take_along_axis(lp, mx.array(ids[1:] + [0])[:, None], axis=-1)[:, 0]
    assert np.allclose(got["next_lp"][:-1], np.array(want)[:-1], atol=2e-2)
    assert np.isnan(got["next_lp"][-1])
    top = np.array(mx.sort(lp, axis=-1)[:, -1])
    assert np.allclose(got["lp"][:, 0], top, atol=2e-2)
    assert not (out / "_current.json").exists()


def test_restored_prefix_writes_nothing(tiny, hooks, monkeypatch):
    """A prefill that does not start at position 0 fails closed (no partial window)."""
    from yunshu_engine.round_driver import driver as rd

    capture, out = hooks
    capture.install("cand", "qwen3_5")
    inner = rd.lane_forward
    seen = []

    def shifted(lm, seg, **kw):
        seen.append(seg)
        return inner(lm, seg, **kw)

    ids = [(5 * i + 3) % 500 for i in range(300)]
    from yunshu_engine.round_driver.driver import Request, RoundDriver

    d = RoundDriver(tiny, stop_tokens=set(), chunk=256)
    d.add(Request(ids, 1, handle=0))
    # pretend the first segment continues a restored prefix: its cache already holds tokens
    first = d.rows[0]
    for layer in first.cache:
        if isinstance(getattr(layer, "offset", None), int):
            layer.offset = 0
    monkeypatch.setattr(capture, "_cache_offset", lambda cache: 64)
    (out / "_current.json").write_text(json.dumps(dict(wid="w1")))
    while d.busy():
        d.step()
    assert not (out / "w1.npz").exists()


def test_flash_next_prefill_is_hooked(monkeypatch, tmp_path):
    """Flash-Next prompts prefill through flashnext.prefill, not lane_forward: the hook must follow it
    (before the fix every Flash-Next window failed with 'the capture hook wrote no window')."""
    from types import SimpleNamespace

    from scripts.verify.kld import capture

    from yunshu_engine.round_driver import driver as rd
    from yunshu_engine.round_driver import flashnext as fn

    for obj, name in (
        (rd, "lane_forward"),
        (rd.RoundDriver, "_prefill_step"),
        (fn, "prefill"),
    ):
        monkeypatch.setattr(obj, name, getattr(obj, name))

    def inner_prefill(lm, seg, **kw):
        # the real one calls the language model, whose body hook must not record a second time
        for hook in body_hooks:
            hook()
        return np.zeros((3, 4))

    body_hooks = []
    monkeypatch.setattr(fn, "prefill", inner_prefill)
    monkeypatch.setattr(fn, "logits", lambda lm, h: np.zeros((h.shape[0], 2)))
    cur = tmp_path / "_current.json"
    cur.write_text("{}")
    state = {"tokens": [], "active": False, "bad": False}
    seen = []

    def record(inputs, hidden, head=None):
        seen.append((inputs.shape, hidden.shape, head(hidden)))
        state["tokens"].extend([0] * inputs.shape[1])

    body_hooks.append(lambda: None if state.get("in_driver") else seen.append("dup"))
    capture.install_driver(state, record, lambda: None, lambda: None, cur)
    seg = SimpleNamespace(
        tokens=np.zeros(3), cache=[SimpleNamespace(offset=0)], embeds=None
    )
    fn.prefill(object(), seg)
    assert state["active"] and not state["bad"]
    assert "dup" not in seen
    assert seen and seen[0][0] == (1, 3) and seen[0][2].shape == (1, 3, 2)
    fn.prefill(
        object(), seg
    )  # offset 0 again but 3 tokens recorded: not the continuation
    assert state["bad"]
