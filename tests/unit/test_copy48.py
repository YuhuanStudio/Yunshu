"""48-row DFlash copy rounds: caps, default, MTP kept at 32, load-time warm-up of the wide window."""

import pytest

pytest.importorskip("mlx.core")
import mlx.core as mx  # noqa: E402

from yunshu_engine import dflash_fast as fast  # noqa: E402
from yunshu_engine import mtp_lane, settings  # noqa: E402
from yunshu_engine import tree_verify as tv  # noqa: E402
from yunshu_engine.kernels import ragged_attention  # noqa: E402


def test_default_and_caps():
    assert settings.REGISTRY["YUNSHU_SPEC_COPY_ROWS"].default == 48
    assert ragged_attention.MAX_WINDOW == 48
    assert tv.FAST_MAX_ROWS == tv.FUSED_ROWS == 48
    assert tv.MAX_ROWS == 32 and mtp_lane.CHAIN_MAX_ROWS == 32


def test_mtp_and_chain_verify_stay_at_32_rows(monkeypatch):
    monkeypatch.setattr(ragged_attention, "tile_ready", lambda: True)
    assert mtp_lane.verify_max_rows(True) == 32
    assert mtp_lane.verify_max_rows(True, fast=True) == 48
    # the MTP engine path sets the process copy rows through the chain limit
    assert mtp_lane.set_copy_rows(48, mtp_lane.verify_max_rows(True)) == 32


def test_dflash_configure_takes_48(monkeypatch):
    from yunshu_engine import dflash_copy

    monkeypatch.setattr(ragged_attention, "tile_ready", lambda: True)
    monkeypatch.setattr(dflash_copy, "install", lambda: True)
    rows, enabled = dflash_copy.configure(
        None, 48, invariant=True, lane_projections=True
    )
    assert (rows, enabled) == (48, True)
    assert mtp_lane._STATE["copy_rows"] == 48


def test_warm_verify_covers_the_48_row_window(monkeypatch):
    ran = []

    class Cache:
        state = mx.zeros((1,))

    class LM:
        def make_cache(self):
            return [Cache()]

        def __call__(self, tokens, cache=None):
            return None

    class Res:
        hidden = mx.zeros((1,))

    def forward(lm, toks, shape, cache):
        ran.append((toks.shape[1], shape.width, shape.is_chain))
        return Res()

    monkeypatch.setattr(tv, "lane_ready", lambda lm, cache: True)
    monkeypatch.setattr(tv, "tree_forward", forward)
    monkeypatch.setattr(tv, "tree_abort", lambda cache, res: None)
    assert fast.warm_verify(LM(), 48) == [16, 32, 48]
    assert ran == [(16, 16, True), (32, 32, True), (48, 48, True)]
    ran.clear()
    assert fast.warm_verify(LM(), 32) == [16, 32]
    assert fast.warm_verify(LM(), 8) == []
