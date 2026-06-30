"""Video-generation option exposure: VAE-decode `tiling` and LTX `fps`.

Covers two fixes:
1. /v1/video/generations exposes a `tiling` option (validated against the
   mlx-video backends' accepted modes) and threads it to generate_video.
2. The LTX branch forwards `fps` (and `tiling`) — but only kwargs the
   introspected backend signature actually accepts, so a missing/older backend
   never gets an unsupported kwarg (which would TypeError every request).

These exercise the plumbing/gating with FAKE mlx_video modules; the real
diffusion run still needs a live model.
"""

from __future__ import annotations

import sys
import types

import pytest


# ── request-model validation ────────────────────────────────────────────────
def test_request_tiling_default_and_normalization():
    from yunshu_gateway.routers.video import VideoGenerateRequest

    assert VideoGenerateRequest(prompt="x").tiling == "auto"
    # case-insensitive + whitespace tolerant
    assert VideoGenerateRequest(prompt="x", tiling="  Aggressive ").tiling == (
        "aggressive"
    )


def test_request_tiling_rejects_unknown():
    from pydantic import ValidationError

    from yunshu_gateway.routers.video import VideoGenerateRequest

    with pytest.raises(ValidationError):
        VideoGenerateRequest(prompt="x", tiling="ginormous")


# ── per-backend kwarg gating ─────────────────────────────────────────────────
def _install_fake_mlx_video(monkeypatch, *, wan_gen=None, ltx_gen=None):
    """Inject a minimal fake mlx_video package tree into sys.modules."""

    def _mod(name):
        m = types.ModuleType(name)
        monkeypatch.setitem(sys.modules, name, m)
        return m

    root = _mod("mlx_video")
    models = _mod("mlx_video.models")
    root.models = models

    if wan_gen is not None:
        wan2 = _mod("mlx_video.models.wan_2")
        gen = _mod("mlx_video.models.wan_2.generate")
        gen.generate_video = wan_gen
        # Pre-mark so _generate_with_mlx_video skips the one-time T5 bf16 patch
        # (that path imports mlx + a real T5Encoder we don't fake here).
        gen._yunshu_t5_bf16 = True
        wan2.generate = gen
        models.wan_2 = wan2

    if ltx_gen is not None:
        ltx2 = _mod("mlx_video.models.ltx_2")
        gen = _mod("mlx_video.models.ltx_2.generate")
        gen.generate_video = ltx_gen
        ltx2.generate = gen
        models.ltx_2 = ltx2


def _make_engine(model_type):
    from yunshu_engine.video_engine import VideoEngine

    eng = VideoEngine()
    eng._model_type = model_type
    return eng


def test_wan_branch_forwards_tiling_not_fps(monkeypatch):
    captured = {}

    def fake_wan(
        *,
        model_dir,
        prompt,
        negative_prompt,
        image,
        width,
        height,
        num_frames,
        steps,
        guide_scale,
        seed,
        output_path,
        scheduler,
        tiling="auto",
    ):
        captured.update(locals())
        return None

    _install_fake_mlx_video(monkeypatch, wan_gen=fake_wan)
    eng = _make_engine("wan_2_2")
    out = eng._generate_with_mlx_video(
        model_dir="/m",
        prompt="p",
        negative_prompt=None,
        image_path=None,
        width=256,
        height=256,
        num_frames=5,
        steps=4,
        guide_scale=5.0,
        seed=1,
        scheduler="unipc",
        tiling="aggressive",
        fps=16,
    )
    import os

    if out and os.path.exists(out):
        os.unlink(out)
    assert captured["tiling"] == "aggressive"
    # Wan has no fps param — it must never be forwarded.
    assert "fps" not in captured


def test_wan_branch_omits_tiling_when_unsupported(monkeypatch):
    """An older backend without a `tiling` param must not receive it."""
    captured = {}

    def fake_wan_no_tiling(
        *,
        model_dir,
        prompt,
        negative_prompt,
        image,
        width,
        height,
        num_frames,
        steps,
        guide_scale,
        seed,
        output_path,
        scheduler,
    ):
        captured.update(locals())
        return None

    _install_fake_mlx_video(monkeypatch, wan_gen=fake_wan_no_tiling)
    eng = _make_engine("wan_2_2")
    # Must not raise TypeError despite passing tiling at the Yunshu layer.
    out = eng._generate_with_mlx_video(
        model_dir="/m",
        prompt="p",
        negative_prompt=None,
        image_path=None,
        width=256,
        height=256,
        num_frames=5,
        steps=4,
        guide_scale=5.0,
        seed=1,
        scheduler="unipc",
        tiling="spatial",
        fps=16,
    )
    import os

    if out and os.path.exists(out):
        os.unlink(out)
    assert "tiling" not in captured


def test_ltx_branch_forwards_fps_and_tiling_not_scheduler(monkeypatch):
    captured = {}

    def fake_ltx(
        *,
        model_repo,
        text_encoder_repo,
        prompt,
        negative_prompt,
        width,
        height,
        num_frames,
        num_inference_steps,
        cfg_scale,
        seed,
        output_path,
        image,
        fps=24,
        tiling="auto",
    ):
        captured.update(locals())
        return None

    _install_fake_mlx_video(monkeypatch, ltx_gen=fake_ltx)
    eng = _make_engine("ltx_2")
    out = eng._generate_with_mlx_video(
        model_dir="/m",
        prompt="p",
        negative_prompt=None,
        image_path=None,
        width=256,
        height=256,
        num_frames=9,
        steps=8,
        guide_scale=4.0,
        seed=7,
        scheduler="unipc",
        tiling="conservative",
        fps=24,
    )
    import os

    if out and os.path.exists(out):
        os.unlink(out)
    assert captured["fps"] == 24
    assert captured["tiling"] == "conservative"
    # LTX has no scheduler param — must never be forwarded (would TypeError).
    assert "scheduler" not in captured
