"""(HIGH): the NON-streaming native video pipeline re-opened W939 — it
returned HTTP 200 with placeholder garbage.

W939 fixed the streaming path + tagged the pipeline's placeholder method as
"..._placeholder_fallback" so the router's `"fallback" in method` 503 guard trips. But:
- BUG 1 (HIGH): _run_generation rebuilt VideoGenOutput with a HARDCODED
  method="native_mlx", DISCARDING the pipeline's fallback tag → router 200 with garbage.
- BUG 2 (MED): _generate_with_native_pipeline never refused a non-callable model (it only
  nulled on load failure), so it generated placeholder frames AND left the pipeline set;
  a later streaming request then found self._native_pipeline is not None and SKIPPED the
  W939 callability guard (nested in `if is None`) → streamed garbage as 200.

Fix (W1010): refuse a non-callable transformer in _generate_with_native_pipeline (mirror
the streaming W939 guard) so the non-streaming caller falls through to the honest fallback
(router 503); re-check callability on EVERY call in BOTH paths (not just on creation); and
propagate result.method as a router-level safety net.
"""
from __future__ import annotations

import inspect
import types

from yunshu_engine.video_engine import VideoEngine


def test_generate_native_refuses_noncallable_model_and_nulls_pipeline():
    eng = VideoEngine.__new__(VideoEngine)
    eng._teacache_config = None
    eng._running = True
    # a non-callable _model (the W939 weights-dict shape) on an already-set pipeline
    fake = types.SimpleNamespace(_model={"w": 1}, cleanup=lambda: None)
    eng._native_pipeline = fake
    out = eng._generate_with_native_pipeline(
        prompt="p", negative_prompt="", image=None, width=64, height=64,
        num_frames=4, num_steps=2, guide_scale=1.0, seed=1, scheduler="unipc",
        model_dir="/tmp/whatever",
    )
    # refused before generating placeholder frames → None → caller falls to fallback (503)
    assert out is None
    # pipeline nulled so a later streaming request re-evaluates (Bug 2 trigger removed)
    assert eng._native_pipeline is None


def test_generate_native_callability_check_outside_is_none_block():
    # Bug 2 root: the non-streaming creator must check callability on EVERY call, not
    # only when it creates the pipeline.
    src = inspect.getsource(VideoEngine._generate_with_native_pipeline)
    code = "\n".join(ln.split("#", 1)[0] for ln in src.splitlines())
    assert 'callable(getattr(self._native_pipeline, "_model", None))' in code


def test_stream_native_callability_check_hoisted():
    # the streaming guard must no longer be nested inside `if self._native_pipeline is None`
    src = inspect.getsource(VideoEngine._stream_native_pipeline)
    lines = src.splitlines()
    none_idx = next(i for i, ln in enumerate(lines) if "self._native_pipeline is None" in ln)
    call_idx = next(i for i, ln in enumerate(lines)
                    if 'not callable(getattr(self._native_pipeline, "_model"' in ln)
    # the callability check sits AFTER (outside) the is-None creation block, at column 8
    assert call_idx > none_idx
    assert lines[call_idx].startswith("        if not callable")  # 8-space indent = method body


def test_run_generation_propagates_pipeline_method():
    # Bug 1: _run_generation must propagate result.method, not hardcode "native_mlx".
    src = inspect.getsource(VideoEngine._run_generation)
    code = "\n".join(ln.split("#", 1)[0] for ln in src.splitlines())
    assert 'getattr(result, "method", None) or "native_mlx"' in code
    # the bare hardcoded native_mlx (discarding the fallback tag) is gone
    assert 'method="native_mlx",' not in code
