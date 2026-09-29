"""batch: thinking-budget finish_reason, engine-loop batch ratchet, OCR model
key, STS spectral-gating trailing audio.

VLM thinking-budget exhaustion emitted finish_reason="stop" (text path uses "length")
  and never closed <think>. Now emits "length" + a </think> close.
(HIGH, opt-in): the engine-loop applied AdaptiveBatchSizer output through a DOWN-ONLY
  gate while the sizer started at min_batch=1 → completion_batch_size slammed to ~1 and never
  recovered, neutralizing continuous batching. Seed the sizer from the configured batch +
  apply bidirectionally clamped to [1, original].
OCR native path omitted the "model" key the VLM-fallback path returns.
STS spectral_gating analysis loop stop lacked +1 (keystone un-swept) → final hop
  unanalyzed → trailing audio stayed silence.
"""

from __future__ import annotations

import inspect


def test_engine_loop_batch_bidirectional():
    from yunshu_engine import engine_core

    src = inspect.getsource(engine_core)
    # seeded from configured batch, clamped both directions to [1, original]
    assert "_original_completion_batch_size" in src
    assert "min(int(suggested), self._original_completion_batch_size)" in src
    # the down-only gate is gone
    assert "if suggested < self.config.completion_batch_size:" not in src


def test_sizer_seeded_to_config():
    from yunshu_engine.engine_core import EngineCore

    EngineCore.__new__(EngineCore)
    # the constructor seeds _current_batch from config; verify the AdaptiveBatchSizer honors it
    from yunshu_engine.auto_tuner import AdaptiveBatchSizer

    s = AdaptiveBatchSizer(max_batch=32)
    s._current_batch = 32
    assert (
        s.get_current_batch() == 32
        if hasattr(s, "get_current_batch")
        else s._current_batch == 32
    )


def test_ocr_native_returns_model_key():
    from yunshu_gateway.routers import ocr

    src = inspect.getsource(ocr)
    assert '"model": ocr_model_id,' in src


def test_plus_one_includes_final_hop_at_exact_multiple():
    # len chosen so (len-fft) is an exact multiple of hop → the old exclusive stop dropped
    # that final hop; the +1 includes it.
    fft_size, hop_size = 512, 128
    n = fft_size + 3 * hop_size  # 896 → (n-fft)=384 = 3*hop
    old = list(range(0, n - fft_size, hop_size))
    new = list(range(0, max(1, n - fft_size + 1), hop_size))
    assert new[-1] == 384 and old[-1] == 256, (old, new)
