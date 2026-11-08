"""Regression tests for call-site/signature contracts (model-free).

These bugs were latent because the bad calls sit inside try/except blocks
that swallowed the TypeError, silently disabling the feature. Each test
pins the contract so it can't regress without a unit-test failure. No model
is loaded.
"""

from __future__ import annotations

import inspect

import pytest


def test_inflight_register_accepts_engine_kwargs():
    # batched_engine.py calls register(req_id, token_ids=..., kv_cache=...).
    from yunshu_engine.inflight_prefix_sharing import InflightPrefixTracker

    params = inspect.signature(InflightPrefixTracker.register).parameters
    assert "kv_cache" in params
    assert "kv_cache_ref" not in params  # the old wrong name
    # The exact engine call (batched_engine.py:7234) must not raise and must
    # store the entry. (An entry with kv_cache=None is intentionally not
    # matchable by find_prefix until a KV ref is attached.)
    tracker = InflightPrefixTracker(max_entries=4)
    tracker.register("rid-1", token_ids=[1, 2, 3], kv_cache=None)
    assert len(tracker._entries) == 1


def test_bench_batch_accepts_prompts_not_num_requests():
    # bench.py router calls runner.bench_batch(prompts=<int>, ...).
    from yunshu_engine.benchmark import BenchmarkRunner

    params = inspect.signature(BenchmarkRunner.bench_batch).parameters
    assert "prompts" in params
    assert "num_requests" not in params


def test_lora_linear_uses_r_not_rank():
    # image_engine.py / video_engine.py construct LoRALinear(..., r=<int>).
    from mlx_lm.tuner.lora import LoRALinear

    params = inspect.signature(LoRALinear.__init__).parameters
    assert "r" in params
    assert "rank" not in params


def test_preprocessor_preprocess_accepts_tokenizer_kwarg():
    # engine_core / batched_engine call preprocess(prompt, tokenizer=...).
    # The ABC signature is (raw_input, **kwargs), so a tokenizer keyword must
    # be accepted (lands in **kwargs) without raising.
    from yunshu_engine.model_preprocessor import Qwen3ASRPreprocessor

    pp = Qwen3ASRPreprocessor()
    # Should not raise TypeError on the keyword; concrete impls ignore it.
    sig = inspect.signature(pp.preprocess)
    has_var_kw = any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )
    assert has_var_kw, "preprocess must accept **kwargs for the tokenizer keyword"


def test_kv_manager_save_guards_unbound_tensors():
    # save_prefix / save_all_cached must raise a clear error (not a cryptic
    # NoneType subscript) when KV tensors were never bound.
    import yunshu_kv.manager as mgr

    src = inspect.getsource(mgr)
    # Both save methods now contain an explicit None-guard.
    assert "call set_kv_tensors() before save_prefix()" in src
    assert "call set_kv_tensors() before save_all_cached()" in src


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
