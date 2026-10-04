"""Joining physical dispatches retains native mixer spans and cache contents."""

import importlib.util
from pathlib import Path

import pytest

mx = pytest.importorskip("mlx.core")
from mlx_vlm.models.cache import ArraysCache, KVCache  # noqa: E402

from tests.unit.test_native_model_cache_probe import _model  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "span_forward_probe",
    Path(__file__).resolve().parents[2] / "scripts/research/span_forward.py",
)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def test_join_only_short_nonstored_boundaries_within_lane_limit():
    kept, removed = probe.joined_boundaries([100, 379, 381], 100, {381})
    assert kept == [100, 381] and removed == [379]
    assert probe.joined_boundaries([100, 700, 702], 100, {702})[1] == []
    assert probe.joined_boundaries([100, 379, 381], 100, {379, 381})[1] == []
    assert probe.joined_boundaries([100, 370, 381], 100, {381})[1] == []


@pytest.mark.parametrize("preserve_descriptors", [False, True])
@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
@pytest.mark.parametrize("prefix,tail", [(15, 2), (66, 2)])
def test_real_qwen_keeps_hidden_and_native_cache_bits(
    prefix, tail, dtype, preserve_descriptors
):
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    uninstall = None
    token = None
    try:
        mx.random.seed(123)
        model = _model()
        model.set_dtype(dtype)
        baseline, candidate = ([ArraysCache(2), KVCache()], [ArraysCache(2), KVCache()])
        prime = mx.array([[1, 2, 3, 4, 5, 6, 7, 8]])
        mx.eval(model(prime, cache=baseline), model(prime, cache=candidate))
        suffix = mx.arange(prefix + tail).reshape(1, -1) % 32
        first = model(suffix[:, :prefix], cache=baseline)
        last = model(suffix[:, prefix:], cache=baseline)
        mx.eval(first, last, [c.state for c in baseline])
        counts, uninstall = probe.install(preserve_descriptors=preserve_descriptors)
        token = probe._PLAN.set((8 + prefix,))
        actual = model(suffix, cache=candidate)
        mx.eval(actual, [c.state for c in candidate])
        assert counts["forwards"] == 1
        assert mx.array_equal(mx.concatenate([first, last], axis=1), actual).item()
        assert candidate[1].offset == baseline[1].offset
        for left, right in zip(baseline, candidate, strict=True):
            for a, b in zip(left.state, right.state, strict=True):
                assert mx.array_equal(a, b).item()
    finally:
        if token is not None:
            probe._PLAN.reset(token)
        if uninstall:
            uninstall()
        mx.set_default_device(previous)


def test_batch_join_keeps_store_points_and_rejects_padding_and_explicit_policy(
    monkeypatch,
):
    from types import SimpleNamespace

    from mlx_vlm.generate.ar import PromptProcessingBatch
    from mlx_vlm.models.cache import BatchKVCache

    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    uninstall = None
    observed = []
    try:
        coordinator = SimpleNamespace(
            is_checkpoint=True,
            manager=SimpleNamespace(prefill_stride=2048),
            _store_lengths={381},
            request=lambda ids: None,
        )
        kv = BatchKVCache([0])
        kv.keys, kv.values = (
            mx.zeros((1, 1, 512, 4), mx.bfloat16),
            mx.zeros((1, 1, 512, 4), mx.bfloat16),
        )

        def initialize(self, padding=0):
            self._apc_coordinator = coordinator
            self.uids = [1]
            self._right_pad_per_row = [padding]
            self._apc_meta = [
                {
                    "full_input_ids": list(range(382)),
                    "prefix_len": 100,
                    "checkpoint_lengths": [100, 379, 381],
                }
            ]
            self.prompt_cache = [ArraysCache(2), kv]

        def step(self):
            observed.append(probe._PLAN.get())
            return 1

        monkeypatch.setattr(PromptProcessingBatch, "__init__", initialize)
        monkeypatch.setattr(PromptProcessingBatch, "prompt_step", step)
        counts, uninstall = probe.install()
        batch = PromptProcessingBatch()
        assert batch._apc_meta[0]["checkpoint_lengths"] == [100, 381]
        batch.prompt_step()
        assert observed == [(379,)] and probe._PLAN.get() == ()
        padded = PromptProcessingBatch(padding=1)
        assert padded._apc_meta[0]["checkpoint_lengths"] == [100, 379, 381]
        coordinator.request = lambda ids: {"points": [(379, 300)]}
        explicit = PromptProcessingBatch()
        assert explicit._apc_meta[0]["checkpoint_lengths"] == [100, 379, 381]
        assert counts["joins"] == 1
    finally:
        if uninstall:
            uninstall()
        mx.set_default_device(previous)


def test_span_installer_is_idempotent():
    from mlx_vlm.models.qwen3_5.language import Qwen3_5Model

    first, uninstall = probe.install()
    installed = Qwen3_5Model.__call__
    try:
        second, same_uninstall = probe.install()
        assert first is second and uninstall is same_uninstall
        assert Qwen3_5Model.__call__ is installed
    finally:
        uninstall()


def test_descriptor_interleave_uses_each_original_projection_shape(monkeypatch):
    import mlx.nn as nn
    from mlx_vlm.models.qwen3_5 import language as q

    from yunshu_engine.kernels import lane_linear

    calls = []

    class FakeLane(nn.Module):
        def __init__(self, name, width):
            super().__init__()
            self.name = name
            self.width = width

        def __call__(self, x):
            calls.append((self.name, int(x.shape[1])))
            return mx.ones((*x.shape[:-1], self.width), dtype=x.dtype)

    monkeypatch.setattr(lane_linear, "LaneLinear", FakeLane)
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    undo = None
    span_token = None
    try:
        mlp = q.Qwen3_5MLP(16, 32)
        mlp.gate_proj, mlp.up_proj, mlp.down_proj = (
            FakeLane("gate", 32),
            FakeLane("up", 32),
            FakeLane("down", 16),
        )
        counts, undo = probe.install(preserve_descriptors=True)
        span_token = probe._SPANS.set(((0, 279), (279, 281)))
        x = mx.ones((1, 281, 16), dtype=mx.bfloat16)
        result = mlp(x)
        mx.eval(result)
        assert result.shape == x.shape
        assert counts["preserve_descriptors"]
        assert calls == [
            ("gate", 279),
            ("gate", 2),
            ("up", 279),
            ("up", 2),
            ("down", 279),
            ("down", 2),
        ]
    finally:
        if span_token is not None:
            probe._SPANS.reset(span_token)
        if undo:
            undo()
        mx.set_default_device(previous)
