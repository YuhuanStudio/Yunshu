"""VM fallbacks must never classify a physical Apple GPU as paravirtual."""

from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from yunshu_engine.keyed_sampling import KeyedSampler
from yunshu_engine.utils import hardware
from yunshu_engine.vlm_batch_runner import RowParams
from yunshu_engine.vlm_engine import VLMEngine


@pytest.mark.parametrize(
    "name,arch,expected",
    [
        ("Apple Paravirtual device", "air64_v26", True),
        ("Apple Paravirtual device", "air64_v27", True),
        ("Apple M1", "applegpu_g13g", False),
        ("Apple M3 Max", "applegpu_g15s", False),
        ("Apple M5 Max", "applegpu_g17s", False),
        ("Apple M1", "air64_v26", False),
        ("Apple Paravirtual device", "applegpu_g13g", False),
    ],
)
def test_identification_requires_both_vm_identifiers(monkeypatch, name, arch, expected):
    hardware.is_paravirtual_metal.cache_clear()
    monkeypatch.setattr(
        mx, "device_info", lambda *_: dict(device_name=name, architecture=arch)
    )
    try:
        assert hardware.is_paravirtual_metal() is expected
    finally:
        hardware.is_paravirtual_metal.cache_clear()


def test_virtual_runner_uses_cold_stock_path(monkeypatch):
    monkeypatch.setattr(hardware, "is_paravirtual_metal", lambda: True)
    eng = VLMEngine.__new__(VLMEngine)
    eng._model = SimpleNamespace(language_model=object())
    eng._processor = object()
    eng._executor = None
    eng._get_eos_ids = lambda: [2]
    eng._active_count = 1
    # No config/weights/caches: attempting any optimized setup would fail.
    runner = eng._build_batch_runner("/models/qwen3_5")
    assert runner.model is eng._model
    assert runner.apc_manager is None and runner.drafter is None
    assert runner.driver is None and runner.ragged_kv is None
    assert not runner.prefix_invariant and not eng._prefix_invariant_dispatch
    assert runner.stop_tokens == {2} and runner.inflight() == 1


def test_virtual_draws_bound_graphs_and_preserve_seed_positions(monkeypatch):
    monkeypatch.setattr(hardware, "is_paravirtual_metal", lambda: False)
    params = RowParams(temperature=1, top_p=1, top_k=0, min_p=0)
    lp = mx.broadcast_to(mx.array([[0.0, -0.5, -1.0, -mx.inf]]), (137, 4))
    positions = mx.arange(137) + 23
    expected = np.array(KeyedSampler(params, 7).sample_positions(lp, positions))
    original = KeyedSampler.sample_positions
    sizes = []

    def traced(self, rows, pos):
        sizes.append(int(pos.size))
        return original(self, rows, pos)

    monkeypatch.setattr(KeyedSampler, "sample_positions", traced)
    monkeypatch.setattr(hardware, "is_paravirtual_metal", lambda: True)
    sampler = KeyedSampler(params, 7)
    got = np.array(sampler.sample_positions(lp, positions))
    assert sizes == [64, 64, 9]
    assert np.array_equal(expected, got)
    assert got.max() < 3
